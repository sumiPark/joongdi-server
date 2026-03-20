from pathlib import Path
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client


BASE_DIR = Path(__file__).resolve().parent
API_KEY_FILE = BASE_DIR / "api_key.txt"
PRODUCTS_FILE = BASE_DIR / "products.json"
SYSTEM_PROMPT_FILE = BASE_DIR / "system_prompt.txt"

def read_api_key():
    if API_KEY_FILE.exists():
        key = API_KEY_FILE.read_text(encoding="utf-8").strip()
        if key and "paste_new_api_key_here" not in key and "여기에" not in key:
            return key
    return os.getenv("OPENAI_API_KEY", "").strip()

PRODUCTS = json.loads(PRODUCTS_FILE.read_text(encoding="utf-8"))
SYSTEM_PROMPT = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")

def find_product(keyword: str):
    k = keyword.strip().lower()
    for item in PRODUCTS.values():
        for alias in item.get("keywords", []):
            alias_l = alias.lower()
            if alias_l in k or k in alias_l:
                return item
    return None

def get_client():
    key = read_api_key()
    if not key:
        raise RuntimeError("SERVER/api_key.txt에 API 키를 넣어주세요.")
    return OpenAI(api_key=key)

def call_openai(model: str, prompt: str):
    client = get_client()
    res = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return res.output_text

class SingleReq(BaseModel):
    work_title: str = ""
    keyword: str
    style: str = "신뢰형"
    purpose: str = "후기형"
    length: str = "보통"
    model: str = "gpt-4.1-mini"

class BulkReq(BaseModel):
    work_title: str = ""
    keyword: str
    count: int = 3
    model: str = "gpt-4.1-mini"

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_ANON_KEY = os.environ["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

supabase_public: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
supabase_admin: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return authorization.replace("Bearer ", "", 1).strip()


def get_current_user_profile(authorization: Optional[str] = Header(None)):
    token = extract_bearer_token(authorization)

    user_resp = supabase_public.auth.get_user(token)
    user = user_resp.user
    if not user:
        raise HTTPException(status_code=401, detail="유효하지 않은 로그인입니다.")

    profile_resp = (
        supabase_admin.table("profiles")
        .select("*")
        .eq("id", user.id)
        .single()
        .execute()
    )

    profile = profile_resp.data
    if not profile:
        raise HTTPException(status_code=403, detail="프로필이 없습니다.")

    return profile


def require_approved_user(profile=Depends(get_current_user_profile)):
    if profile["status"] != "approved":
        raise HTTPException(status_code=403, detail="승인된 회원만 사용할 수 있습니다.")
    return profile


def require_admin(profile=Depends(get_current_user_profile)):
    if profile["role"] != "admin":
        raise HTTPException(status_code=403, detail="관리자만 접근할 수 있습니다.")
    return profile


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "api_key_loaded": bool(read_api_key())}

@app.post("/generate-single")
def generate_single(req: SingleReq, profile=Depends(require_approved_user)):
    product = find_product(req.keyword)
    product_json = json.dumps(product, ensure_ascii=False, indent=2) if product else "매칭된 상품 데이터 없음"
    prompt = f"""
키워드: {req.keyword}
문체: {req.style}
목적: {req.purpose}
길이: {req.length}

상품 데이터:
{product_json}

요청:
블로그용 완성형 대본 1개를 작성해줘.

형식:
[제목]
[후킹]
[제품/주제 분석]
[좋은점]
[아쉬운점]
[추천 대상]
[마무리]
[썸네일 문구]
"""
    try:
        result = call_openai(req.model, prompt)

        supabase_admin.table("generation_history").insert({
            "user_id": profile["id"],
            "kind": "content",
            "work_title": getattr(req, "work_title", "") or "",
            "keyword": req.keyword,
            "meta": {
                "style": req.style,
                "purpose": req.purpose,
                "length": req.length
            },
            "result": result
        }).execute()

        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-bulk")
def generate_bulk(req: BulkReq, profile=Depends(require_approved_user)):
    if req.count < 1 or req.count > 10:
        raise HTTPException(status_code=400, detail="count는 1 이상 10 이하만 허용합니다.")
    product = find_product(req.keyword)
    product_json = json.dumps(product, ensure_ascii=False, indent=2) if product else "매칭된 상품 데이터 없음"
    prompt = f"""
키워드: {req.keyword}
생성 개수: {req.count}

상품 데이터:
{product_json}

요청:
다중 초안 생성 결과를 정확히 {req.count}개 작성해줘.

규칙:
- 완성형 대본만 나열
- 각 대본은 후킹, 전개, 마무리가 겹치지 않게 작성
- 데이터에 있는 특징/후기/타겟을 적극 반영
- 마지막에 추천 3개만 별도로 정리

형식:
========== 초안 N ==========
[제목]
[후킹]
[제품/주제 분석]
[좋은점]
[아쉬운점]
[추천 대상]
[마무리]

마지막:
========== 추천 ==========
[추천 1]
제목:
추천 이유:
[추천 2]
제목:
추천 이유:
[추천 3]
제목:
추천 이유:
"""
    try:
        result = call_openai(req.model, prompt)

        supabase_admin.table("generation_history").insert({
            "user_id": profile["id"],
            "kind": "ideas",
            "work_title": req.work_title or "",
            "keyword": req.keyword,
            "meta": {
                "count": req.count
            },
            "result": result
        }).execute()

        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server_api:app", host="0.0.0.0", port=8000, reload=False)


@app.get("/admin/pending-users")
def list_pending_users(admin=Depends(require_admin)):
    resp = (
        supabase_admin.table("profiles")
        .select("*")
        .eq("status", "pending")
        .order("created_at", desc=True)
        .execute()
    )
    return {"items": resp.data}


@app.post("/admin/users/{user_id}/approve")
def approve_user(user_id: str, admin=Depends(require_admin)):
    resp = (
        supabase_admin.table("profiles")
        .update({
            "status": "approved",
            "approved_by": admin["id"],
            "approved_at": "now()"
        })
        .eq("id", user_id)
        .execute()
    )
    return {"ok": True, "data": resp.data}


@app.post("/admin/users/{user_id}/reject")
def reject_user(user_id: str, admin=Depends(require_admin)):
    resp = (
        supabase_admin.table("profiles")
        .update({"status": "rejected"})
        .eq("id", user_id)
        .execute()
    )
    return {"ok": True, "data": resp.data}


@app.post("/admin/users/{user_id}/suspend")
def suspend_user(user_id: str, admin=Depends(require_admin)):
    resp = (
        supabase_admin.table("profiles")
        .update({"status": "suspended"})
        .eq("id", user_id)
        .execute()
    )
    return {"ok": True, "data": resp.data}
