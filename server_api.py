import os, json
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

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
    keyword: str
    style: str = "신뢰형"
    purpose: str = "후기형"
    length: str = "보통"
    model: str = "gpt-4.1-mini"

class BulkReq(BaseModel):
    keyword: str
    count: int = 3
    model: str = "gpt-4.1-mini"

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
def generate_single(req: SingleReq):
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
        return {"result": call_openai(req.model, prompt)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-bulk")
def generate_bulk(req: BulkReq):
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
        return {"result": call_openai(req.model, prompt)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server_api:app", host="0.0.0.0", port=8000, reload=False)
