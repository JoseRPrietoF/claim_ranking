from fastapi import FastAPI
from pydantic import BaseModel
from vector_db.db_handler import search_top_k
from models.model import classify_claim
from models.model_openai import get_chatgpt_results

app = FastAPI()

class ClaimRequest(BaseModel):
    claim: str
    api_key: str
    model: str

@app.post("/classify")
def classify_claims(request: ClaimRequest):
    claim = request.claim
    top_k_claims = search_top_k(claim)  # Busca en la base de datos
    predictions = classify_claim(claim, top_k_claims)
    return {"claim": claim, "top_k": predictions}

@app.post("/classifyopenai")
def classify_claims_openai(request: ClaimRequest):
    claim = request.claim
    model_choice = request.model
    api_key = request.api_key
    top_k_claims = search_top_k(claim)  # Busca en la base de datos
    predictions = get_chatgpt_results(claim, top_k_claims, model_choice, api_key)
    return {"claim": claim, "top_k": predictions}
