from fastapi import APIRouter, HTTPException

from app.classification.stance_classifier import get_nli_model_name, is_nli_model_loaded
from app.models.schemas import AnalyzeRequest, AnalyzeResponse
from app.pipeline import NoPapersFound, analyze_claim

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok", "nli_model": get_nli_model_name(), "loaded": is_nli_model_loaded()}


@router.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    try:
        return analyze_claim(request.claim, request.max_papers)
    except NoPapersFound as e:
        raise HTTPException(status_code=502, detail=str(e))
