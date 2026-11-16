import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.classification.stance_classifier import get_nli_model
from app.retrieval.quality_scorer import get_embedding_model

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load both models once at startup so the first request is not slow.
    if os.getenv("SKIP_WARMUP") != "1":
        get_nli_model()
        get_embedding_model()
    yield


app = FastAPI(title="EvidenceTrace", lifespan=lifespan)

app.include_router(router)


@app.get("/")
def root():
    return {"message": "EvidenceTrace API running"}
