from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="EvidenceTrace")

app.include_router(router)


@app.get("/")
def root():
    return {"message": "EvidenceTrace API running"}
