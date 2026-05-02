from app.llm.gemini_client import call_gemini
from app.llm.groq_client import call_groq

def call_llm(prompt: str) -> str:
    try:
        return call_gemini(prompt)
    except Exception:
        return call_groq(prompt)