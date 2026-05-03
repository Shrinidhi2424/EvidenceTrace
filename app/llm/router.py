import logging

from app.llm.gemini_client import call_gemini
from app.llm.groq_client import call_groq

logger = logging.getLogger(__name__)

# global flag
USE_GEMINI = True


def call_llm(prompt: str) -> str:
    global USE_GEMINI

    if USE_GEMINI:
        try:
            return call_gemini(prompt)
        except Exception as e:
            logger.warning("Gemini failed, switching to Groq: %s", e)
            USE_GEMINI = False

    return call_groq(prompt)
