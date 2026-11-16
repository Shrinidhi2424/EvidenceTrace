import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")


settings = Settings()
