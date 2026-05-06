import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    GLADIA_API_KEY: str = os.getenv("GLADIA_API_KEY", "")
    GLADIA_BASE_URL: str = "https://api.gladia.io"
    GLADIA_POLL_INTERVAL: int = 3
    GLADIA_TIMEOUT: int = int(os.getenv("GLADIA_TIMEOUT", "600"))

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    REF_SRT: str = os.getenv("REF_SRT", "")
    HYP_SRT: str = os.getenv("HYP_SRT", "")
    MEDIA_PATH: str = os.getenv("MEDIA_PATH", "")

    GLADIA_VOCABULARY_PATH: str = os.getenv("GLADIA_VOCABULARY_PATH", "")

    def validate(self) -> list[str]:
        errors = []
        if not self.GLADIA_API_KEY:
            errors.append("GLADIA_API_KEY manquant")
        if self.LLM_PROVIDER == "groq" and not self.GROQ_API_KEY:
            errors.append("GROQ_API_KEY manquant")
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY manquant")
        return errors


config = Config()