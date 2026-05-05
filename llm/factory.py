from llm.base import LLMClient
from llm.groq_client import GroqClient
from llm.openai_client import OpenAIClient
from config import config


def get_llm_client() -> LLMClient:
    if config.LLM_PROVIDER == "openai":
        return OpenAIClient()
    return GroqClient()