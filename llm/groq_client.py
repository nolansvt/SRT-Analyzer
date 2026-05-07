from groq import Groq
from llm.base import LLMClient, TokenUsage
from config import config

PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant":    {"input": 0.05, "output": 0.08},
    "mixtral-8x7b-32768":      {"input": 0.24, "output": 0.24},
}


class GroqClient(LLMClient):
    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model = config.GROQ_MODEL

    def generate(self, prompt: str) -> str:
        print(f"Envoi du prompt à Groq : {prompt[:50]}...")
        text, _ = self.generate_with_usage(prompt)
        return text

    def generate_with_usage(self, prompt: str) -> tuple[str, TokenUsage]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw_usage = response.usage
        if raw_usage is None:
            raise RuntimeError("Aucune donnée d'usage retournée par l'API")

        prices = PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        cost = (raw_usage.prompt_tokens * prices["input"] + raw_usage.completion_tokens * prices["output"]) / 1_000_000

        return response.choices[0].message.content or "", TokenUsage(
            prompt_tokens=raw_usage.prompt_tokens,
            completion_tokens=raw_usage.completion_tokens,
            total_tokens=raw_usage.total_tokens,
            estimated_cost_usd=cost,
        )