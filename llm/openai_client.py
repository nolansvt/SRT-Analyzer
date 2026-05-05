from openai import OpenAI
from llm.base import LLMClient, TokenUsage
from config import config

PRICING = {
    "gpt-4o-mini":  {"input": 0.15,  "output": 0.60},
    "gpt-4o":       {"input": 2.50,  "output": 10.00},
    "gpt-4-turbo":  {"input": 10.00, "output": 30.00},
}


class OpenAIClient(LLMClient):
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL

    def generate(self, prompt: str) -> str:
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