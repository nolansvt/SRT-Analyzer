from openai import OpenAI
from llm.base import LLMClient, TokenUsage
from config import config

PRICING = {
    # GPT-4o
    "gpt-4o-mini":      {"input": 0.15,   "output": 0.60},
    "gpt-4o":           {"input": 2.50,   "output": 10.00},
    "gpt-4-turbo":      {"input": 10.00,  "output": 30.00},
    # GPT-5 family
    "gpt-5":            {"input": 0.625,  "output": 5.00},
    "gpt-5.2":          {"input": 1.75,   "output": 14.00},
    "gpt-5.4":          {"input": 2.50,   "output": 15.00},
    "gpt-5.4-mini":     {"input": 0.75,   "output": 4.50},
    "gpt-5.4-nano":     {"input": 0.20,   "output": 1.25},
    "gpt-5.5":          {"input": 5.00,   "output": 30.00},
    "gpt-5.5-pro":      {"input": 30.00,  "output": 180.00},
}
NO_TEMPERATURE_MODELS = {
        "gpt-5", "gpt-5.2", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
        "gpt-5.5", "gpt-5.5-pro", "o1", "o1-mini", "o3", "o3-mini",
    }

class OpenAIClient(LLMClient):
    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.model = config.OPENAI_MODEL

    def generate(self, prompt: str) -> str:
        print(f"Envoi du prompt à OpenAI : {prompt[:50]}...")
        text, _ = self.generate_with_usage(prompt)
        return text

      

    def generate_with_usage(self, prompt: str) -> tuple[str, TokenUsage]:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.model not in NO_TEMPERATURE_MODELS:
            kwargs["temperature"] = 0.3

        response = self.client.chat.completions.create(**kwargs)
        raw_usage = response.usage
        if raw_usage is None:
            raise RuntimeError("Aucune donnée d'usage retournée par l'API")

        prices = PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        if prices["input"] == 0.0:
            print(f"[WARNING] Modèle '{self.model}' absent du PRICING, coût estimé à 0.")

        cost = (
            raw_usage.prompt_tokens * prices["input"]
            + raw_usage.completion_tokens * prices["output"]
        ) / 1_000_000

        return response.choices[0].message.content or "", TokenUsage(
            prompt_tokens=raw_usage.prompt_tokens,
            completion_tokens=raw_usage.completion_tokens,
            total_tokens=raw_usage.total_tokens,
            estimated_cost_usd=cost,
        )