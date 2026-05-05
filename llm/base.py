from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float

    def __str__(self) -> str:
        return (
            f"Tokens : {self.prompt_tokens} input + {self.completion_tokens} output "
            f"= {self.total_tokens} total | Coût estimé : ${self.estimated_cost_usd:.5f}"
        )


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_with_usage(self, prompt: str) -> tuple[str, TokenUsage]:
        pass