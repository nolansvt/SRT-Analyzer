from abc import ABC, abstractmethod


class BaseSTTClient(ABC):
    @abstractmethod
    def transcribe(self, file_path: str) -> str:
        pass