import time
import requests
from transcription.base import BaseSTTClient
from config import config
import mimetypes


class GladiaClient(BaseSTTClient):
    def __init__(self):
        self.api_key = config.GLADIA_API_KEY
        self.base_url = config.GLADIA_BASE_URL
        self.headers = {"x-gladia-key": self.api_key}


    def _upload_file(self, file_path: str) -> str:
        url = f"{self.base_url}/v2/upload"
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"
        with open(file_path, "rb") as f:
            files = {"audio": (file_path.split("/")[-1], f, mime_type)}
            response = requests.post(url, headers=self.headers, files=files, timeout=120)
        if not response.ok:
            print(f"[Gladia upload] Status: {response.status_code}")
            print(f"[Gladia upload] Body: {response.text}")
        response.raise_for_status()
        return response.json()["audio_url"]

    def _request_transcription(self, audio_url: str) -> str:
        url = f"{self.base_url}/v2/transcription"
        payload = {
            "audio_url": audio_url,
            "subtitles": True,
            "subtitles_config": {"formats": ["srt"]},
            "language_config": {"languages": ["fr"], "code_switching": False},
        }
        response = requests.post(url, headers=self.headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["id"]

    def _poll_result(self, transcription_id: str) -> str:
        url = f"{self.base_url}/v2/transcription/{transcription_id}"
        elapsed = 0
        while elapsed < config.GLADIA_TIMEOUT:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            status = data.get("status")
            if status == "done":
                subtitles = data["result"]["transcription"].get("subtitles", [])
                for sub in subtitles:
                    if sub.get("format") == "srt":
                        return sub["content"]
                raise RuntimeError("SRT non trouvé dans la réponse Gladia")
            elif status == "error":
                raise RuntimeError(f"Gladia error: {data.get('error_message', 'unknown')}")
            time.sleep(config.GLADIA_POLL_INTERVAL)
            elapsed += config.GLADIA_POLL_INTERVAL
        raise TimeoutError("Gladia transcription timed out")

    def transcribe(self, file_path: str) -> str:
        audio_url = self._upload_file(file_path)
        transcription_id = self._request_transcription(audio_url)
        return self._poll_result(transcription_id)