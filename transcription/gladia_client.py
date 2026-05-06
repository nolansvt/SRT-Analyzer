import os
import time
import mimetypes
import requests
from transcription.base import BaseSTTClient
from utils.audio_converter import to_wav
from config import config


class GladiaClient(BaseSTTClient):
    def __init__(self):
        self.api_key = config.GLADIA_API_KEY
        self.base_url = config.GLADIA_BASE_URL
        self.headers = {"x-gladia-key": self.api_key}

    def _upload_file(self, file_path: str) -> str:
        url = f"{self.base_url}/v2/upload"
        mime_type, _ = mimetypes.guess_type(file_path)
        mime_type = mime_type or "application/octet-stream"
        size_mb = os.path.getsize(file_path) / 1e6
        print(f"[Gladia upload] Envoi {file_path.split('/')[-1]} ({size_mb:.1f} MB)...")
        with open(file_path, "rb") as f:
            files = {"audio": (file_path.split("/")[-1], f, mime_type)}
            response = requests.post(url, headers=self.headers, files=files, timeout=300)
        print(f"[Gladia upload] Status: {response.status_code}")
        if not response.ok:
            print(f"[Gladia upload] Body: {response.text}")
        response.raise_for_status()
        audio_url = response.json()["audio_url"]
        print(f"[Gladia upload] OK → {audio_url}")
        return audio_url

    def _request_transcription(self, audio_url: str, custom_vocabulary: list[str] | None = None) -> str:
        url = f"{self.base_url}/v2/transcription"
        payload = {
            "audio_url": audio_url,
            "language_config": {"languages": ["fr"], "code_switching": False},
        }
        if custom_vocabulary:
            payload["custom_vocabulary"] = custom_vocabulary
        for attempt in range(10):
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            if response.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"[Gladia] 429 rate limit, attente {wait}s (tentative {attempt + 1}/10)...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()["id"]
        raise RuntimeError("Gladia rate limit persistant après 10 tentatives")

    def _seconds_to_srt_time(self, seconds: float) -> str:
        ms = int((seconds % 1) * 1000)
        s = int(seconds) % 60
        m = int(seconds // 60) % 60
        h = int(seconds // 3600)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    def _utterances_to_srt(self, utterances: list) -> str:
        lines = []
        for i, utt in enumerate(utterances, start=1):
            start = self._seconds_to_srt_time(utt["start"])
            end = self._seconds_to_srt_time(utt["end"])
            lines.append(f"{i}\n{start} --> {end}\n{utt['text'].strip()}\n")
        return "\n".join(lines)

    def _poll_result(self, transcription_id: str, on_progress=None) -> str:
        url = f"{self.base_url}/v2/transcription/{transcription_id}"
        elapsed = 0
        while elapsed < config.GLADIA_TIMEOUT:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            status = data.get("status")
            print(f"[Gladia poll] {elapsed}s → status: {status}")
            if status == "done":
                utterances = data["result"]["transcription"].get("utterances", [])
                if not utterances:
                    raise RuntimeError("Aucune utterance dans la réponse Gladia")
                return self._utterances_to_srt(utterances)
            elif status == "error":
                raise RuntimeError(f"Gladia error: {data.get('error_message', 'unknown')}")
            if on_progress:
                on_progress(elapsed)
            time.sleep(config.GLADIA_POLL_INTERVAL)
            elapsed += config.GLADIA_POLL_INTERVAL
        raise TimeoutError("Gladia transcription timed out")

    def transcribe(self, file_path: str, custom_vocabulary: list[str] | None = None, on_progress=None) -> str:
        wav_path = to_wav(file_path)
        audio_url = self._upload_file(wav_path)
        transcription_id = self._request_transcription(audio_url, custom_vocabulary)
        return self._poll_result(transcription_id, on_progress)