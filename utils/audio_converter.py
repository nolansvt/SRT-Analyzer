import subprocess
import os


def to_wav(file_path: str, sample_rate: int = 16000, channels: int = 1) -> str:
    wav_path = os.path.splitext(file_path)[0] + "_converted.wav"

    if os.path.exists(wav_path):
        print(f"[audio_converter] WAV déjà existant : {wav_path}")
        return wav_path

    cmd = [
        "ffmpeg", "-y",
        "-i", file_path,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-c:a", "pcm_s16le",
        wav_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{result.stderr}")

    print(f"[audio_converter] Converti : {wav_path} ({os.path.getsize(wav_path) / 1e6:.1f} MB)")
    return wav_path