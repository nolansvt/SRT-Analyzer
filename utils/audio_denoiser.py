import os
import re
import subprocess


PROFILES = {
    "léger": {
        "highpass": 80,
        "afftdn_nf": -20,
        "description": "Bruit léger, voix intérieure",
    },
    "modéré": {
        "highpass": 120,
        "afftdn_nf": -30,
        "description": "Bruit ambiant, légère brise",
    },
    "fort": {
        "highpass": 180,
        "afftdn_nf": -40,
        "description": "Vent soutenu, bruit de fond marqué",
    },
    "extrême": {
        "highpass": 250,
        "afftdn_nf": -50,
        "description": "Vent fort, environnement très bruité",
    },
}


def _measure_rms(wav_path: str, lowpass_hz: int | None = None) -> float:
    af = f"lowpass=f={lowpass_hz},astats=metadata=1" if lowpass_hz else "astats=metadata=1"
    result = subprocess.run(
        ["ffmpeg", "-i", wav_path, "-af", af, "-f", "null", "-"],
        capture_output=True, text=True
    )
    output = result.stderr
    matches = re.findall(r"RMS level dB:\s+([-\d.]+)", output)
    if not matches:
        return -100.0
    values = [float(v) for v in matches if v != "-inf"]
    return sum(values) / len(values) if values else -100.0


def detect_profile(wav_path: str) -> str:
    rms_total = _measure_rms(wav_path)
    rms_bass = _measure_rms(wav_path, lowpass_hz=200)

    if rms_total <= -100 or rms_bass <= -100:
        print(f"[denoiser] Détection impossible, profil par défaut : modéré")
        return "modéré"


    ratio = rms_bass - rms_total
    print(ratio)
    print(f"[denoiser] RMS total={rms_total:.1f} dB | RMS <200Hz={rms_bass:.1f} dB | ratio={ratio:.1f} dB")

    if ratio > -1:
        profile = "extrême"
    elif ratio > -3:
        profile = "fort"
    elif ratio > -10:
        profile = "modéré"
    elif ratio > -18:
        profile = "léger"
    else:
        profile = "Aucun"

    print(f"[denoiser] Profil détecté automatiquement : {profile}")
    return profile


def denoise_audio(wav_path: str, profile: str = "modéré") -> str:
    if profile == "Auto":
        profile = detect_profile(wav_path)
        if profile == "Aucun":
            return wav_path
    params = PROFILES.get(profile, PROFILES["modéré"])
    out_path = os.path.splitext(wav_path)[0] + f"_denoised_{profile}.wav"

    if os.path.exists(out_path):
        print(f"[denoiser] Déjà existant : {out_path}")
        return out_path

    highpass = params["highpass"]
    nf = params["afftdn_nf"]
    filters = f"highpass=f={highpass},afftdn=nf={nf},loudnorm"

    cmd = [
        "ffmpeg", "-y",
        "-i", wav_path,
        "-af", filters,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        out_path,
    ]

    print(f"[denoiser] Profil '{profile}' ({params['description']}) → {os.path.basename(out_path)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Erreur débruitage ffmpeg:\n{result.stderr}")

    original_mb = os.path.getsize(wav_path) / 1e6
    out_mb = os.path.getsize(out_path) / 1e6
    print(f"[denoiser] OK : {original_mb:.1f} MB → {out_mb:.1f} MB")
    return out_path
