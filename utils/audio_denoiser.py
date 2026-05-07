import os
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


def denoise_audio(wav_path: str, profile: str = "modéré") -> str:
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
