
from analysis.srt_parser import parse_srt, srt_file_to_plain_text
import os
from dotenv import load_dotenv
from utils.text_cleaning import normalize_text, strip_srt_tags, words



load_dotenv()

REF_SRT = os.getenv("REF_SRT")
HYP_SRT = os.getenv("HYP_SRT")

# --- SRT parser
print("[SRT parser] Test de parsing de fichier SRT")

with open(REF_SRT, "r", encoding="utf-8") as f:
    content = f.read()

segments = parse_srt(content)
print(f"[SRT parser] {len(segments)} segments parsés")
print(f"Premier segment : {segments[0]}")

text = srt_file_to_plain_text(content)
print(f"\n[Plain text] {len(text.split())} mots")
print(f"  Début : {text[:100]}")

# --- Text cleaning 
print("[Text cleaning] Test de nettoyage de texte")

sample = "Bonjour, comment ça va ?!"
cleaned = normalize_text(strip_srt_tags(sample))
w = words(sample)

print(f"[Text cleaning] Input  : {sample}")
print(f"[Text cleaning] Output : {cleaned}")
print(f"[Text cleaning] Mots   : {w}")
print("[Text cleaning] OK\n")


