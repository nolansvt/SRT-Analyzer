
from analysis.srt_parser import parse_srt, srt_file_to_plain_text
import os
from dotenv import load_dotenv
from utils.text_cleaning import normalize_text, strip_srt_tags, words
from analysis.wer import compute_wer
from analysis.srt_parser import srt_file_to_plain_text as to_text
from config import config
from analysis.comparator import compare_srt

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


# --- WER 

with open(config.REF_SRT, encoding="utf-8") as f:
    ref_content = f.read()
with open(config.HYP_SRT, encoding="utf-8") as f:
    hyp_content = f.read()

wer_result = compute_wer(to_text(ref_content), to_text(hyp_content))

print(f"[WER] Mots ref={wer_result.ref_word_count}  Mots hyp={wer_result.hyp_word_count}")
print(f"  WER={wer_result.wer:.1%}")
print(f"  Sub={wer_result.substitutions}  Ins={wer_result.insertions}  Del={wer_result.deletions}")
if wer_result.substitution_examples:
    print(f"  Ex sub : '{wer_result.substitution_examples[0][0]}' → '{wer_result.substitution_examples[0][1]}'")
assert 0.0 <= wer_result.wer <= 1.0
print("[WER] OK\n")

# --- Comparateur + diff

report = compare_srt(ref_content, hyp_content)

print(f"[Comparateur] WER={report.wer_result.wer:.1%}")
print(f"  Diff HTML ({len(report.diff_html)} chars) : {report.diff_html[:150]}...")
assert "<span" in report.diff_html
print("[Comparateur] OK\n")

