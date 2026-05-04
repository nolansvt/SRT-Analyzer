from analysis.srt_parser import parse_srt
import os
from dotenv import load_dotenv

load_dotenv()

REF_SRT = os.getenv("REF_SRT")
HYP_SRT = os.getenv("HYP_SRT")

# --- Test 1 : SRT parser
with open(REF_SRT, "r", encoding="utf-8") as f:
    content = f.read()

segments = parse_srt(content)
print(f"[SRT parser] {len(segments)} segments parsés")
print(f"Premier segment : {segments[0]}")