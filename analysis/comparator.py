import difflib
from dataclasses import dataclass
from analysis.srt_parser import srt_file_to_plain_text
from analysis.wer import WERResult, compute_wer


@dataclass
class ComparisonReport:
    wer_result: WERResult
    reference_text: str
    hypothesis_text: str
    diff_html: str


def generate_diff_html(reference: str, hypothesis: str) -> str:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(ref_words[i1:i2]))
        elif tag == "replace":
            parts.append(f'<span style="background:#ffd6d6;padding:2px 4px;border-radius:3px">{" ".join(ref_words[i1:i2])}</span>')
            parts.append(f'<span style="background:#d6ffd6;padding:2px 4px;border-radius:3px">{" ".join(hyp_words[j1:j2])}</span>')
        elif tag == "delete":
            parts.append(f'<span style="background:#ffd6d6;padding:2px 4px;border-radius:3px">{" ".join(ref_words[i1:i2])}</span>')
        elif tag == "insert":
            parts.append(f'<span style="background:#d6ffd6;padding:2px 4px;border-radius:3px">{" ".join(hyp_words[j1:j2])}</span>')

    return "<p style='line-height:2;font-family:monospace;color:black'>" + " ".join(parts) + "</p>"


def compare_srt(reference_srt: str, hypothesis_srt: str) -> ComparisonReport:
    ref_text = srt_file_to_plain_text(reference_srt)
    hyp_text = srt_file_to_plain_text(hypothesis_srt)
    wer_result = compute_wer(ref_text, hyp_text)
    diff_html = generate_diff_html(ref_text, hyp_text)

    return ComparisonReport(
        wer_result=wer_result,
        reference_text=ref_text,
        hypothesis_text=hyp_text,
        diff_html=diff_html,
    )