import difflib
from dataclasses import dataclass
from analysis.srt_parser import srt_file_to_plain_text
from analysis.wer import WERResult, compute_wer
from utils.text_cleaning import normalize_text, normalize_format


@dataclass
class ComparisonReport:
    wer_result: WERResult
    reference_text: str
    hypothesis_text: str
    diff_html: str


@dataclass
class FourWayReport:
    wer_gladia: WERResult
    wer_gladia_cv: WERResult
    wer_llm: WERResult
    diff_gladia_html: str
    diff_gladia_cv_html: str
    diff_llm_html: str


def generate_diff_html(reference: str, hypothesis: str) -> str:
    ref_words = normalize_format(normalize_text(reference)).split()
    hyp_words = normalize_format(normalize_text(hypothesis)).split()
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(ref_words[i1:i2]))
        elif tag == "replace":
            parts.append(f'<span style="background:#d6ffd6;padding:2px 4px;border-radius:3px;color:#000">{" ".join(ref_words[i1:i2])}</span>')
            parts.append(f'<span style="background:#ffd6d6;padding:2px 4px;border-radius:3px;color:#000">{" ".join(hyp_words[j1:j2])}</span>')
        elif tag == "delete":
            parts.append(f'<span style="background:#d6ffd6;padding:2px 4px;border-radius:3px;color:#000">{" ".join(ref_words[i1:i2])}</span>')
        elif tag == "insert":
            parts.append(f'<span style="background:#ffd6d6;padding:2px 4px;border-radius:3px;color:#000">{" ".join(hyp_words[j1:j2])}</span>')

    return "<p style='line-height:2;font-family:monospace'>" + " ".join(parts) + "</p>"


def generate_diff_html_multi(ref_texts: list[str], hyp_texts: list[str], labels: list[str]) -> str:
    parts = []
    for i, (ref, hyp) in enumerate(zip(ref_texts, hyp_texts)):
        if len(labels) > 1:
            parts.append(
                f"<div style='margin:16px 0 8px;padding:6px 12px;background:#2a2a2a;border-left:4px solid #888;"
                f"color:#ccc;font-family:monospace;font-size:0.9em'>── {labels[i]} ──</div>"
            )
        parts.append(generate_diff_html(ref, hyp))
    return "".join(parts)


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


def compare_four_way(
    ref_list: list[str],
    gladia_list: list[str],
    gladia_cv_list: list[str],
    llm_list: list[str],
    labels: list[str],
) -> FourWayReport:
    ref_texts = [srt_file_to_plain_text(s) for s in ref_list]
    gladia_texts = [srt_file_to_plain_text(s) for s in gladia_list]
    gladia_cv_texts = [srt_file_to_plain_text(s) for s in gladia_cv_list]
    llm_texts = [srt_file_to_plain_text(s) for s in llm_list]

    ref_combined = " ".join(ref_texts)
    gladia_combined = " ".join(gladia_texts)
    gladia_cv_combined = " ".join(gladia_cv_texts)
    llm_combined = " ".join(llm_texts)

    return FourWayReport(
        wer_gladia=compute_wer(ref_combined, gladia_combined),
        wer_gladia_cv=compute_wer(ref_combined, gladia_cv_combined),
        wer_llm=compute_wer(ref_combined, llm_combined),
        diff_gladia_html=generate_diff_html_multi(ref_texts, gladia_texts, labels),
        diff_gladia_cv_html=generate_diff_html_multi(ref_texts, gladia_cv_texts, labels),
        diff_llm_html=generate_diff_html_multi(ref_texts, llm_texts, labels),
    )