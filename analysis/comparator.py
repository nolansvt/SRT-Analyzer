import difflib
import tempfile
import os
from dataclasses import dataclass
from analysis.srt_parser import srt_file_to_plain_text
from analysis.wer import WERResult, compute_wer


@dataclass
class ComparisonReport:
    wer_result: WERResult
    reference_text: str
    hypothesis_text: str
    diff_html: str


@dataclass
class ThreeWayReport:
    wer_gladia: WERResult
    wer_llm: WERResult
    diff_gladia_html: str
    diff_llm_html: str
    reference_text: str
    gladia_text: str
    llm_text: str


def generate_diff_html(reference: str, hypothesis: str) -> str:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
    parts = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(" ".join(ref_words[i1:i2]))
        elif tag == "replace":
            parts.append(f'<span style="background:#ffd6d6;padding:2px 4px;border-radius:3px;color:#000">{" ".join(ref_words[i1:i2])}</span>')
            parts.append(f'<span style="background:#d6ffd6;padding:2px 4px;border-radius:3px;color:#000">{" ".join(hyp_words[j1:j2])}</span>')
        elif tag == "delete":
            parts.append(f'<span style="background:#ffd6d6;padding:2px 4px;border-radius:3px;color:#000">{" ".join(ref_words[i1:i2])}</span>')
        elif tag == "insert":
            parts.append(f'<span style="background:#d6ffd6;padding:2px 4px;border-radius:3px;color:#000">{" ".join(hyp_words[j1:j2])}</span>')

    return "<p style='line-height:2;font-family:monospace'>" + " ".join(parts) + "</p>"


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


def compare_three_way(reference_srt: str, gladia_srt: str, llm_srt: str) -> ThreeWayReport:
    ref_text = srt_file_to_plain_text(reference_srt)
    gladia_text = srt_file_to_plain_text(gladia_srt)
    llm_text = srt_file_to_plain_text(llm_srt)
    return ThreeWayReport(
        wer_gladia=compute_wer(ref_text, gladia_text),
        wer_llm=compute_wer(ref_text, llm_text),
        diff_gladia_html=generate_diff_html(ref_text, gladia_text),
        diff_llm_html=generate_diff_html(ref_text, llm_text),
        reference_text=ref_text,
        gladia_text=gladia_text,
        llm_text=llm_text,
    )