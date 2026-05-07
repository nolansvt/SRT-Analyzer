from dataclasses import dataclass, field
from jiwer import process_words
from utils.text_cleaning import normalize_text, normalize_format, strip_srt_tags


@dataclass
class WERResult:
    wer: float
    mer: float
    wil: float
    ref_word_count: int
    hyp_word_count: int
    substitutions: int
    insertions: int
    deletions: int
    substitution_examples: list[tuple[str, str]] = field(default_factory=list)
    insertion_examples: list[str] = field(default_factory=list)
    deletion_examples: list[str] = field(default_factory=list)


def compute_wer(reference: str, hypothesis: str) -> WERResult:
    ref_clean = normalize_format(normalize_text(strip_srt_tags(reference)))
    hyp_clean = normalize_format(normalize_text(strip_srt_tags(hypothesis)))

    output = process_words(ref_clean, hyp_clean)

    sub_ex, ins_ex, del_ex = [], [], []

    for chunk in output.alignments[0]:
        if chunk.type == "substitute":
            ref_words = output.references[0][chunk.ref_start_idx : chunk.ref_end_idx]
            hyp_words = output.hypotheses[0][chunk.hyp_start_idx : chunk.hyp_end_idx]
            sub_ex.append((" ".join(ref_words), " ".join(hyp_words)))
        elif chunk.type == "insert":
            hyp_words = output.hypotheses[0][chunk.hyp_start_idx : chunk.hyp_end_idx]
            ins_ex.append(" ".join(hyp_words))
        elif chunk.type == "delete":
            ref_words = output.references[0][chunk.ref_start_idx : chunk.ref_end_idx]
            del_ex.append(" ".join(ref_words))

    return WERResult(
        wer=output.wer,
        mer=output.mer,
        wil=output.wil,
        ref_word_count=len(output.references[0]),
        hyp_word_count=len(output.hypotheses[0]),
        substitutions=output.substitutions,
        insertions=output.insertions,
        deletions=output.deletions,
        substitution_examples=sub_ex,
        insertion_examples=ins_ex,
        deletion_examples=del_ex,
    )