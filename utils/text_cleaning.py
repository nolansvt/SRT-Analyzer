import re


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_srt_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def words(text: str) -> list[str]:
    return normalize_text(text).split()