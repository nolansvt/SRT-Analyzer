import re
from num2words import num2words

SYNONYMS = {
    "ce sont": "c est",
    "ne sont": "n est",
    "ce sont pas": "c est pas",
    "toutes": "tous",
    "toute": "tout",
    "bienvenues": "bienvenus",
    "bienvenue": "bienvenu",
    "etes": "ete",
    "allez": "aller",
}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_srt_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def words(text: str) -> list[str]:
    return normalize_text(text).split()


def normalize_format(text: str) -> str:
    text = re.sub(r"\b(\d+)\s*heures?\b", r"\1h", text)
    text = re.sub(r"\b(\d+)\s*h\s*(\d+)\b", r"\1h\2", text)
    text = re.sub(r"\b(\d+)\s*h\b", r"\1h", text)

    text = re.sub(r"\b([a-z]{1,3})\s+(\d+)\b", r"\1\2", text)

    text = re.sub(r"\bnoeuds?\b|\bnœuds?\b", "noeuds", text)
    text = re.sub(r"\bça\b", "cela", text)

    for variant, canonical in SYNONYMS.items():
        text = re.sub(rf"\b{re.escape(variant)}\b", canonical, text)

    text = re.sub(r"\b(\w{4,})s\b", r"\1", text)

    def numbers_to_words(m):
        try:
            return num2words(int(m.group()), lang="fr")
        except Exception:
            return m.group()

    text = re.sub(r"\b\d+\b", numbers_to_words, text)
    return text
