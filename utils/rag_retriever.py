import os
import re
from collections import Counter

STOPWORDS = {
    "le","la","les","de","du","des","un","une","et","en","au","aux",
    "je","tu","il","elle","nous","vous","ils","elles","que","qui","quoi",
    "dans","sur","pour","par","avec","est","sont","pas","plus","mais",
    "ou","donc","car","ni","or","ce","se","sa","son","ses","mon","ma",
    "mes","ton","ta","tes","lui","leur","leurs","y","on","ne","si","très",
    "bien","aussi","tout","tous","même","comme","alors","après","avant",
    "ça","cest","cetait","jai","javais","quil","quelle"
}

CONTAMINATION_MARKERS = {
    "AnimationType", "xmlns", "DOCTYPE", "<?xml", "<html", "function(",
    "undefined", "null", "NaN", "localhost", "http://", "https://"
}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t not in STOPWORDS and len(t) >= 3]


def _is_contaminated(text: str) -> bool:
    return any(marker in text for marker in CONTAMINATION_MARKERS)


def _parse_srt_passages(content: str, window: int = 5) -> list[str]:
    blocs = []
    for bloc in content.split("\n\n"):
        lines = []
        for line in bloc.split("\n"):
            t = line.strip()
            if not t or re.match(r"^\d+$", t) or "-->" in t:
                continue
            lines.append(t)
        text = " ".join(lines).strip()
        if len(text) > 15 and not _is_contaminated(text):
            blocs.append(text)

    passages = []
    seen = set()
    for i in range(len(blocs)):
        passage = " ".join(blocs[i:i+window])
        if passage not in seen:
            seen.add(passage)
            passages.append(passage)
    return passages


class RAGRetriever:
    def __init__(self, srt_folder: str):
        self.passages: list[str] = []
        self.tokenized: list[list[str]] = []
        self.doc_freq: Counter = Counter()
        self._index(srt_folder)

    def _index(self, folder: str):
        if not os.path.isdir(folder):
            print(f"[RAG] Dossier introuvable : {folder}")
            return

        files = [f for f in os.listdir(folder) if f.endswith(".srt")]
        print(f"[RAG] Indexation de {len(files)} SRTs...")
        seen_passages = set()

        for fname in files:
            path = os.path.join(folder, fname)
            content = open(path, encoding="utf-8", errors="ignore").read()
            for passage in _parse_srt_passages(content):
                if passage in seen_passages:
                    continue
                seen_passages.add(passage)
                tokens = _tokenize(passage)
                self.passages.append(passage)
                self.tokenized.append(tokens)
                for w in set(tokens):
                    self.doc_freq[w] += 1

        print(f"[RAG] {len(self.passages)} passages indexés.")

    def find_similar(self, query: str, n: int = 6) -> list[str]:
        if not self.passages:
            return []

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        results = []
        for i, tokens in enumerate(self.tokenized):
            passage_set = set(tokens)
            score = 0.0
            for w in query_tokens:
                if w in passage_set:
                    df = self.doc_freq.get(w, 1)
                    weight = 3.0 if df <= 3 else (1.5 if df <= 10 else 1.0)
                    score += weight
            score /= len(query_tokens)
            if score > 0:
                results.append((score, self.passages[i]))

        results.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in results[:n]]