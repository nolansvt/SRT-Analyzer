import re
from dataclasses import dataclass


@dataclass
class SRTSegment:
    index: int
    start: str
    end: str
    text: str


def parse_srt(content: str) -> list[SRTSegment]:
    content = content.strip().replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", content)
    segments = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1].strip(),
        )
        if not ts_match:
            continue

        start, end = ts_match.group(1), ts_match.group(2)
        text = " ".join(lines[2:]).strip()
        segments.append(SRTSegment(index=index, start=start, end=end, text=text))

    return segments


def segments_to_plain_text(segments: list[SRTSegment]) -> str:
    return " ".join(seg.text for seg in segments)


def srt_file_to_plain_text(content: str) -> str:
    return segments_to_plain_text(parse_srt(content))