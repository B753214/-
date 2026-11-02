"""Pure retrieval helpers, kept separate from model and network adapters."""
import re
from dataclasses import dataclass
from typing import Sequence


def language_of(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def tokens(text: str) -> list[str]:
    import jieba
    result = []
    for part in jieba.cut(text.lower()):
        result.extend(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", part))
    return result


@dataclass
class Hit:
    node_id: str
    text: str
    metadata: dict
    score: float = 0.0


def fuse_rrf(lists: Sequence[Sequence[Hit]], limit: int, k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    nodes: dict[str, Hit] = {}
    for hits in lists:
        for rank, hit in enumerate(hits, 1):
            nodes[hit.node_id] = hit
            scores[hit.node_id] = scores.get(hit.node_id, 0) + 1 / (k + rank)
    return [Hit(id_, nodes[id_].text, nodes[id_].metadata, score)
            for id_, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]]


def relevant(score: float | None, threshold: float) -> bool:
    # Only a calibrated reranker score can be compared to this threshold. RRF scores cannot.
    return score is not None and score >= threshold
