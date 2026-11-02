"""Pure retrieval helpers, kept separate from model and network adapters."""
import re
from dataclasses import dataclass
from typing import Sequence


def language_of(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def conversational_answer(question: str) -> str | None:
    """Answer a small set of app identity/greeting questions without retrieval."""
    normalized = re.sub(r"[\s，。！？、,.!?]+", "", question).lower()
    if normalized in {"你是谁", "你是什么", "你是啥", "你叫什么", "你叫什么名字", "自我介绍", "介绍一下你自己", "请问你是谁"}:
        return "我是 Atlas 知识问答助手。可以根据已导入的 Notion 和 PDF 文档回答问题，并标注资料来源。你可以用中文或英文提问，也可以先将音频转成文字。"
    if normalized in {"whoareyou", "whatareyou", "whatisyourname", "tellyourself", "tellmeaboutyourself"}:
        return "I'm Atlas, a knowledge-base assistant. I answer questions using the imported Notion pages and PDF documents, with source links. You can ask in English or Chinese, or transcribe audio into a question."
    if normalized in {"你好", "您好", "嗨", "hello", "hi", "hey"}:
        return ("你好！我是 Atlas 知识问答助手。你可以直接问我知识库中的问题。" if language_of(question) == "zh"
                else "Hello! I'm Atlas. Ask me a question about your knowledge base.")
    return None


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
