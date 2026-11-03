"""Persist the prepared BM25 statistics alongside the vector index."""

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from .retrieval import tokens


FILENAME = "bm25.json"


def build(nodes) -> BM25Okapi:
    return BM25Okapi([tokens(node.text) or ["_"] for node in nodes])


def save(directory: Path, nodes, index: BM25Okapi) -> None:
    data = {
        "version": 1,
        "node_ids": [node.node_id for node in nodes],
        "corpus_size": index.corpus_size,
        "avgdl": index.avgdl,
        "doc_freqs": index.doc_freqs,
        "idf": index.idf,
        "doc_len": index.doc_len,
        "average_idf": index.average_idf,
        "k1": index.k1,
        "b": index.b,
        "epsilon": index.epsilon,
    }
    target = directory / FILENAME
    temporary = directory / (FILENAME + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)


def load(directory: Path, nodes) -> BM25Okapi:
    target = directory / FILENAME
    if not target.exists():
        raise RuntimeError("缺少离线 BM25 索引；请运行 python -m rag_agent.ingest --prepare-existing-index")
    data = json.loads(target.read_text(encoding="utf-8"))
    if data.get("version") != 1 or data.get("node_ids") != [node.node_id for node in nodes]:
        raise RuntimeError("BM25 索引与文档分块不匹配；请重新运行 python -m rag_agent.ingest")
    index = BM25Okapi.__new__(BM25Okapi)
    for name in ("corpus_size", "avgdl", "doc_freqs", "idf", "doc_len", "average_idf", "k1", "b", "epsilon"):
        setattr(index, name, data[name])
    return index
