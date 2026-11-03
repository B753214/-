from types import SimpleNamespace

import pytest

from rag_agent import bm25_store


def test_prepared_bm25_survives_reload_and_rejects_other_chunks(tmp_path):
    nodes = [SimpleNamespace(node_id="a", text="Webpack tree shaking removes unused code"),
             SimpleNamespace(node_id="b", text="MVCC handles concurrent transactions")]
    original = bm25_store.build(nodes)
    bm25_store.save(tmp_path, nodes, original)
    loaded = bm25_store.load(tmp_path, nodes)
    assert loaded.get_scores(["webpack"]).tolist() == pytest.approx(original.get_scores(["webpack"]).tolist())
    with pytest.raises(RuntimeError, match="不匹配"):
        bm25_store.load(tmp_path, list(reversed(nodes)))
