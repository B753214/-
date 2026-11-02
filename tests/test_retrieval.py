from rag_agent.retrieval import Hit, fuse_rrf, language_of, relevant


def test_language_tracks_question():
    assert language_of("What does the document say?") == "en"
    assert language_of("这个文档说了什么？") == "zh"
    assert language_of("请解释 RAG 的工作方式") == "zh"
    assert language_of("OpenAI API 是做什么的？") == "zh"


def test_rrf_deduplicates_and_rewards_agreement():
    a = Hit("a", "A", {})
    b = Hit("b", "B", {})
    c = Hit("c", "C", {})
    results = fuse_rrf([[a, b], [c, a]], 3)
    assert [hit.node_id for hit in results] == ["a", "c", "b"]
    assert results[0].score > results[1].score


def test_quality_uses_grader_not_rrf():
    assert relevant(0.8, 0.55)
    assert not relevant(0.3, 0.55)
    assert not relevant(None, 0.55)
