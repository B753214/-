import asyncio
from dataclasses import replace

from rag_agent.config import Settings
from rag_agent.engine import Engine
from rag_agent.retrieval import Hit, conversational_answer, fuse_rrf, language_of, relevant


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


def test_identity_question_skips_retrieval_and_web():
    engine = Engine(replace(Settings(), openai_key="", tavily_key=""))
    result = asyncio.run(engine.ask("你是谁？"))
    assert result["source_mode"] == "conversation"
    assert result["retrieval_ms"] == 0
    assert result["sources"] == []
    assert "Atlas" in result["answer"]
    assert conversational_answer("Who are you?").startswith("I'm Atlas")
    assert conversational_answer("Webpack 的 HMR 是什么？") is None


def test_low_relevance_without_web_key_reports_unavailable_fallback():
    engine = Engine(replace(Settings(), openai_key="test", tavily_key="", web_fallback_enabled=True))

    async def no_hits(_):
        return [], 0.0

    engine._retrieve = no_hits
    result = asyncio.run(engine.ask("知识库里没有的内容是什么？"))
    assert result["source_mode"] == "unavailable"
    assert result["sources"] == []
    assert "知识库未找到" in result["answer"]
    assert "联网" in result["answer"]


def test_low_relevance_with_web_key_falls_back_to_web():
    engine = Engine(replace(Settings(), openai_key="test", tavily_key="test", web_fallback_enabled=True))

    async def no_hits(_):
        return [], 0.0

    async def web_hits(_):
        return [Hit("https://example.com", "Evidence", {"title": "Web", "source": "https://example.com", "kind": "web"})]

    async def answer(*_, **__):
        return "来自网络的答案 [1]"

    engine._retrieve = no_hits
    engine._web = web_hits
    engine._generate = answer
    result = asyncio.run(engine.ask("我的知识库里这个问题如何回答？"))
    assert result["source_mode"] == "web"
    assert result["sources"][0]["kind"] == "web"
