import asyncio
from dataclasses import replace
from types import SimpleNamespace

from rag_agent.config import Settings
from rag_agent.engine import Engine
from rag_agent.routing import explicit_need, public_web_url, source_for


def test_need_is_decided_before_source():
    assert explicit_need("HMR 的原理是什么？").search is False
    assert explicit_need("现在最新的 Qwen 版本是什么？").reason == "time_sensitive"
    assert explicit_need("我的知识库中对 HMR 怎么描述？").reason == "internal_data"
    assert explicit_need("投资需要注意什么？").reason == "high_stakes"
    assert source_for("我的知识库中对 HMR 怎么描述？", "internal_data")[0] == "knowledge_base"
    assert source_for("现在最新的 Qwen 版本是什么？", "time_sensitive")[0] == "web"
    assert source_for("解释 https://app.notion.com/p/Test-24fa020ef1cf80b68685fcfade70e314", "supplied_url")[0] == "notion_url"
    assert source_for("解释 https://example.com/article", "supplied_url")[0] == "web_url"
    assert public_web_url("https://example.com/article")
    assert not public_web_url("http://127.0.0.1:8000/")


def test_stable_concept_uses_model_without_retrieval():
    engine = Engine(replace(Settings(), openai_key="test", tavily_key=""))

    async def fake_create(**_):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="HMR 是热模块替换。"))])

    async def unexpected(_):
        raise AssertionError("stable concept must not retrieve")

    engine.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    engine._retrieve = unexpected
    result = asyncio.run(engine.ask("HMR 的原理是什么？"))
    assert result["source_mode"] == "model"
    assert result["retrieval_ms"] == 0
    assert result["route_ms"] >= 0
    assert result["sources"] == []
    assert engine._chat_extra("qwen3.7-max") == {"enable_thinking": False}
    assert engine._chat_extra("qwen-flash") == {"enable_thinking": False}


def test_current_fact_without_web_key_is_not_guessed():
    engine = Engine(replace(Settings(), openai_key="test", tavily_key=""))
    result = asyncio.run(engine.ask("现在最新的 Qwen 版本是什么？"))
    assert result["source_mode"] == "unavailable"
    assert "需要查证" in result["answer"]


def test_ambiguous_need_and_comparison_queries_use_small_model_calls():
    engine = Engine(replace(Settings(), openai_key="test"))
    responses = iter(['{"search":false}', '{"queries":["A speed", "B speed"]}'])
    models = []

    async def fake_create(**kwargs):
        models.append(kwargs["model"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))])

    engine.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    assert asyncio.run(engine._decide_need("解释一个未列出的成熟概念")).search is False
    assert asyncio.run(engine._web_queries("比较 A 和 B 的速度")) == ["A speed", "B speed"]
    assert asyncio.run(engine._web_queries("A 是什么")) == ["A 是什么"]
    assert models == [engine.config.router_model] * 2


def test_rerank_uses_its_own_model():
    engine = Engine(replace(Settings(), openai_key="test", router_model="qwen-flash", rerank_model="qwen3.7-flash"))
    models = []

    async def fake_create(**kwargs):
        models.append(kwargs["model"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"scores":[0.9]}'))])

    engine.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    from rag_agent.retrieval import Hit
    hits, _ = asyncio.run(engine._rank("test", [Hit("1", "text", {}, 0.5)], 0.5))
    assert len(hits) == 1
    assert models == ["qwen3.7-flash"]
