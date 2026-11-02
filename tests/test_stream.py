import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

from rag_agent import app as app_module
from rag_agent.config import Settings
from rag_agent.engine import Engine


def test_model_tokens_are_forwarded_as_they_arrive():
    engine = Engine(replace(Settings(), openai_key="test"))

    async def fake_create(**kwargs):
        assert kwargs["stream"] is True

        async def chunks():
            for token in ("第一", "部分"):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=token))])

        return chunks()

    engine.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    received = []

    async def on_token(token):
        received.append(token)

    answer = asyncio.run(engine._generate("HMR 是什么？", "zh", on_token=on_token))
    assert answer == "第一部分"
    assert received == ["第一", "部分"]


def test_stream_endpoint_sends_delta_then_metadata():
    original = app_module.engine

    class FakeEngine:
        async def ask(self, question, on_token=None):
            await on_token("你好")
            return {"answer": "你好", "source_mode": "model", "sources": []}

    async def read_events():
        response = await app_module.ask_stream(app_module.Question(question="测试"))
        return [json.loads(item) async for item in response.body_iterator]

    app_module.engine = FakeEngine()
    try:
        events = asyncio.run(read_events())
    finally:
        app_module.engine = original
    assert [item["type"] for item in events] == ["delta", "done"]
    assert events[0]["text"] == "你好"
    assert events[1]["result"]["source_mode"] == "model"
