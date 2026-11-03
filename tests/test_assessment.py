import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

from rag_agent.config import Settings
from rag_agent.engine import Engine


def test_assessment_reuses_existing_answer_and_marks_unverified_reference():
    engine = Engine(replace(Settings(), openai_key="test", rerank_model="qwen3.7-flash"))
    calls = []

    async def no_lookup(_):
        raise AssertionError("existing reference should not be regenerated")

    async def fake_create(**kwargs):
        calls.append(kwargs)
        result = {"verdict": "partial", "correctness": "核心说法正确", "completeness": "漏掉恢复过程",
                  "explanation": "还缺少持久化恢复。", "missing_points": ["AOF 重写"],
                  "supplement": "可以补充 AOF 重写与恢复流程。"}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])

    engine.ask = no_lookup
    engine.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    result = asyncio.run(engine.assess_interview_answer("Redis 持久化怎么做？", "用 RDB 快照。",
                                                        "可用 RDB 和 AOF。", "model_fallback"))
    assert result["verdict"] == "partial"
    assert result["missing_points"] == ["AOF 重写"]
    assert result["verification_tag"] == "模型评估 · 未经资料核实"
    assert calls[0]["model"] == "qwen3.7-flash"


def test_assessment_never_runs_second_retrieval_without_reference():
    engine = Engine(replace(Settings(), openai_key="test"))
    async def no_lookup(_):
        raise AssertionError("candidate assessment must not search")

    async def fake_create(**_):
        result = {"verdict": "incorrect", "correctness": "Incorrect", "completeness": "Incomplete",
                  "explanation": "AOF was omitted", "missing_points": ["AOF"], "supplement": "Explain RDB and AOF"}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])

    engine.ask = no_lookup
    engine.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    result = asyncio.run(engine.assess_interview_answer("How does Redis persist data?", "Only RDB snapshots."))
    assert result["verification_tag"] == "Model assessment · Unverified"
    assert result["reference_sources"] == []


def test_interviewer_question_analysis_reuses_retrieved_answer():
    engine = Engine(replace(Settings(), openai_key="test"))

    async def no_lookup(_):
        raise AssertionError("question analysis must reuse the interviewer answer")

    async def fake_create(**_):
        result = {"focus": "Persistence tradeoffs", "expected_points": ["RDB", "AOF", "Recovery"]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])

    engine._interview_reference = no_lookup
    engine.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    result = asyncio.run(engine.analyze_interview_question("How does Redis persist data?",
        "Redis supports RDB snapshots and AOF logs.", "knowledge_base"))
    assert result["focus"] == "Persistence tradeoffs"
    assert result["expected_points"] == ["RDB", "AOF", "Recovery"]
    assert result["reference_mode"] == "knowledge_base"
    assert result["verification_tag"] == "Evidence-based analysis"
