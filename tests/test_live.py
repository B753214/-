import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

from rag_agent.config import Settings
from rag_agent.engine import Engine
from rag_agent.live_audio import asr_url, public_asr_event


def test_realtime_events_preserve_turn_boundaries_and_final_text():
    assert public_asr_event('{"type":"input_audio_buffer.speech_started"}') == {"type": "speech_started"}
    assert public_asr_event('{"type":"conversation.item.input_audio_transcription.text","text":"你能","stash":"说说"}') == {"type": "partial", "text": "你能说说"}
    assert public_asr_event('{"type":"conversation.item.input_audio_transcription.completed","transcript":" 你能说说 Webpack 吗？ "}') == {"type": "transcript", "text": "你能说说 Webpack 吗？"}
    assert public_asr_event('{"type":"session.updated"}') is None


def test_realtime_url_uses_configured_endpoint_and_model():
    settings = replace(Settings(), live_asr_ws_url="wss://example.com/api-ws/v1/realtime", live_asr_model="qwen3-asr-flash-realtime")
    assert asr_url(settings) == "wss://example.com/api-ws/v1/realtime?model=qwen3-asr-flash-realtime"


def test_question_classifier_uses_small_nonthinking_router():
    engine = Engine(replace(Settings(), openai_key="test", router_model="qwen-flash"))
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"is_question": True, "question": "Describe HMR."})))])

    engine.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    assert asyncio.run(engine.interview_question("Describe HMR.")) == "Describe HMR."
    assert calls[0]["model"] == "qwen-flash"
    assert calls[0]["extra_body"] == {"enable_thinking": False}
    assert "ASR can mishear" in calls[0]["messages"][0]["content"]
    assert "otherwise preserve" in calls[0]["messages"][0]["content"]
