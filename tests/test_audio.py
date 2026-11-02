import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from rag_agent.config import Settings
from rag_agent.engine import Engine


def test_dashscope_audio_uses_data_uri_without_language_lock():
    engine = Engine(Settings(openai_key="test-key", transcribe_model="qwen3-asr-flash"))
    create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Hello 世界"))]))
    engine.client.chat.completions.create = create
    assert asyncio.run(engine.transcribe("question.webm", b"audio bytes")) == "Hello 世界"
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "qwen3-asr-flash"
    assert kwargs["messages"][0]["content"][0]["input_audio"]["data"].startswith("data:audio/webm;base64,")
    assert "language" not in kwargs["extra_body"]["asr_options"]


def test_rejects_unsupported_audio():
    engine = Engine(Settings(openai_key="test-key"))
    with pytest.raises(ValueError):
        asyncio.run(engine.transcribe("question.exe", b"audio bytes"))
