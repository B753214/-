"""Bridge one browser audio track to one realtime ASR session."""
import asyncio
import base64
import json
from urllib.parse import urlencode, urlsplit, urlunsplit

import websockets
from fastapi import WebSocket, WebSocketDisconnect

from .config import Settings


def asr_url(settings: Settings) -> str:
    url = urlsplit(settings.live_asr_ws_url)
    if url.scheme != "wss" or not url.hostname:
        raise ValueError("LIVE_ASR_WS_URL 必须是 wss 地址")
    return urlunsplit((url.scheme, url.netloc, url.path.rstrip("/"), urlencode({"model": settings.live_asr_model}), ""))


def public_asr_event(raw: str) -> dict | None:
    event = json.loads(raw)
    kind = event.get("type")
    if kind == "input_audio_buffer.speech_started":
        return {"type": "speech_started"}
    if kind == "conversation.item.input_audio_transcription.text":
        return {"type": "partial", "text": (event.get("text") or "") + (event.get("stash") or "")}
    if kind == "conversation.item.input_audio_transcription.completed":
        return {"type": "transcript", "text": (event.get("transcript") or "").strip()}
    if kind in {"error", "conversation.item.input_audio_transcription.failed"}:
        return {"type": "error", "message": (event.get("error") or {}).get("message") or "语音识别失败"}
    return None


async def relay_audio(browser: WebSocket, settings: Settings) -> None:
    await browser.accept()
    if not settings.openai_key:
        await browser.send_json({"type": "error", "message": "请先配置 DashScope API Key"})
        await browser.close()
        return
    try:
        async with websockets.connect(
            asr_url(settings),
            additional_headers={"Authorization": f"Bearer {settings.openai_key}", "OpenAI-Beta": "realtime=v1"},
            open_timeout=10,
            max_size=2 * 1024 * 1024,
        ) as upstream:
            await upstream.send(json.dumps({"type": "session.update", "event_id": "session_start", "session": {
                "input_audio_format": "pcm", "sample_rate": 16000,
                "turn_detection": {"type": "server_vad", "silence_duration_ms": 400, "threshold": 0.2},
            }}))
            await browser.send_json({"type": "ready"})

            async def send_audio() -> bool:
                while True:
                    message = await browser.receive()
                    if message["type"] == "websocket.disconnect":
                        return False
                    chunk = message.get("bytes")
                    if chunk:
                        if len(chunk) > 65536:
                            raise ValueError("音频块过大")
                        await upstream.send(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")}))
                    elif message.get("text") == "stop":
                        await upstream.send(json.dumps({"type": "session.finish", "event_id": "session_stop"}))
                        return True

            async def receive_text() -> None:
                async for raw in upstream:
                    message = json.loads(raw)
                    event = public_asr_event(raw)
                    if event:
                        await browser.send_json(event)
                    if message.get("type") == "session.finished":
                        break

            sending = asyncio.create_task(send_audio())
            receiving = asyncio.create_task(receive_text())
            try:
                done, _ = await asyncio.wait({sending, receiving}, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                if sending in done and sending.result():
                    await asyncio.wait_for(receiving, 5)
            finally:
                for task in (sending, receiving):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(sending, receiving, return_exceptions=True)
    except (OSError, TimeoutError, ValueError, WebSocketDisconnect, websockets.WebSocketException):
        try:
            await browser.send_json({"type": "error", "message": "实时识别连接失败，请检查网络和模型配置"})
        except (OSError, RuntimeError, WebSocketDisconnect):
            pass
    finally:
        try:
            await browser.close()
        except (OSError, RuntimeError, WebSocketDisconnect):
            pass
