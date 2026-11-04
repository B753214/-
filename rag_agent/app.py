import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, Header, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APITimeoutError
from pydantic import BaseModel, Field
from .engine import Engine
from .live_audio import relay_audio
from .overlay_state import OverlaySnapshot, current as overlay_current, update as overlay_update

engine = Engine()
_overlay_process: subprocess.Popen | None = None


def check_overlay_request(value: str | None) -> None:
    if value != "atlas-live":
        raise HTTPException(403, "仅允许本地通话页访问")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        engine.load()
    except Exception as exc:
        print(f"Index load failed: {exc}")
    yield


app = FastAPI(title="Bilingual RAG Agent", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    force_knowledge_base: bool = False


class InterviewSpeech(BaseModel):
    text: str = Field(min_length=1, max_length=1200)


class InterviewAnswer(BaseModel):
    question: str = Field(min_length=1, max_length=1200)
    answer: str = Field(min_length=1, max_length=4000)
    reference_answer: str | None = Field(default=None, max_length=8000)
    reference_mode: str | None = None


@app.get("/")
async def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/live")
async def live_page():
    return FileResponse(Path(__file__).parent / "static" / "live.html")


@app.get("/health")
async def health():
    return {"ready": engine.index is not None, "chunks": len(engine.nodes), "bm25": engine.config.bm25_enabled, "rerank": engine.config.rerank_enabled}


@app.get("/api/live/overlay/state")
async def overlay_state(x_atlas_overlay: str | None = Header(default=None)):
    check_overlay_request(x_atlas_overlay)
    return overlay_current()


@app.post("/api/live/overlay/state")
async def publish_overlay(snapshot: OverlaySnapshot, x_atlas_overlay: str | None = Header(default=None)):
    check_overlay_request(x_atlas_overlay)
    overlay_update(snapshot)
    return {"ok": True}


@app.post("/api/live/overlay/open")
async def open_overlay(x_atlas_overlay: str | None = Header(default=None)):
    check_overlay_request(x_atlas_overlay)
    if sys.platform != "win32":
        raise HTTPException(501, "原生浮窗目前仅支持 Windows")
    global _overlay_process
    if _overlay_process is not None and _overlay_process.poll() is None:
        return {"started": False, "pid": _overlay_process.pid}
    try:
        _overlay_process = subprocess.Popen(
            [sys.executable, "-m", "rag_agent.overlay_window"],
            cwd=Path(__file__).resolve().parents[1],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except OSError as exc:
        raise HTTPException(503, "无法启动 Windows 原生浮窗") from exc
    return {"started": True, "pid": _overlay_process.pid}


@app.post("/api/ask")
async def ask(payload: Question):
    try:
        return await engine.ask(payload.question)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except (APITimeoutError, APIConnectionError) as exc:
        raise HTTPException(503, "云端模型暂时无响应，请稍后重试") from exc


@app.websocket("/api/live/asr/{role}")
async def live_asr(socket: WebSocket, role: str):
    if role not in {"interviewer", "me"}:
        await socket.close(code=1008)
        return
    await relay_audio(socket, engine.config)


@app.post("/api/live/question")
async def live_question(payload: InterviewSpeech):
    try:
        return {"question": await engine.interview_question(payload.text)}
    except (APITimeoutError, APIConnectionError) as exc:
        raise HTTPException(503, "问题判断暂时无响应") from exc


@app.post("/api/live/assess")
async def live_assess(payload: InterviewAnswer):
    try:
        return await engine.assess_interview_answer(payload.question, payload.answer,
                                                    payload.reference_answer, payload.reference_mode)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    except (APITimeoutError, APIConnectionError) as exc:
        raise HTTPException(503, "回答分析暂时无响应") from exc


@app.post("/api/ask/stream")
async def ask_stream(payload: Question):
    async def events():
        queue = asyncio.Queue()

        async def on_token(token: str):
            await queue.put({"type": "delta", "text": token})

        async def worker():
            try:
                if payload.force_knowledge_base:
                    result = await engine.ask(payload.question, on_token=on_token, force_knowledge_base=True)
                else:
                    result = await engine.ask(payload.question, on_token=on_token)
                await queue.put({"type": "done", "result": result})
            except (ValueError, RuntimeError) as exc:
                await queue.put({"type": "error", "message": str(exc)})
            except (APITimeoutError, APIConnectionError):
                await queue.put({"type": "error", "message": "云端模型暂时无响应，请稍后重试"})
            except Exception:
                await queue.put({"type": "error", "message": "问答暂时失败，请稍后重试"})

        task = asyncio.create_task(worker())
        try:
            while True:
                event = await queue.get()
                yield json.dumps(event, ensure_ascii=False) + "\n"
                if event["type"] in {"done", "error"}:
                    break
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        return {"text": await engine.transcribe(audio.filename or "audio.webm", await audio.read())}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
