import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from openai import APIConnectionError, APITimeoutError
from pydantic import BaseModel, Field
from .engine import Engine

engine = Engine()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        engine.load()
    except Exception as exc:
        print(f"Index load failed: {exc}")
    yield


app = FastAPI(title="Bilingual RAG Agent", lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


@app.get("/")
async def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
async def health():
    return {"ready": engine.index is not None, "chunks": len(engine.nodes), "bm25": engine.config.bm25_enabled, "rerank": engine.config.rerank_enabled}


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


@app.post("/api/ask/stream")
async def ask_stream(payload: Question):
    async def events():
        queue = asyncio.Queue()

        async def on_token(token: str):
            await queue.put({"type": "delta", "text": token})

        async def worker():
            try:
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
