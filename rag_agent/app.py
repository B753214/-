from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
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


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        return {"text": await engine.transcribe(audio.filename or "audio.webm", await audio.read())}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
