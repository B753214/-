"""Small in-memory bridge from the live browser page to the native overlay."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class OverlayEntry(BaseModel):
    role: Literal["interviewer", "me"]
    text: str = Field(max_length=1500)
    at: str = Field(max_length=40)
    answer: str | None = Field(default=None, max_length=8000)
    answer_tag: str | None = Field(default=None, max_length=120)
    assessment: str | None = Field(default=None, max_length=3000)


class OverlaySnapshot(BaseModel):
    running: bool = False
    status: str = Field(default="等待会话…", max_length=400)
    interviewer_preview: str = Field(default="", max_length=500)
    me_preview: str = Field(default="", max_length=500)
    entries: list[OverlayEntry] = Field(default_factory=list, max_length=12)


_snapshot = OverlaySnapshot()
_updated_at = datetime.now(timezone.utc)


def update(snapshot: OverlaySnapshot) -> None:
    global _snapshot, _updated_at
    _snapshot = snapshot
    _updated_at = datetime.now(timezone.utc)


def current() -> dict:
    return {"snapshot": _snapshot.model_dump(), "updated_at": _updated_at.isoformat()}
