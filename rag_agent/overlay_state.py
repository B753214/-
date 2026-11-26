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
    client_id: str = Field(default="", max_length=64)
    running: bool = False
    has_session: bool = False
    has_entries: bool = False
    ai_enabled: bool = False
    assess_enabled: bool = False
    bleed_enabled: bool = True
    status: str = Field(default="等待会话…", max_length=400)
    interviewer_preview: str = Field(default="", max_length=500)
    me_preview: str = Field(default="", max_length=500)
    entries: list[OverlayEntry] = Field(default_factory=list, max_length=12)


_snapshot = OverlaySnapshot()
_updated_at = datetime.now(timezone.utc)
_commands: list[str] = []


def update(snapshot: OverlaySnapshot) -> None:
    global _snapshot, _updated_at, _commands
    if snapshot.client_id != _snapshot.client_id:
        _commands = []
    _snapshot = snapshot
    _updated_at = datetime.now(timezone.utc)


def current() -> dict:
    return {"snapshot": _snapshot.model_dump(), "updated_at": _updated_at.isoformat()}


def enqueue(action: str) -> None:
    _commands.append(action)


def take(client_id: str) -> list[str]:
    global _commands
    if not client_id or client_id != _snapshot.client_id:
        return []
    commands, _commands = _commands, []
    return commands
