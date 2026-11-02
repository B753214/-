import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rag_agent.config import Settings
from rag_agent.notion import _read, child_page_ids


ROOT = "24fa020ef1cf80b68685fcfade70e314"
CHILD = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_child_pages_from_markdown_and_blocks():
    content = (f"[Guide](https://app.notion.com/p/Guide-{CHILD}?source=copy_link)\n"
               f'[{{"type":"child_page","id":"{ROOT}"}}]')
    assert child_page_ids(content) == [CHILD]
    assert child_page_ids(f'{{"results":[{{"type":"child_page","id":"{ROOT}"}}]}}') == [ROOT]


def test_read_recurses_and_deduplicates():
    class Session:
        def __init__(self):
            self.calls = []

        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(name="fetch")])

        async def call_tool(self, name, args):
            page_id = args["page_id"]
            self.calls.append(page_id)
            body = f"[child](https://www.notion.so/{CHILD})" if page_id == ROOT else f"[root](https://www.notion.so/{ROOT})"
            return SimpleNamespace(isError=False, structuredContent=None, content=[SimpleNamespace(type="text", text=body)])

    session = Session()
    settings = replace(Settings(), notion_page_ids=(ROOT,), notion_tool="fetch")
    pages = asyncio.run(_read(session, settings))
    assert [page_id for page_id, _ in pages] == [ROOT, CHILD]
    assert session.calls == [ROOT, CHILD]

    with pytest.raises(RuntimeError, match="NOTION_MAX_PAGES"):
        asyncio.run(_read(Session(), replace(settings, notion_max_pages=1)))
