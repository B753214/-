import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rag_agent.config import Settings
from rag_agent.notion import _read, child_page_ids, page_title, text_from_result


ROOT = "24fa020ef1cf80b68685fcfade70e314"
CHILD = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def test_child_pages_from_markup_and_blocks():
    content = (f'<page url="https://app.notion.com/p/{ROOT}"><content>'
               f'<page url="https://app.notion.com/p/Guide-{CHILD}"/>'
               f'[cross-link](https://app.notion.com/p/{"b" * 32})'
               '</content></page>')
    assert child_page_ids(content) == [CHILD]
    assert child_page_ids(f'{{"results":[{{"type":"child_page","id":"{ROOT}"}}]}}') == [ROOT]
    assert page_title('<properties>{"title":"Webpack"}</properties>', ROOT) == "Webpack"
    assert page_title('<properties>{"title":"<b>优化</b>"}</properties>', ROOT) == "优化"


def test_fetch_json_wrapper_is_unwrapped():
    result = SimpleNamespace(isError=False, structuredContent=None,
                             content=[SimpleNamespace(type="text", text='{"title":"Webpack","text":"<properties>{\\"title\\":\\"Webpack\\"}</properties>"}')])
    assert page_title(text_from_result(result), ROOT) == "Webpack"


def test_read_recurses_and_deduplicates():
    class Session:
        def __init__(self):
            self.calls = []

        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(name="fetch")])

        async def call_tool(self, name, args):
            page_id = args["page_id"]
            self.calls.append(page_id)
            body = (f'<page url="https://app.notion.com/p/{page_id}"><content>'
                    f'<page url="https://app.notion.com/p/{CHILD if page_id == ROOT else ROOT}"/>'
                    '</content></page>')
            return SimpleNamespace(isError=False, structuredContent=None, content=[SimpleNamespace(type="text", text=body)])

    session = Session()
    settings = replace(Settings(), notion_page_ids=(ROOT,), notion_tool="fetch", notion_page_argument="page_id")
    pages = asyncio.run(_read(session, settings))
    assert [page_id for page_id, _ in pages] == [ROOT, CHILD]
    assert session.calls == [ROOT, CHILD]

    with pytest.raises(RuntimeError, match="NOTION_MAX_PAGES"):
        asyncio.run(_read(Session(), replace(settings, notion_max_pages=1)))
