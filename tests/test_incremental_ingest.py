import asyncio
from dataclasses import replace

import pymupdf
from llama_index.core.embeddings import MockEmbedding

from rag_agent.config import Settings
from rag_agent.engine import Engine
import rag_agent.engine as engine_module


def _write_pdf(path, text):
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


def test_pdf_upsert_keeps_other_sources_and_replaces_its_own_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(Engine, "_embed", lambda self: MockEmbedding(embed_dim=8))
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    config = replace(Settings(), openai_key="test", index_dir=tmp_path / "index", pdf_dir=pdf_dir)
    first, second = pdf_dir / "first.pdf", pdf_dir / "second.pdf"
    _write_pdf(first, "First document about Webpack")
    _write_pdf(second, "Second document about Docker")

    asyncio.run(Engine(config).upsert_pdf(first))
    added = asyncio.run(Engine(config).upsert_pdf(second))
    assert added["documents_replaced"] == 0
    assert added["chunks_total"] == 2

    second.unlink()
    _write_pdf(second, "Updated second document about containers")
    updated = asyncio.run(Engine(config).upsert_pdf(second))
    assert updated["documents_replaced"] == 1
    reloaded = Engine(config)
    assert reloaded.load()
    assert len(reloaded.nodes) == 2
    assert any("Webpack" in node.text for node in reloaded.nodes)
    assert any("Updated second" in node.text for node in reloaded.nodes)
    assert not any("Docker" in node.text for node in reloaded.nodes)
    assert reloaded.bm25 is not None


def test_notion_upsert_replaces_only_fetched_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(Engine, "_embed", lambda self: MockEmbedding(embed_dim=8))
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    config = replace(Settings(), openai_key="test", index_dir=tmp_path / "index", pdf_dir=pdf_dir,
                     rerank_enabled=False)
    pdf_path = pdf_dir / "local.pdf"
    _write_pdf(pdf_path, "Local PDF stays indexed")
    asyncio.run(Engine(config).upsert_pdf(pdf_path))

    page_id = "a" * 32
    content = "Notion article about Docker"

    async def read_one(_):
        return [(page_id, content)]

    monkeypatch.setattr(engine_module, "read_pages", read_one)
    asyncio.run(Engine(config).upsert_notion(f"https://app.notion.com/p/{page_id}"))
    content = "Notion article about updated containers"
    updated = asyncio.run(Engine(config).upsert_notion(page_id))

    reloaded = Engine(config)
    assert reloaded.load()
    assert updated["documents_replaced"] == 1
    assert len(reloaded.nodes) == 2
    assert any("Local PDF" in node.text for node in reloaded.nodes)
    assert any("updated containers" in node.text for node in reloaded.nodes)
    assert not any("about Docker" in node.text for node in reloaded.nodes)

    async def never_fetch(_):
        raise AssertionError("online Notion fetch is forbidden during startup and Q&A")

    async def answer(*_, **__):
        return "Local answer [1]"

    monkeypatch.setattr(engine_module, "read_pages", never_fetch)
    reloaded = Engine(config)
    assert reloaded.load()
    reloaded._generate = answer
    result = asyncio.run(reloaded.ask(f"请根据 https://app.notion.com/p/{page_id} 解释容器"))
    assert result["source_mode"] == "notion_url"
    assert result["sources"][0]["kind"] == "notion"
    missing = asyncio.run(reloaded.ask(f"请根据 https://app.notion.com/p/{'b' * 32} 回答"))
    assert missing["source_mode"] == "unavailable"
    assert "尚未导入" in missing["answer"]
