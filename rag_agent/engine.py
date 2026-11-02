import asyncio
import base64
import json
import re
import time
from pathlib import Path

import pymupdf
from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
from .config import Settings
from .notion import read_pages
from .retrieval import Hit, fuse_rrf, language_of, relevant, tokens


class Engine:
    def __init__(self, config: Settings | None = None):
        self.config = config or Settings()
        self.index = None
        self.nodes = []
        self.bm25 = None
        self.client = AsyncOpenAI(api_key=self.config.openai_key or "unset", base_url=self.config.openai_base_url or None)

    def _embed(self):
        return OpenAIEmbedding(model=self.config.embed_model, api_key=self.config.openai_key, api_base=self.config.openai_base_url or None)

    def load(self) -> bool:
        if not (self.config.index_dir / "docstore.json").exists():
            return False
        storage = StorageContext.from_defaults(persist_dir=str(self.config.index_dir))
        self.index = load_index_from_storage(storage, embed_model=self._embed())
        self.nodes = list(self.index.docstore.docs.values())
        self._build_bm25()
        return True

    def _build_bm25(self):
        self.bm25 = BM25Okapi([tokens(node.text) or ["_"] for node in self.nodes]) if self.nodes and self.config.bm25_enabled else None

    async def rebuild(self) -> dict:
        if not self.config.openai_key:
            raise RuntimeError("请在 .env 设置 OPENAI_API_KEY")
        documents = []
        if self.config.notion_page_ids:
            for page_id, text in await read_pages(self.config):
                documents.append(Document(text=text, id_=f"notion:{page_id}", metadata={"title": f"Notion {page_id}", "source": f"https://www.notion.so/{page_id}", "kind": "notion"}))
        for path in sorted(self.config.pdf_dir.glob("*.pdf")):
            with pymupdf.open(path) as pdf:
                for number, page in enumerate(pdf, 1):
                    content = page.get_text().strip()
                    if content:
                        documents.append(Document(text=content, id_=f"pdf:{path.name}:{number}", metadata={"title": path.name, "source": f"{path.name} · 第 {number} 页", "kind": "pdf", "page": number}))
        if not documents:
            raise RuntimeError("没有可索引的文档；请配置 Notion MCP 或在 data/pdfs 放入可提取文字的 PDF")
        nodes = SentenceSplitter(chunk_size=512, chunk_overlap=80).get_nodes_from_documents(documents)
        index = await asyncio.to_thread(VectorStoreIndex, nodes, embed_model=self._embed(), show_progress=False)
        self.config.index_dir.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(index.storage_context.persist, persist_dir=str(self.config.index_dir))
        self.index, self.nodes = index, nodes
        self._build_bm25()
        return {"documents": len(documents), "chunks": len(nodes)}

    def _bm25_hits(self, query: str) -> list[Hit]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokens(query))
        indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.config.bm25_top_k]
        return [Hit(self.nodes[i].node_id, self.nodes[i].text, self.nodes[i].metadata, float(scores[i])) for i in indexes if scores[i] > 0]

    async def _retrieve(self, query: str) -> tuple[list[Hit], float]:
        if self.index is None:
            raise RuntimeError("知识库尚未建立索引；请先运行 python -m rag_agent.ingest")
        retriever = self.index.as_retriever(similarity_top_k=self.config.vector_top_k)
        vector_task = asyncio.to_thread(retriever.retrieve, query)
        bm25_task = asyncio.to_thread(self._bm25_hits, query) if self.config.bm25_enabled else None
        vector_nodes, lexical = await asyncio.gather(vector_task, bm25_task) if bm25_task else (await vector_task, [])
        vectors = [Hit(item.node.node_id, item.node.text, item.node.metadata, item.score or 0) for item in vector_nodes]
        return fuse_rrf([vectors, lexical], self.config.fusion_top_k), max((hit.score for hit in vectors), default=0.0)

    async def _rank(self, question: str, hits: list[Hit], vector_score: float) -> tuple[list[Hit], float]:
        if not hits:
            return [], 0.0
        if not self.config.rerank_enabled:
            return hits[:self.config.rerank_top_k], vector_score
        candidates = [{"id": i, "text": h.text[:1100]} for i, h in enumerate(hits)]
        prompt = ("Rate whether each passage contains evidence to answer the question. Return JSON only: "
                  "{\"scores\": [numbers between 0 and 1 in candidate order]}. Do not reward keyword overlap without answer evidence.\n"
                  f"Question: {question}\nCandidates: {json.dumps(candidates, ensure_ascii=False)}")
        response = await self.client.chat.completions.create(model=self.config.llm_model, temperature=0, response_format={"type": "json_object"}, messages=[{"role": "user", "content": prompt}])
        scores = json.loads(response.choices[0].message.content or "{}").get("scores", [])
        if not isinstance(scores, list) or len(scores) != len(hits):
            raise RuntimeError("重排服务未返回有效分数")
        graded = [Hit(h.node_id, h.text, h.metadata, max(0.0, min(1.0, float(scores[i])))) for i, h in enumerate(hits)]
        graded.sort(key=lambda h: h.score, reverse=True)
        return graded[:self.config.rerank_top_k], max(h.score for h in graded)

    async def _web(self, question: str) -> list[Hit]:
        if not self.config.tavily_key:
            raise RuntimeError("已触发网络搜索，但未配置 TAVILY_API_KEY")
        from tavily import AsyncTavilyClient
        result = await AsyncTavilyClient(api_key=self.config.tavily_key).search(question, max_results=5, search_depth="basic")
        return [Hit(item.get("url", ""), item.get("content", ""), {"title": item.get("title", "网页"), "source": item.get("url", ""), "kind": "web"}) for item in result.get("results", []) if item.get("content")]

    async def ask(self, question: str) -> dict:
        question = question.strip()
        if not question or len(question) > 4000:
            raise ValueError("问题长度需为 1 至 4000 字")
        if not self.config.openai_key:
            raise RuntimeError("请在 .env 设置 OPENAI_API_KEY")
        start = time.perf_counter()
        lang = language_of(question)
        hits, vector_score = await self._retrieve(question)
        retrieved_ms = round((time.perf_counter() - start) * 1000)
        hits, quality = await self._rank(question, hits, vector_score)
        source_mode = "knowledge_base"
        if not relevant(quality, self.config.relevance_threshold):
            if self.config.web_fallback_enabled:
                hits = await self._web(question)
                source_mode = "web"
            else:
                hits = []
        if not hits:
            return {"answer": "未找到足够可靠的资料来回答这个问题。" if lang == "zh" else "I couldn't find reliable information to answer that question.", "language": lang, "source_mode": source_mode, "sources": [], "retrieval_ms": retrieved_ms}
        context = "\n\n".join(f"[{i}] {h.metadata.get('title', '')} | {h.metadata.get('source', '')}\n{h.text[:1800]}" for i, h in enumerate(hits, 1))
        instruction = ("仅根据资料回答。中文问题用中文回答，英文问题用英文回答。引用资料时使用 [1] 这样的编号；资料不足就明确说明。"
                       if lang == "zh" else "Answer in English only, using the supplied evidence. Cite passages as [1], [2]. If evidence is insufficient, say so.")
        response = await self.client.chat.completions.create(model=self.config.llm_model, temperature=0, messages=[{"role": "system", "content": instruction}, {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"}])
        return {"answer": response.choices[0].message.content or "", "language": lang, "source_mode": source_mode, "sources": [{"title": h.metadata.get("title", ""), "source": h.metadata.get("source", ""), "kind": h.metadata.get("kind", "")} for h in hits], "retrieval_ms": retrieved_ms, "total_ms": round((time.perf_counter() - start) * 1000)}

    async def transcribe(self, filename: str, data: bytes) -> str:
        if not self.config.openai_key:
            raise RuntimeError("请在 .env 设置 OPENAI_API_KEY")
        if not data or len(data) > self.config.max_audio_mb * 1024 * 1024:
            raise ValueError(f"音频不能为空且不能超过 {self.config.max_audio_mb} MB")
        suffix = Path(filename).suffix.lower()
        mime = {".mp3": "audio/mpeg", ".mpeg": "audio/mpeg", ".wav": "audio/wav", ".webm": "audio/webm", ".ogg": "audio/ogg"}.get(suffix)
        if not mime:
            raise ValueError("请上传 mp3、wav、webm 或 ogg 音频")
        uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"
        result = await self.client.chat.completions.create(model=self.config.transcribe_model, messages=[{"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": uri}}]}], extra_body={"asr_options": {"enable_itn": False}})
        return (result.choices[0].message.content or "").strip()
