import asyncio
import base64
import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Awaitable, Callable
from urllib.parse import urlparse

import pymupdf
from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi
from .config import Settings
from .notion import PAGE_ID, page_title, read_pages
from .retrieval import Hit, conversational_answer, fuse_rrf, language_of, relevant, tokens
from .routing import NeedDecision, explicit_need, public_web_url, source_for


class Engine:
    def __init__(self, config: Settings | None = None):
        self.config = config or Settings()
        self.index = None
        self.nodes = []
        self.bm25 = None
        self.client = AsyncOpenAI(api_key=self.config.openai_key or "unset", base_url=self.config.openai_base_url or None,
                                  timeout=30.0, max_retries=0)

    def _embed(self):
        return OpenAIEmbedding(model_name=self.config.embed_model, embed_batch_size=10, api_key=self.config.openai_key, api_base=self.config.openai_base_url or None)

    def _chat_extra(self, model: str) -> dict:
        if "dashscope.aliyuncs.com" in self.config.openai_base_url and (model.startswith("qwen3") or model == "qwen-flash"):
            return {"enable_thinking": False}
        return {}

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
                documents.append(Document(text=text, id_=f"notion:{page_id}", metadata={"title": page_title(text, page_id), "source": f"https://app.notion.com/p/{page_id}", "kind": "notion"}))
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
        response = await self.client.chat.completions.create(model=self.config.rerank_model, temperature=0,
            extra_body=self._chat_extra(self.config.rerank_model), response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}])
        scores = json.loads(response.choices[0].message.content or "{}").get("scores", [])
        if not isinstance(scores, list) or len(scores) != len(hits):
            raise RuntimeError("重排服务未返回有效分数")
        graded = [Hit(h.node_id, h.text, h.metadata, max(0.0, min(1.0, float(scores[i])))) for i, h in enumerate(hits)]
        graded.sort(key=lambda h: h.score, reverse=True)
        return graded[:self.config.rerank_top_k], max(h.score for h in graded)

    async def _web_queries(self, question: str) -> list[str]:
        if not re.search(r"比较|对比|区别|调研|选型|\b(?:compare|comparison|versus|vs\.?|research)\b", question, re.I):
            return [question]
        prompt = ("Generate 2 or 3 distinct, focused web search queries for this comparison or research question. "
                  "Return JSON only: {\"queries\":[\"...\"]}. Keep the question's language and do not add private information.\n"
                  f"Question: {question}")
        response = await self.client.chat.completions.create(model=self.config.router_model, temperature=0,
            extra_body=self._chat_extra(self.config.router_model), response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}])
        try:
            queries = json.loads(response.choices[0].message.content or "{}").get("queries", [])
        except ValueError:
            queries = []
        if not isinstance(queries, list):
            queries = []
        return [q.strip() for q in queries[:3] if isinstance(q, str) and q.strip()] or [question]

    async def _web(self, question: str) -> list[Hit]:
        if not self.config.tavily_key:
            raise RuntimeError("已触发网络搜索，但未配置 TAVILY_API_KEY")
        from tavily import AsyncTavilyClient
        client = AsyncTavilyClient(api_key=self.config.tavily_key)
        results = await asyncio.gather(*(client.search(q, max_results=5, search_depth="basic") for q in await self._web_queries(question)))
        seen = set()
        hits = []
        for result in results:
            for item in result.get("results", []):
                url = item.get("url", "")
                if item.get("content") and url not in seen:
                    seen.add(url)
                    hits.append(Hit(url, item["content"], {"title": item.get("title", "网页"), "source": url, "kind": "web"}))
        return hits[:8]

    async def _extract_url(self, url: str, question: str) -> list[Hit]:
        if not public_web_url(url):
            raise ValueError("不能通过网络提取本地或私有地址")
        from tavily import AsyncTavilyClient
        result = await AsyncTavilyClient(api_key=self.config.tavily_key).extract(urls=[url], format="text", query=question)
        return [Hit(item.get("url", url), item.get("raw_content") or item.get("content", ""),
                    {"title": urlparse(url).hostname or url, "source": item.get("url", url), "kind": "web"})
                for item in result.get("results", []) if item.get("raw_content") or item.get("content")]

    async def _decide_need(self, question: str) -> NeedDecision:
        explicit = explicit_need(question)
        if explicit is not None:
            return explicit
        prompt = ("Decide ONLY whether an accurate answer requires external information. "
                  "Search for time-sensitive facts, high-stakes medical/legal/financial/political matters, unfamiliar or new names, "
                  "user-specific data, and supplied URLs. Do not search for stable math, algorithms, protocols, or mature concepts. "
                  "When uncertain about a fact, choose search. Return JSON only, such as {\"search\":true} or {\"search\":false}.\n"
                  f"Question: {question}")
        response = await self.client.chat.completions.create(model=self.config.router_model, temperature=0,
            extra_body=self._chat_extra(self.config.router_model), response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}])
        try:
            decision = json.loads(response.choices[0].message.content or "{}").get("search")
        except ValueError:
            decision = None
        return NeedDecision(decision is not False, "uncertain" if decision is not False else "stable_concept")

    async def interview_question(self, speech: str) -> str | None:
        """Extract an interview question without inventing missing speech."""
        prompt = (
            "Determine whether this completed interviewer utterance asks the candidate to answer or explain something. "
            "Imperatives such as 'describe', 'tell me about', '说说' and '介绍一下' count as questions. "
            "Greetings, statements, filler and incomplete fragments do not. "
            "Return JSON only: {\"is_question\":true,\"question\":\"...\"}. "
            "Keep the original language and meaning; never add unheard words.\nSpeech: " + speech.strip()[:1200]
        )
        response = await self.client.chat.completions.create(
            model=self.config.router_model, temperature=0,
            extra_body=self._chat_extra(self.config.router_model), response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            result = json.loads(response.choices[0].message.content or "{}")
        except ValueError:
            return None
        question = result.get("question")
        return question.strip() if result.get("is_question") is True and isinstance(question, str) and question.strip() else None

    async def _generate(self, question: str, lang: str, hits: list[Hit] | None = None,
                        on_token: Callable[[str], Awaitable[None]] | None = None) -> str:
        if hits is None:
            instruction = ("用中文回答。只回答稳定的通用知识；如果问题依赖最新信息、个人资料或你没有把握，明确说明需要核实，不要编造。"
                           if lang == "zh" else "Answer in English. Answer stable general knowledge only. If current or private facts are needed, say they need verification; do not guess.")
            user_content = question
        else:
            context = "\n\n".join(f"[{i}] {h.metadata.get('title', '')} | {h.metadata.get('source', '')}\n{h.text[:12000 if len(hits) == 1 else 1800]}" for i, h in enumerate(hits, 1))
            instruction = ("仅根据资料回答。中文问题用中文回答，英文问题用英文回答。引用资料时使用 [1] 这样的编号；资料不足就明确说明。"
                           if lang == "zh" else "Answer in English only, using the supplied evidence. Cite passages as [1], [2]. If evidence is insufficient, say so.")
            user_content = f"Question: {question}\n\nEvidence:\n{context}"
        params = {"model": self.config.llm_model, "temperature": 0,
                  "extra_body": self._chat_extra(self.config.llm_model),
                  "messages": [{"role": "system", "content": instruction}, {"role": "user", "content": user_content}]}
        if on_token is None:
            response = await self.client.chat.completions.create(**params)
            return response.choices[0].message.content or ""
        stream = await self.client.chat.completions.create(**params, stream=True)
        parts = []
        async for chunk in stream:
            token = chunk.choices[0].delta.content if chunk.choices else None
            if token:
                parts.append(token)
                await on_token(token)
        return "".join(parts)

    async def ask(self, question: str, on_token: Callable[[str], Awaitable[None]] | None = None) -> dict:
        question = question.strip()
        if not question or len(question) > 4000:
            raise ValueError("问题长度需为 1 至 4000 字")
        start = time.perf_counter()
        lang = language_of(question)
        direct = conversational_answer(question)
        if direct is not None:
            return {"answer": direct, "language": lang, "source_mode": "conversation", "sources": [], "retrieval_ms": 0,
                    "route_ms": 0, "total_ms": round((time.perf_counter() - start) * 1000)}
        if not self.config.openai_key:
            raise RuntimeError("请在 .env 设置 OPENAI_API_KEY")
        need = await self._decide_need(question)
        route_ms = round((time.perf_counter() - start) * 1000)
        if not need.search:
            answer = await self._generate(question, lang, on_token=on_token)
            return {"answer": answer, "language": lang, "source_mode": "model", "sources": [], "retrieval_ms": 0,
                    "route_reason": need.reason, "route_ms": route_ms,
                    "total_ms": round((time.perf_counter() - start) * 1000)}

        source_mode, url = source_for(question, need.reason)
        retrieval_start = time.perf_counter()
        if source_mode == "knowledge_base":
            hits, vector_score = await self._retrieve(question)
            hits, quality = await self._rank(question, hits, vector_score)
            if not relevant(quality, self.config.relevance_threshold):
                if self.config.web_fallback_enabled and self.config.tavily_key:
                    hits = await self._web(question)
                    source_mode = "web"
                elif self.config.web_fallback_enabled:
                    answer = ("知识库未找到足够可靠的资料；需要继续联网查证，但当前未配置网络搜索。" if lang == "zh"
                              else "The knowledge base did not provide reliable evidence, and web search is not configured.")
                    return {"answer": answer, "language": lang, "source_mode": "unavailable", "sources": [],
                            "retrieval_ms": round((time.perf_counter() - retrieval_start) * 1000),
                            "route_reason": "knowledge_base_low_quality", "route_ms": route_ms,
                            "total_ms": round((time.perf_counter() - start) * 1000)}
                else:
                    hits = []
        elif source_mode == "notion_url":
            match = PAGE_ID.search(url or "")
            if not match:
                hits = []
            else:
                page_id = match.group().replace("-", "").lower()
                pages = await read_pages(replace(self.config, notion_page_ids=(page_id,), notion_include_subpages=False))
                hits = [Hit(page_id, content, {"title": page_title(content, page_id), "source": url, "kind": "notion"})
                        for _, content in pages]
        elif not self.config.web_fallback_enabled or not self.config.tavily_key:
            answer = ("这个问题需要查证外部资料，但当前未配置可用的网络搜索。" if lang == "zh"
                      else "This question needs external verification, but web search is not configured.")
            return {"answer": answer, "language": lang, "source_mode": "unavailable", "sources": [], "retrieval_ms": 0,
                    "route_reason": need.reason, "route_ms": route_ms,
                    "total_ms": round((time.perf_counter() - start) * 1000)}
        elif source_mode == "web_url":
            hits = await self._extract_url(url or "", question)
        else:
            hits = await self._web(question)
        retrieved_ms = round((time.perf_counter() - retrieval_start) * 1000)
        if not hits:
            answer = "未找到足够可靠的资料来回答这个问题。" if lang == "zh" else "I couldn't find reliable information to answer that question."
            source_mode = "none"
        else:
            answer = await self._generate(question, lang, hits, on_token=on_token)
        return {"answer": answer, "language": lang, "source_mode": source_mode,
                "sources": [{"title": h.metadata.get("title", ""), "source": h.metadata.get("source", ""), "kind": h.metadata.get("kind", "")} for h in hits],
                "retrieval_ms": retrieved_ms, "route_reason": need.reason, "route_ms": route_ms,
                "total_ms": round((time.perf_counter() - start) * 1000)}

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
