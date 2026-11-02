import json
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def enabled(key: str, default: bool = True) -> bool:
    return os.getenv(key, str(default)).lower() in {"true", "1", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    openai_key: str = os.getenv("AIROBOT_LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("AIROBOT_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    llm_model: str = os.getenv("AIROBOT_LLM_MODEL") or os.getenv("LLM_MODEL", "qwen3.7-max")
    router_model: str = os.getenv("AIROBOT_ROUTER_MODEL") or os.getenv("ROUTER_MODEL", "qwen3.7-flash")
    embed_model: str = os.getenv("EMBED_MODEL", "text-embedding-v4")
    transcribe_model: str = os.getenv("TRANSCRIBE_MODEL", "qwen3-asr-flash")
    tavily_key: str = os.getenv("TAVILY_API_KEY", "")
    notion_url: str = os.getenv("NOTION_MCP_URL", "")
    notion_token: str = os.getenv("NOTION_MCP_TOKEN", "")
    notion_command: str = os.getenv("NOTION_MCP_COMMAND", "")
    notion_args: tuple[str, ...] = tuple(json.loads(os.getenv("NOTION_MCP_ARGS", "[]")))
    notion_tool: str = os.getenv("NOTION_MCP_TOOL", "")
    notion_page_argument: str = os.getenv("NOTION_MCP_PAGE_ARGUMENT", "page_id")
    notion_page_ids: tuple[str, ...] = tuple(x.strip().replace("-", "") for x in os.getenv("NOTION_PAGE_IDS", "").split(",") if x.strip())
    notion_include_subpages: bool = enabled("NOTION_INCLUDE_SUBPAGES")
    notion_max_pages: int = int(os.getenv("NOTION_MAX_PAGES", "1000"))
    bm25_enabled: bool = enabled("BM25_ENABLED")
    rerank_enabled: bool = enabled("RERANK_ENABLED", True)
    web_fallback_enabled: bool = enabled("WEB_FALLBACK_ENABLED")
    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "8"))
    bm25_top_k: int = int(os.getenv("BM25_TOP_K", "8"))
    fusion_top_k: int = int(os.getenv("FUSION_TOP_K", "10"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "4"))
    relevance_threshold: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.70"))
    max_audio_mb: int = int(os.getenv("MAX_AUDIO_MB", "7"))
    index_dir: Path = ROOT / "data" / "index"
    pdf_dir: Path = ROOT / "data" / "pdfs"
