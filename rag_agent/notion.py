"""Read-only adapter for an existing Notion MCP server."""
import json
import re
from urllib.parse import urlparse
from .config import Settings


PAGE_ID = re.compile(r"(?<![a-fA-F0-9])(?:[a-fA-F0-9]{32}|[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12})(?![a-fA-F0-9])")
PAGE_TAG = re.compile(r'<page\s+url="([^"]+)"[^>]*>', re.I)
PROPERTIES = re.compile(r"<properties>\s*(\{.*?\})\s*</properties>", re.S)


def page_title(content: str, page_id: str) -> str:
    match = PROPERTIES.search(content)
    if match:
        try:
            title = json.loads(match.group(1)).get("title")
            if isinstance(title, str) and title.strip():
                return re.sub(r"<[^>]+>", "", title).strip()
        except ValueError:
            pass
    return f"Notion {page_id}"


def child_page_ids(content: str) -> list[str]:
    """Find direct child pages in Notion fetch markup or explicit child_page blocks."""
    found: list[str] = []

    def add(value: str):
        match = PAGE_ID.search(value)
        if match:
            page_id = match.group().replace("-", "").lower()
            if page_id not in found:
                found.append(page_id)

    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        payload = None

    def scan_markup(markup: str):
        body = markup.split("<content>", 1)[-1].split("</content>", 1)[0]
        for link in PAGE_TAG.findall(body):
            add(urlparse(link).path)

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            if value.get("type") == "child_page" or value.get("object") == "child_page":
                add(str(value.get("id", "")))
            for item in value.values():
                walk(item)
        elif isinstance(value, str):
            scan_markup(value)

    walk(payload if payload is not None else content)
    return found


def text_from_result(result) -> str:
    if result.isError:
        raise RuntimeError("Notion MCP 读取失败")
    raw = (result.structuredContent if result.structuredContent else
           "\n".join(getattr(block, "text", "") for block in result.content if getattr(block, "type", "") == "text"))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return raw
    if isinstance(raw, dict) and isinstance(raw.get("text"), str):
        return raw["text"]
    return json.dumps(raw, ensure_ascii=False)


async def read_pages(settings: Settings) -> list[tuple[str, str]]:
    import httpx
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
    if not settings.notion_page_ids:
        return []
    if not settings.notion_tool:
        raise RuntimeError("请在 .env 配置 NOTION_MCP_TOOL；可先运行 python -m rag_agent.tools 查看工具")
    if settings.notion_url:
        parsed = urlparse(settings.notion_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("NOTION_MCP_URL 必须为 HTTP(S) 地址")
        headers = {"Authorization": f"Bearer {settings.notion_token}"} if settings.notion_token else {}
        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            async with streamable_http_client(settings.notion_url, http_client=client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await _read(session, settings)
    if settings.notion_command:
        params = StdioServerParameters(command=settings.notion_command, args=list(settings.notion_args))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await _read(session, settings)
    raise RuntimeError("请配置 NOTION_MCP_URL 或 NOTION_MCP_COMMAND")


async def _read(session, settings: Settings) -> list[tuple[str, str]]:
    tools = await session.list_tools()
    if settings.notion_tool not in {tool.name for tool in tools.tools}:
        raise RuntimeError(f"MCP 工具 {settings.notion_tool!r} 不存在，请检查工具列表")
    if settings.notion_max_pages < 1:
        raise ValueError("NOTION_MAX_PAGES 必须大于 0")
    pages = []
    queue = [page_id.lower() for page_id in settings.notion_page_ids]
    visited = set()
    while queue:
        page_id = queue.pop(0)
        if page_id in visited:
            continue
        if len(visited) >= settings.notion_max_pages:
            raise RuntimeError(f"Notion 页面超过 NOTION_MAX_PAGES={settings.notion_max_pages}，已停止导入")
        visited.add(page_id)
        result = await session.call_tool(settings.notion_tool, {settings.notion_page_argument: page_id})
        try:
            content = text_from_result(result)
        except RuntimeError as exc:
            raise RuntimeError(f"Notion 页面 {page_id} 读取失败；请检查该页面的授权或是否已删除") from exc
        if not content.strip():
            raise RuntimeError(f"Notion 页面 {page_id} 返回空内容")
        pages.append((page_id, content))
        if settings.notion_include_subpages:
            queue.extend(child for child in child_page_ids(content) if child not in visited)
    return pages
