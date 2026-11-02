"""Read-only adapter for an existing Notion MCP server."""
import json
import re
from urllib.parse import urlparse
from .config import Settings


PAGE_ID = re.compile(r"(?<![a-fA-F0-9])(?:[a-fA-F0-9]{32}|[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12})(?![a-fA-F0-9])")
NOTION_LINK = re.compile(r"https?://(?:[\w-]+\.)?(?:notion\.so|notion\.site|notion\.com)/[^\s<>\]\)\"']+", re.I)


def child_page_ids(content: str) -> list[str]:
    """Find linked Notion pages and explicit child_page blocks in MCP output."""
    found: list[str] = []

    def add(value: str):
        match = PAGE_ID.search(value)
        if match:
            page_id = match.group().replace("-", "").lower()
            if page_id not in found:
                found.append(page_id)

    for link in NOTION_LINK.findall(content):
        add(urlparse(link).path)
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        payload = None

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            if value.get("type") == "child_page" or value.get("object") == "child_page":
                add(str(value.get("id", "")))
            for item in value.values():
                walk(item)

    walk(payload)
    return found


def text_from_result(result) -> str:
    if result.isError:
        raise RuntimeError("Notion MCP 读取失败")
    if result.structuredContent:
        return json.dumps(result.structuredContent, ensure_ascii=False)
    return "\n".join(getattr(block, "text", "") for block in result.content if getattr(block, "type", "") == "text")


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
        content = text_from_result(result)
        if not content.strip():
            raise RuntimeError(f"Notion 页面 {page_id} 返回空内容")
        pages.append((page_id, content))
        if settings.notion_include_subpages:
            queue.extend(child for child in child_page_ids(content) if child not in visited)
    return pages
