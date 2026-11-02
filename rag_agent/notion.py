"""Read-only adapter for an existing Notion MCP server."""
import json
from urllib.parse import urlparse
from .config import Settings


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
    pages = []
    for page_id in settings.notion_page_ids:
        result = await session.call_tool(settings.notion_tool, {settings.notion_page_argument: page_id})
        content = text_from_result(result)
        if not content.strip():
            raise RuntimeError(f"Notion 页面 {page_id} 返回空内容")
        pages.append((page_id, content))
    return pages
