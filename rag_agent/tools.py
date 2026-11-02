import asyncio
from .config import Settings
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client


async def main():
    settings = Settings()
    def show(tools):
        for tool in tools:
            if not settings.notion_tool or tool.name == settings.notion_tool:
                print(tool.name, tool.inputSchema)
    if settings.notion_url:
        async with streamable_http_client(settings.notion_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                show((await session.list_tools()).tools)
    elif settings.notion_command:
        async with stdio_client(StdioServerParameters(command=settings.notion_command, args=list(settings.notion_args))) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                show((await session.list_tools()).tools)
    else:
        raise SystemExit("先配置 NOTION_MCP_URL 或 NOTION_MCP_COMMAND")

if __name__ == "__main__":
    asyncio.run(main())
