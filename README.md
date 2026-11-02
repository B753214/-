# Atlas 知识问答 Agent

Python + LlamaIndex 双语 RAG 问答。文字或音频提问，纯文字回答。Notion 通过现有 MCP 读取，可复制文字的 PDF 从 `data/pdfs/` 导入。向量与中文 BM25 并行召回，RRF 融合，可选重排，证据不足时回退 Tavily 网络搜索。索引持久化到本地，启动时加载到内存，适合约 1000 篇以内的文档。首次检索的云端 embedding、重排和生成仍会产生网络耗时；接口返回检索及总耗时。

## 启动

建议 Python 3.11–3.13（本机 Python 3.14 尚未验证全部依赖）。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，填入 `AIROBOT_LLM_API_KEY`、`TAVILY_API_KEY` 和 Notion MCP 连接方式。**不要提交 `.env`**。默认使用阿里云 DashScope 北京兼容接口：`qwen3.7-max` 回答、`text-embedding-v4` 向量、`qwen3-asr-flash` 音频转写。用户最初指定的 `qwen3-max` 在当前密钥和地址下返回 `model_not_found`，因此采用已验证可调用的 `qwen3.7-max`。`AIROBOT_LLM_BASE_URL` 可改为同一账号对应区域的地址。切换 embedding 模型后必须重新建索引。音频文件最大 7 MB，以保证 Base64 请求低于模型 10 MB 上限；支持 mp3、wav、webm、ogg。

Notion MCP 默认通过 Notion 官方托管服务 `https://mcp.notion.com/mcp` 连接，使用官方文档推荐的 `mcp-remote` stdio 桥接。Windows 下示例已配置 `npx.cmd`；其他系统可改为 `npx`。首次执行会打开浏览器进行 Notion OAuth 授权，需要选择能访问目标页面的工作区。也可直接使用 Streamable HTTP (`NOTION_MCP_URL`，可选 Bearer `NOTION_MCP_TOKEN`)，但此方式需自行提供可用的 MCP token；不要把普通 Notion API token 当作 MCP token。连接后运行：

```powershell
python -m rag_agent.tools
```

官方只读工具为 `notion-fetch`，页面参数为 `id`，已在示例中填写。`NOTION_PAGE_IDS` 支持逗号分隔多个 ID，示例默认指向 [Webpack 页面](https://app.notion.com/p/Webpack-24fa020ef1cf80b68685fcfade70e314?source=copy_link)。`NOTION_INCLUDE_SUBPAGES=true` 会递归读取 MCP 结果中 `<content>` 内的 `<page>` 子页面标签及显式的 `child_page` 块，按页面 ID 去重，不会跟随正文里的普通链接；`NOTION_MAX_PAGES` 限制遍历上限。此行为依赖 MCP 工具返回子页面标签或块；如果工具只返回当前页面正文，应显式列出页面 ID。页面和子页面都须授予 MCP 读取权限。索引构建完成后可通过返回的 `documents` 数量核对导入范围。

把可提取文字的 PDF 放进 `data/pdfs/`，然后建立索引并启动：

```powershell
python -m rag_agent.ingest
uvicorn rag_agent.app:app --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000/ 。修改 Notion 或 PDF 后重新运行 ingest 并重启服务。扫描版 PDF 需要 OCR，当前版本不会从图片中提取文字。

## 配置和接口

`BM25_ENABLED`、`RERANK_ENABLED`、`WEB_FALLBACK_ENABLED` 可单独开关。`VECTOR_TOP_K`、`BM25_TOP_K`、`FUSION_TOP_K`、`RERANK_TOP_K`、`RELEVANCE_THRESHOLD` 控制召回与回退。默认关闭云端重排以减少延迟，回退质量以最佳向量相似度判断；开启重排后以重排评分判断。RRF 分数不作为质量阈值。两种评分的阈值需分别用真实中英文问题校准。网页搜索密钥缺失时，低质量检索会返回明确错误。

- `GET /health`：索引状态与分块数量。
- `POST /api/ask`：JSON `{"question":"..."}`，返回答案语言、来源类型、引用和耗时。
- `POST /api/transcribe`：multipart `audio`，返回转写文字，用户可检查后再提问。

网页只显示文字答案，不做 PDF/Notion 上传管理。PDF 在服务端目录维护。内容提取、索引构建和模型调用都会产生云端 API 用量。

## 测试

```powershell
python -m pytest -q
```
