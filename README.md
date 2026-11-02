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

编辑 `.env`，填入 `OPENAI_API_KEY`、`TAVILY_API_KEY` 和 Notion MCP 连接方式。**不要提交 `.env`**。云端模型可通过 `OPENAI_BASE_URL` 和模型名切换为兼容 OpenAI API 的服务，但语音转写接口、JSON 输出与 embedding 兼容性需实测。

Notion MCP 可以使用 Streamable HTTP (`NOTION_MCP_URL`，可选 Bearer `NOTION_MCP_TOKEN`) 或本地 stdio (`NOTION_MCP_COMMAND` 与 JSON 数组 `NOTION_MCP_ARGS`)。当前运行环境没有暴露 Notion MCP 工具，因此无法预设真实工具名和参数。连接后运行：

```powershell
python -m rag_agent.tools
```

根据输出在 `.env` 填写只读工具名 `NOTION_MCP_TOOL` 和页面 ID 参数名 `NOTION_MCP_PAGE_ARGUMENT`。`NOTION_PAGE_IDS` 支持逗号分隔多个 ID。不同 MCP 服务返回结构可能不同；复杂嵌套页面需按实际工具结果完善解析。文档应具有读取权限。

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
