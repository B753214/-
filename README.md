# Atlas 知识问答 Agent

基于 Python、FastAPI 和 LlamaIndex 的中英文 RAG 问答应用。用户可输入文字或音频，应用只输出文字。知识来源为 Notion 页面和服务端 PDF；问答页不提供文档上传功能。

问答先判断是否需要查资料，再选择资料来源。稳定的数学、算法、协议和成熟技术概念由模型直接回答；内部资料走 Notion/PDF 知识库；当前信息、高风险事实和陌生的新名词需要外部核实；用户给出链接时优先读取该链接。中文问题用中文回答，英文问题用英文回答。

知识库检索使用向量与可选 BM25 召回、RRF 融合、可选重排。网络搜索通过 Tavily 完成；对比和调研问题可拆成 2–3 个并行查询。常见问候和应用身份问题直接回答，不调用模型或搜索。

## 环境要求

- Python 3.11–3.13；本机 Python 3.14 尚未完整验证。
- 使用默认 Notion MCP 配置时，需要 Node.js 和 `npx`。首次导入时会在浏览器中完成 Notion OAuth 授权。
- DashScope API Key。仅在需要网络搜索回退时才需要 Tavily API Key。

## 快速开始

在项目根目录执行以下 PowerShell 命令。已有 `.env` 时不要再次复制示例文件，以免覆盖配置。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，至少填入 `AIROBOT_LLM_API_KEY`，并配置知识来源：

```env
AIROBOT_LLM_API_KEY=你的DashScope密钥
NOTION_PAGE_IDS=24fa020ef1cf80b68685fcfade70e314
```

如果只用 PDF，可将 `NOTION_PAGE_IDS` 留空。默认回答模型为 `qwen3.7-max`，路由模型为 `qwen-flash`，重排模型为 `qwen3.7-flash`，向量模型为 `text-embedding-v4`，音频模型为 `qwen3-asr-flash`。DashScope 的文本模型调用默认关闭思考模式以缩短等待时间；其他兼容服务不额外传此参数。问答页使用流式回答，长答案会逐步显示；模型请求 30 秒超时，页面会显示超时提示。默认地址是 DashScope 北京兼容接口；使用其他区域或兼容服务时，调整 `.env` 中的模型和地址。切换 embedding 模型后必须重新建索引。

先导入文档，再启动问答服务：

```powershell
python -m rag_agent.ingest
uvicorn rag_agent.app:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/>。首次 Notion 导入会触发浏览器授权，请选择能够读取目标页面的工作区。`GET /health` 中 `ready: true` 表示索引已加载。

### 通话模式

在桌面版 Chrome 或 Edge 打开问答页，点击“通话模式”，或直接打开 <http://127.0.0.1:8000/live>。点击“开始会话”后，选择外部会议窗口或整个屏幕，并在浏览器共享对话框中勾选音频，再允许麦克风。页面分别标记“面试官 · 电脑声音”和“我 · 麦克风”。若所选共享来源没有音轨，会提示重新选择；浏览器对系统音频的支持取决于浏览器、操作系统和共享来源。

“AI 回答”默认关闭，此时只转写。开启后，仅对开启之后开始说出的面试官问题生成文字答案；你的麦克风内容不会触发回答。面试官停顿、实时识别给出完整转写后，才调用小模型判断是否为问题；技术主题短语（如“Webpack tree shaking”）也按隐含提问处理。每条面试官转写会显示判断状态；未识别为问题时，可用“按这段话回答”手动触发。回答沿用现有问答路由和知识库检索。点击“悬浮小窗”可在支持 Document Picture-in-Picture 的浏览器中打开置顶窗口，用户可拖动及调整大小；主标签页需保持打开。结束后点击“导出记录”下载 Markdown，包含双方转写、AI 答案和来源。记录只保留在当前标签页的 `sessionStorage`；关闭标签页后清除，开始新会话会覆盖旧记录。应用不保存音频文件。

实时识别使用两路 `qwen3-asr-flash-realtime` WebSocket，会产生语音模型用量。可通过 `LIVE_ASR_WS_URL` 指向所在业务空间的实时 WebSocket 地址，`LIVE_ASR_MODEL` 设置模型名。密钥仍只保存在服务端 `.env`，不会发送给浏览器。建议使用耳机：会议音频仍从共享来源的数字音轨采集，耳机能避免播放声音再次进入麦克风。页面默认开启“防串音”，面试官说话时会暂停麦克风转写；如果需要记录双方同时说话，请关闭该开关。浏览器也会尝试启用更强的麦克风回声消除，但效果取决于设备和浏览器。

**启动服务只加载已有索引，不会重新读取文档。** Notion 或 PDF 内容变更后，再运行 `python -m rag_agent.ingest`，随后重启服务。目前每次导入都会全量重建索引。

## 添加知识库文档

在问题中明确写“我的知识库”“这份文档”“根据我的 Notion 文档”等，会选择本地知识库。普通稳定概念不会自动查知识库；若希望按内部文档回答，请在问题中点明文档或提供 Notion 页面链接。

### Notion：多个链接

Notion 文档无需下载到本地。将每个链接中的页面 ID 写入 `.env` 的 `NOTION_PAGE_IDS`，用英文逗号分隔，不要直接粘贴完整 URL。例如链接 `https://app.notion.com/p/Webpack-24fa020ef1cf80b68685fcfade70e314?source=copy_link` 的页面 ID 是 `24fa020ef1cf80b68685fcfade70e314`。

```env
NOTION_PAGE_IDS=24fa020ef1cf80b68685fcfade70e314,aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
NOTION_INCLUDE_SUBPAGES=true
NOTION_MAX_PAGES=1000
```

建议只列出几个父页面：导入器会递归读取其子页面、按 ID 去重，并且不会跟随正文中的普通链接。分散在不同位置的页面可列出多个 ID。所有页面须对已授权的 Notion 工作区可读。若 MCP 返回内容中没有子页面结构，需要逐个列出页面 ID。超过 `NOTION_MAX_PAGES` 时导入会停止并报错。

默认使用 Notion 官方托管 MCP，通过 `npx.cmd` 启动 `mcp-remote` 桥接；非 Windows 系统将 `NOTION_MCP_COMMAND` 改为 `npx`。只读工具 `notion-fetch` 和参数 `id` 已在示例配置好。可运行 `python -m rag_agent.tools` 检查连接和工具参数。若使用自己的 MCP 服务，可配置 `NOTION_MCP_URL` 和相应的工具名；带 Bearer token 的 HTTP 连接可设置 `NOTION_MCP_TOKEN`。普通 Notion API token 不能直接当作托管 MCP token 使用。

### PDF

将可提取文字的 `.pdf` 文件放到 `data/pdfs/`，再运行导入命令。每页按来源保留文件名和页码。扫描件需要先做 OCR；当前版本不会识别 PDF 图片里的文字。网页没有 PDF 上传入口。

### 索引存储

导入后，向量索引保存在 `data/index/`；BM25 在服务启动加载索引时于内存中建立。`data/index/`、`data/pdfs/` 和 `.env` 均被 Git 忽略，不随代码推送。换电脑或重新部署时，需要提供文档、配置和授权，再运行导入。

## 项目结构

```text
agent/
├─ rag_agent/
│  ├─ app.py             # FastAPI、问答页、健康检查和接口
│  ├─ config.py          # 读取 .env 和默认配置
│  ├─ engine.py          # 文档切块、建索引、检索、回答和音频转写
│  ├─ ingest.py          # 手动执行全量导入的入口
│  ├─ live_audio.py      # 两路实时语音识别的 WebSocket 桥接
│  ├─ notion.py          # Notion MCP 读取及子页面遍历
│  ├─ retrieval.py       # 分词、RRF、语言和相关性判断
│  ├─ tools.py           # 查看当前 Notion MCP 读取工具
│  └─ static/            # 问答页和通话悬浮窗界面
├─ data/
│  ├─ pdfs/             # 待导入的 PDF
│  └─ index/            # 生成的本地索引
├─ tests/                # 检索、Notion 和音频测试
├─ .env.example          # 配置示例；复制成 .env 后填写密钥
├─ requirements.txt      # Python 依赖
└─ README.md
```

## 主要配置

| 配置 | 用途 | 默认值 |
| --- | --- | --- |
| `AIROBOT_LLM_API_KEY` | 云端模型密钥 | 必填 |
| `AIROBOT_LLM_BASE_URL` | OpenAI 兼容接口地址 | DashScope 北京地址 |
| `AIROBOT_LLM_MODEL` / `AIROBOT_ROUTER_MODEL` / `AIROBOT_RERANK_MODEL` / `EMBED_MODEL` / `TRANSCRIBE_MODEL` | 回答、路由、重排、向量和音频模型 | 见 `.env.example` |
| `NOTION_PAGE_IDS` | Notion 入口页面 ID，多个用逗号分隔 | Webpack 示例页 |
| `NOTION_INCLUDE_SUBPAGES` / `NOTION_MAX_PAGES` | 是否遍历子页面及页面数上限 | `true` / `1000` |
| `BM25_ENABLED` / `RERANK_ENABLED` | 关键词召回和云端重排 | `true` / `true` |
| `WEB_FALLBACK_ENABLED` / `TAVILY_API_KEY` | 是否允许网络搜索及网页提取 | `true` / 空 |
| `VECTOR_TOP_K` / `BM25_TOP_K` / `FUSION_TOP_K` / `RERANK_TOP_K` | 各阶段候选数量 | `8` / `8` / `10` / `4` |
| `RELEVANCE_THRESHOLD` | 内部文档检索的可靠性阈值 | `0.70` |
| `MAX_AUDIO_MB` | 单个音频文件大小上限 | `7` |
| `LIVE_ASR_MODEL` / `LIVE_ASR_WS_URL` | 通话模式的实时识别模型和 WebSocket 地址 | 见 `.env.example` |

默认开启快速模型重排，以避免向量相似度低估正确文档。知识库检索结果低于 `RELEVANCE_THRESHOLD` 时，如已启用网络搜索并配置 `TAVILY_API_KEY`，会继续搜索网页；否则明确说明搜索不可用，不会猜测答案。`WEB_FALLBACK_ENABLED=false` 可关闭网络搜索。相关性阈值应结合实际问题校准，开启重排前后评分含义不同。路由对明确线索使用快速规则，对模糊问题调用一次模型判断是否需要查资料，因此可能增加少量延迟。索引构建、问题向量化、重排和回答会产生云端 API 用量。

## 接口与测试

| 接口 | 说明 |
| --- | --- |
| `GET /` | 问答页 |
| `GET /health` | 索引就绪状态、分块数及检索开关 |
| `POST /api/ask` | JSON `{"question":"..."}`；返回文字答案、语言、路由原因、来源和耗时 |
| `POST /api/ask/stream` | JSON `{"question":"..."}`；逐行 NDJSON 返回答案片段和最终来源、耗时 |
| `POST /api/transcribe` | multipart `audio`；返回转写文字，确认后可作为问题发送 |
| `GET /live` | 通话模式页面 |
| `WS /api/live/asr/{role}` | `role` 为 `me` 或 `interviewer`；输入 PCM16/16kHz 音频块，输出实时转写事件 |
| `POST /api/live/question` | 判断面试官发言是否为问题；仅 AI 开关打开后调用 |

音频支持 mp3、wav、webm、ogg，输出始终为文字。运行测试：

```powershell
python -m pytest -q
```

## 常见问题

- **页面打开后提示没有索引**：先运行 `python -m rag_agent.ingest`，再启动服务。
- **新增页面后仍检索不到**：确认页面属于已授权的工作区、`NOTION_PAGE_IDS` 已更新；重新导入并重启服务。
- **Notion 授权失败**：检查 Node.js 与 `npx`、工作区访问权限，以及 `.env` 中的 MCP 配置。可先运行 `python -m rag_agent.tools` 验证连接。
- **扫描版 PDF 没有内容**：先用 OCR 生成可选中文字的 PDF。
- **没有检索到可靠资料**：检查文档是否已导入，或在需要网络回退时设置 `TAVILY_API_KEY`。

密钥只放在 `.env` 或部署环境变量中，不要提交到仓库。
