"""Two-stage routing: decide whether evidence is needed, then where to get it."""
import re
from dataclasses import dataclass
from ipaddress import ip_address
from urllib.parse import urlparse


URL = re.compile(r"https?://[^\s<>\"']+", re.I)
INTERNAL = re.compile(
    r"我的(?:文档|笔记|知识库|文件|资料|项目|仓库)|我们(?:的)?(?:文档|知识库|文件|资料|项目|公司|团队|仓库)|"
    r"这份文档|这篇文档|知识库|上传的(?:文档|文件|pdf)|notion\s*(?:文档|页面)|"
    r"\b(?:my|our|internal)\s+(?:docs?|documents?|notes?|files?|knowledge base|project|repo|repository|company|team)\b|"
    r"\b(?:this|the uploaded)\s+(?:document|pdf|file|repo|repository)\b",
    re.I,
)
FRESH = re.compile(
    r"现在|目前|最新|今年|今天|近期|最近|现任|现价|多少钱|价格|新闻|刚发布|是否还|还能否|"
    r"(?:20\d{2})年|\b(?:current|currently|latest|newest|today|this year|recent|now|price|pricing|news|as of|still supported)\b|"
    r"\bv?\d+\.\d+(?:\.\d+)?\b",
    re.I,
)
HIGH_STAKES = re.compile(
    r"医疗|诊断|药物|用药|法律|法规|税务|投资|金融|保险|选举|总统|首相|"
    r"\b(?:medical|diagnosis|medicine|medication|legal|law|tax|investment|finance|insurance|election|president|prime minister)\b",
    re.I,
)
STABLE = re.compile(
    r"数学|算法|原理|事件循环|复杂度|数据结构|MVCC|SSO|HMR|Tree.?Shaking|"
    r"\b(?:algorithm|data structure|event loop|how does|principle|complexity|protocol|sorting|binary search)\b",
    re.I,
)


@dataclass(frozen=True)
class NeedDecision:
    search: bool
    reason: str


def explicit_need(question: str) -> NeedDecision | None:
    """Fast, high-confidence rules; return None for ambiguous questions."""
    if URL.search(question):
        return NeedDecision(True, "supplied_url")
    if INTERNAL.search(question):
        return NeedDecision(True, "internal_data")
    if FRESH.search(question):
        return NeedDecision(True, "time_sensitive")
    if HIGH_STAKES.search(question):
        return NeedDecision(True, "high_stakes")
    if STABLE.search(question):
        return NeedDecision(False, "stable_concept")
    return None


def source_for(question: str, reason: str) -> tuple[str, str | None]:
    """Choose a source only after search has been deemed necessary."""
    match = URL.search(question)
    if match:
        url = match.group().rstrip(".,!?。，！？)")
        host = (urlparse(url).hostname or "").lower()
        if host in {"notion.so", "notion.site", "notion.com"} or host.endswith((".notion.so", ".notion.site", ".notion.com")):
            return "notion_url", url
        return "web_url", url
    if reason == "internal_data" or INTERNAL.search(question):
        return "knowledge_base", None
    return "web", None


def public_web_url(url: str) -> bool:
    """Never send local/private addresses to the external extraction service."""
    host = (urlparse(url).hostname or "").lower()
    if not host or host == "localhost" or host.endswith((".local", ".localhost")):
        return False
    try:
        return ip_address(host).is_global
    except ValueError:
        return True
