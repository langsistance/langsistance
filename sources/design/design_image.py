"""design_image 页→PDF 直链与降级 (design P1 T4)。

对应 spec: docs/superpowers/specs/2026-09-06-us-design-clearance-design.md §5.2;
task-4-brief.md。
- extract_pdf_url: 从 Google Patents 设计专利页 HTML 抽首个 patentimages …USD….pdf
  附图 PDF 直链; 无 → None。纯函数零 IO, 零产品词。
- fetch_design_pdf: (fetch_page, pid) 先拉页面 → 抽 PDF → GET PDF 二进制。
  页非 200 / 页面无 pdf / PDF 非 200 / PDF >2MB → None (Task 7 降级"文本维度候选")。
  fetch_page(pid)->(status, html) 由调用方注入; pdf_fetch(url)->(status, bytes) 为可选注入,
  缺省走 httpx 生产包装。本模块只拼 DESIGN_PAGE_URL + 判定, 测试恒注入 mock 禁真实网络。
"""
import re

_PDF_RE = re.compile(r"https://patentimages\.storage\.googleapis\.com/[^\"'\s]+\.pdf")
_MAX_PDF_BYTES = 2 * 1024 * 1024   # PDF >2MB 截断返回 None

DESIGN_PAGE_URL = "https://patents.google.com/patent/{pid}/en"


def extract_pdf_url(html: str | None) -> str | None:
    """返回 HTML 中首 patentimages …pdf 直链; 无/空 → None。"""
    if not html:
        return None
    match = _PDF_RE.search(html)
    return match.group(0) if match else None


async def _fetch_pdf_http(url: str) -> tuple[int, bytes]:
    """生产默认 pdf_fetch: httpx GET → (status, bytes); 网络异常视 status 0。"""
    import httpx
    try:
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        return resp.status_code, resp.content
    except httpx.HTTPError:
        return 0, b""


async def fetch_design_pdf(fetch_page, pid: str, pdf_fetch=None) -> bytes | None:
    """页→附图 PDF 直链→PDF 二进制; 任一环节非 200 / 无 pdf / PDF>2MB → None。

    fetch_page(pid) -> (status, html)。pdf_fetch(url) -> (status, bytes),
    缺省时使用 httpx 全流程去拉真实页与 PDF (响应式: 若 fetch_page 已返回完整 html,
    仍以其 html 经 extract_pdf_url 取直链, 复用同一注入路径)。
    """
    fetch_pdf = pdf_fetch if pdf_fetch is not None else _fetch_pdf_http
    result = await fetch_page(pid)
    status, html = result if isinstance(result, tuple) else (result, "")
    if status != 200:
        return None
    pdf_url = extract_pdf_url(html)
    if not pdf_url:
        return None
    pdf_status, body = await fetch_pdf(pdf_url)
    if pdf_status != 200 or len(body) > _MAX_PDF_BYTES:
        return None
    return bytes(body)
