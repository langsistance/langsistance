"""design_image 页→PDF 直链与降级 (design P1 T4) — 契约测试。

- fetch_page / pdf_fetch 全程 mock (异步注入), 无真实网络, 无 httpx 调用。
- fixture HTML 只用通用 D 号样例 (brief 复刻), 生产代码零产品词, extract 零产品词。
- 覆盖: extract_pdf_url 首命中/缺失; fetch_design_pdf 页地址拼接 + PDF 降级分支。
"""
import asyncio

from sources.design.design_image import (
    DESIGN_PAGE_URL,
    extract_pdf_url,
    fetch_design_pdf,
)

# brief UTF 样例 (fixture HTML: patentimages US.pdf 首命中; 后有非 patentimages png)。
HTML = ('<img src="https://patentimages.storage.googleapis.com/fa/21/81/'
        '670fbfd00d78ff/USD504889.pdf"><img src="https://example.com/other.png">')

_PDF_BYTES = b"%PDF-1.4 mock empty pdf"

# ---------- extract_pdf_url ----------

def test_extract_first_pdf():
    assert extract_pdf_url(HTML).endswith("USD504889.pdf")


def test_extract_none_when_missing():
    assert extract_pdf_url("<html>no image</html>") is None


def test_extract_ignores_non_patentimages():
    # .png 及非 patentimages 主机不命中 .pdf 规则 → None
    assert extract_pdf_url('<img src="https://example.com/a.pdf">') is None


def test_extract_none_on_empty_or_none():
    assert extract_pdf_url("") is None
    assert extract_pdf_url(None) is None  # type: ignore[arg-type]


# ---------- DESIGN_PAGE_URL ----------

def test_page_url_format():
    assert DESIGN_PAGE_URL.format(pid="USDA1") == "https://patents.google.com/patent/USDA1/en"


async def _ok_page(pid):
    return 200, HTML


async def _ok_pdf(url):  # noqa: ARG001 (mock 只回字节)
    return 200, _PDF_BYTES


def test_page_url_uses_pid():
    seen = {}

    async def page(pid):
        seen["pid"] = pid
        return 200, HTML

    async def pdf(url):
        seen["url"] = url
        return 200, _PDF_BYTES

    data = asyncio.run(fetch_design_pdf(page, "USD504889", pdf_fetch=pdf))
    assert data == _PDF_BYTES
    assert seen["pid"] == "USD504889"
    assert seen["url"].endswith("USD504889.pdf")


# ---------- fetch_design_pdf 降级 ----------

def test_fetch_degrade_page_404():
    async def bad(pid):  # noqa: ARG001
        return 404, "nope"

    assert asyncio.run(fetch_design_pdf(bad, "USD1A")) is None


def test_degrade_when_page_no_pdf():
    async def page(pid):  # noqa: ARG001
        return 200, "<html>no drawings</html>"

    assert asyncio.run(fetch_design_pdf(page, "USD1B")) is None


def test_degrade_when_pdf_http_error():
    async def pdf(url):  # noqa: ARG001
        return 404, b"gone"

    assert asyncio.run(fetch_design_pdf(_ok_page, "USD1C", pdf_fetch=pdf)) is None


def test_degrade_when_pdf_too_large():
    big = b"x" * (2 * 1024 * 1024 + 1)  # 严格 >2MB

    async def pdf(url):  # noqa: ARG001
        return 200, big

    assert asyncio.run(fetch_design_pdf(_ok_page, "USD1D", pdf_fetch=pdf)) is None


def test_accepts_pdf_at_threshold():
    at_size = b"x" * (2 * 1024 * 1024)  # ==2MB (未超阈值, 允许)

    async def pdf(url):  # noqa: ARG001
        return 200, at_size

    data = asyncio.run(fetch_design_pdf(_ok_page, "USD1E", pdf_fetch=pdf))
    assert len(data) == 2 * 1024 * 1024
