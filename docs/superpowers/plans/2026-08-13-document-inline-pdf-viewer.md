# 文档详情内嵌 PDF 阅读器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 结果页文档行详情（doc tab）删除「查看原文」按钮，改为 iframe 内嵌浏览器 PDF 阅读器直接渲染文档；非 PDF 文档诚实降级。

**Architecture:** 后端 `/uspto/download` 增加 `inline` 查询参数切换 `Content-Disposition: inline`（iframe 需要，attachment 会触发下载）；`_lift_download_url` 提升 downloadOptionBag 时偏好 PDF 选项；前端 DocTab 用纯函数 `buildDocPreview(url)` 决策 iframe / 降级 / 不可用三种模式。

**Tech Stack:** Python FastAPI + unittest；Next.js App Router（client component）+ 纯函数 node 测试（`node --test`）。

**Spec:** `docs/superpowers/specs/2026-08-13-document-inline-pdf-viewer-design.md`

## Global Constraints

- 分支：`feature/search-results-split-view`（当前 HEAD `6c9b0e8`）
- 后端测试命令（cwd `E:/online/workspace/copiioai/langsistance`）：`PYTHONUTF8=1 ./venv/Scripts/python.exe -m unittest tests.<file> -v`
- 前端测试命令（cwd `frontend/nextjs`）：`node --test lib/*.test.mjs`；类型检查 `npx tsc --noEmit`；构建 `npm run build`
- 每任务末尾独立 commit（conventional commits，无 Co-Authored-By 尾注——本项目全局关闭 attribution）
- 不可变更新（返回新对象，不修改入参）；无 console.log；遵循周边代码风格
- i18n 文案中英两份必须同步（`lib/app-i18n/locales/zh.ts` 与 `en.ts`）

---

### Task 1: `_lift_download_url` 偏好 PDF 选项

**Files:**
- Modify: `sources/result_export.py`（函数 `_lift_download_url`，约 96-111 行区域）
- Test: `tests/test_result_export_roles.py`

**Interfaces:**
- Consumes: 无（独立任务）
- Produces: `downloadUrl` 顶层键（前端 `buildRowModel.url` 读取；行下载按钮与 Task 4 iframe 共用）。PDF 选项存在时必选 PDF；否则回退第一个非空 URL；无选项时不产生键。

- [ ] **Step 1: 写失败测试**

在 `tests/test_result_export_roles.py` 的 `TestBuildResultArtifactsJson` 类里、`test_document_items_skip_empty_download_url_options` 之后追加：

```python
    def _doc_item_with_options(self, options):
        item = self._document_item(None)
        item["downloadOptionBag"] = options
        return item

    def _json_payload_of(self, items):
        artifacts = build_result_artifacts(items, source="uspto_documents")
        return json.loads(
            next(a for a in artifacts if a["format"] == "json")["content"]
            .decode("utf-8")
        )

    def test_prefers_pdf_option_over_earlier_docx(self):
        items = [
            self._doc_item_with_options([
                {"mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                 "downloadUrl": "https://api.copiioai.com/uspto/download?url=docx"},
                {"mimeType": "application/pdf",
                 "downloadUrl": "https://api.copiioai.com/uspto/download?url=pdf"},
            ])
        ] * 6
        payload = self._json_payload_of(items)
        self.assertEqual(
            payload["rows"][0]["downloadUrl"],
            "https://api.copiioai.com/uspto/download?url=pdf",
        )

    def test_pdf_detected_by_mime_type_identifier(self):
        items = [
            self._doc_item_with_options([
                {"mimeTypeIdentifier": "application/pdf",
                 "downloadUrl": "https://api.copiioai.com/uspto/download?url=by-mime"},
            ])
        ] * 6
        payload = self._json_payload_of(items)
        self.assertEqual(
            payload["rows"][0]["downloadUrl"],
            "https://api.copiioai.com/uspto/download?url=by-mime",
        )

    def test_pdf_detected_by_url_extension_without_mime(self):
        items = [
            self._doc_item_with_options([
                {"downloadUrl": "https://api.copiioai.com/uspto/download?url=a"},
                {"downloadUrl": "https://example.com/file.PDF"},
            ])
        ] * 6
        payload = self._json_payload_of(items)
        self.assertEqual(payload["rows"][0]["downloadUrl"], "https://example.com/file.PDF")

    def test_falls_back_to_first_url_when_no_pdf_option(self):
        items = [
            self._doc_item_with_options([
                {"mimeType": "application/msword",
                 "downloadUrl": "https://example.com/file.doc"},
                {"mimeType": "application/xml",
                 "downloadUrl": "https://example.com/file.xml"},
            ])
        ] * 6
        payload = self._json_payload_of(items)
        self.assertEqual(payload["rows"][0]["downloadUrl"], "https://example.com/file.doc")
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 ./venv/Scripts/python.exe -m unittest tests.test_result_export_roles.TestBuildResultArtifactsJson.test_prefers_pdf_option_over_earlier_docx tests.test_result_export_roles.TestBuildResultArtifactsJson.test_pdf_detected_by_mime_type_identifier tests.test_result_export_roles.TestBuildResultArtifactsJson.test_pdf_detected_by_url_extension_without_mime tests.test_result_export_roles.TestBuildResultArtifactsJson.test_falls_back_to_first_url_when_no_pdf_option -v`
Expected: 前两个 FAIL（当前实现取第一个非空 URL，docx 先到先得）；后两个应 PASS（无 mime 时当前实现也取第一个 —— 第三个取 "…url=a" 会 FAIL；第四个 PASS）。实际以「至少两个 FAIL」为准。

- [ ] **Step 3: 实现**

替换 `sources/result_export.py` 中 `_lift_download_url` 整体为：

```python
def _lift_download_url(item: dict[str, Any]) -> dict[str, Any]:
    """Lift the best ``downloadOptionBag[].downloadUrl`` to a top-level key.

    PDF options are preferred — the frontend renders PDFs inline in an
    embedded reader (DOCX/XML cannot render in an iframe).  Falls back to
    the first non-empty URL when no PDF option exists.  ``downloadUrl``
    maps to the ``url`` role via :data:`_ROLE_SUFFIXES`.
    """
    options = item.get("downloadOptionBag")
    if not isinstance(options, list):
        return item
    fallback: str | None = None
    for option in options:
        if not isinstance(option, dict):
            continue
        download_url = option.get("downloadUrl")
        if not isinstance(download_url, str) or not download_url:
            continue
        if fallback is None:
            fallback = download_url
        mime = str(
            option.get("mimeTypeIdentifier", "") or option.get("mimeType", "")
        ).lower()
        if "pdf" in mime or download_url.lower().split("?", 1)[0].endswith(".pdf"):
            return {**item, "downloadUrl": download_url}
    if fallback is not None:
        return {**item, "downloadUrl": fallback}
    return item
```

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 ./venv/Scripts/python.exe -m unittest tests.test_result_export_roles -v`
Expected: 全部 PASS（13 个测试，含既有回归）

- [ ] **Step 5: Commit**

```bash
git add sources/result_export.py tests/test_result_export_roles.py
git commit -m "feat: prefer PDF option when lifting downloadOptionBag URL"
```

---

### Task 2: `/uspto/download` 支持 inline 模式

**Files:**
- Modify: `api_routes/uspto.py`（`download_uspto_file` handler）
- Test: `tests/test_uspto_download_route.py`

**Interfaces:**
- Consumes: 无
- Produces: `GET /uspto/download?url=...&inline=1` → `Content-Disposition: inline; filename="..."`；缺省 `inline` → `attachment`（行为不变）。Task 4 的 iframe src 依赖此参数。

- [ ] **Step 1: 写失败测试**

在 `tests/test_uspto_download_route.py` 末尾（`if __name__ == "__main__"` 之前）追加：

```python
class TestUsptoDownloadInlineMode(unittest.IsolatedAsyncioTestCase):
    """Route handler tests — fetch_uspto_download_file is patched out."""

    def _fake_download_file(self):
        class FakeDownloadFile:
            content = b"%PDF-1.4 fake"
            media_type = "application/pdf"
            filename = "document.pdf"
        return FakeDownloadFile()

    async def test_inline_param_sets_inline_content_disposition(self):
        from unittest.mock import patch
        from api_routes.uspto import download_uspto_file

        with patch(
            "api_routes.uspto.fetch_uspto_download_file",
            return_value=self._fake_download_file(),
        ):
            response = await download_uspto_file(
                url="https://api.uspto.gov/api/v1/download/applications/18244278/documents/file.pdf",
                inline=True,
            )
        self.assertEqual(response.media_type, "application/pdf")
        self.assertIn(
            'inline; filename="document.pdf"',
            response.headers["Content-Disposition"],
        )

    async def test_default_disposition_is_attachment(self):
        from unittest.mock import patch
        from api_routes.uspto import download_uspto_file

        with patch(
            "api_routes.uspto.fetch_uspto_download_file",
            return_value=self._fake_download_file(),
        ):
            response = await download_uspto_file(
                url="https://api.uspto.gov/api/v1/download/applications/18244278/documents/file.pdf",
            )
        self.assertIn(
            'attachment; filename="document.pdf"',
            response.headers["Content-Disposition"],
        )
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONUTF8=1 ./venv/Scripts/python.exe -m unittest tests.test_uspto_download_route.TestUsptoDownloadInlineMode -v`
Expected: FAIL —— `download_uspto_file` 不接受 `inline` 关键字（TypeError）。

- [ ] **Step 3: 实现**

`api_routes/uspto.py`：

1. handler 签名增加参数（`Query` 已从 fastapi 导入）：

```python
@router.get("/uspto/download")
async def download_uspto_file(
    url: str = Query(..., min_length=1),
    inline: bool = Query(False),
):
```

2. 末尾返回处，把固定 `attachment` 改为按参数切换：

```python
    disposition = "inline" if inline else "attachment"
    logger.info(f"USPTO lazy download proxied: {download_file.filename}")
    return Response(
        content=download_file.content,
        media_type=download_file.media_type,
        headers={
            "Content-Disposition": f'{disposition}; filename="{download_file.filename}"'
        },
    )
```

（其余 try/except 逻辑一行不动。）

- [ ] **Step 4: 运行确认通过**

Run: `PYTHONUTF8=1 ./venv/Scripts/python.exe -m unittest tests.test_uspto_download_route -v`
Expected: 全部 PASS（8 个测试：6 个既有 fetch 逻辑 + 2 个新 inline 测试）

- [ ] **Step 5: Commit**

```bash
git add api_routes/uspto.py tests/test_uspto_download_route.py
git commit -m "feat: inline mode for /uspto/download proxy"
```

---

### Task 3: 前端纯函数 `buildDocPreview`

**Files:**
- Create: `frontend/nextjs/lib/docPreview.js`
- Create: `frontend/nextjs/lib/docPreview.test.mjs`

**Interfaces:**
- Consumes: `row.url`（Task 1 产出的 downloadUrl 列值；多数为代理形式 `/uspto/download?url=<编码上游>`）
- Produces: `buildDocPreview(url)` → `{ mode: 'iframe', src }` | `{ mode: 'fallback', url }` | `{ mode: 'unavailable' }`。Task 4 消费。

- [ ] **Step 1: 写失败测试**

创建 `frontend/nextjs/lib/docPreview.test.mjs`：

```js
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { buildDocPreview } from './docPreview.js'

const PROXY = 'https://api.copiioai.com/uspto/download'

test('proxy url with inner pdf builds iframe src with inline param', () => {
  const url = `${PROXY}?url=${encodeURIComponent('https://api.uspto.gov/api/v1/download/applications/1/file.pdf')}`
  const preview = buildDocPreview(url)
  assert.equal(preview.mode, 'iframe')
  assert.equal(preview.src, `${url}&inline=1`)
})

test('proxy url with inner docx falls back', () => {
  const url = `${PROXY}?url=${encodeURIComponent('https://api.uspto.gov/api/v1/download/applications/1/file.docx')}`
  const preview = buildDocPreview(url)
  assert.equal(preview.mode, 'fallback')
  assert.equal(preview.url, url)
})

test('plain pdf url embeds without inline param', () => {
  const preview = buildDocPreview('https://example.com/patent.pdf?token=1')
  assert.equal(preview.mode, 'iframe')
  assert.equal(preview.src, 'https://example.com/patent.pdf?token=1')
})

test('plain docx url falls back', () => {
  const preview = buildDocPreview('https://example.com/patent.docx')
  assert.equal(preview.mode, 'fallback')
})

test('empty url is unavailable', () => {
  assert.deepEqual(buildDocPreview(''), { mode: 'unavailable' })
  assert.deepEqual(buildDocPreview(null), { mode: 'unavailable' })
})

test('proxy url without inner url param falls back', () => {
  const preview = buildDocPreview(`${PROXY}`)
  assert.equal(preview.mode, 'fallback')
})
```

- [ ] **Step 2: 运行确认失败**

Run: `node --test lib/docPreview.test.mjs`（cwd `frontend/nextjs`）
Expected: FAIL —— `Cannot find module './docPreview.js'`

- [ ] **Step 3: 实现**

创建 `frontend/nextjs/lib/docPreview.js`：

```js
/**
 * Document preview decision for document rows in the results page.
 * Pure function — the iframe/fallback/unavailable choice is derived only
 * from the row URL so it stays unit-testable.
 */

function extractInnerUrl(proxyUrl) {
  try {
    const inner = new URL(proxyUrl).searchParams.get('url')
    if (!inner) return null
    try {
      return decodeURIComponent(inner)
    } catch {
      return inner // already decoded or malformed — keep raw value
    }
  } catch {
    const match = /[?&]url=([^&]+)/.exec(proxyUrl)
    if (!match) return null
    try {
      return decodeURIComponent(match[1])
    } catch {
      return match[1]
    }
  }
}

function isPdfPath(url) {
  try {
    return new URL(url).pathname.toLowerCase().endsWith('.pdf')
  } catch {
    return url.toLowerCase().split('?')[0].endsWith('.pdf')
  }
}

export function buildDocPreview(url) {
  if (!url || typeof url !== 'string') return { mode: 'unavailable' }

  if (url.includes('/uspto/download')) {
    const inner = extractInnerUrl(url)
    if (!inner) return { mode: 'fallback', url }
    return isPdfPath(inner)
      ? { mode: 'iframe', src: `${url}&inline=1` }
      : { mode: 'fallback', url }
  }

  return isPdfPath(url)
    ? { mode: 'iframe', src: url }
    : { mode: 'fallback', url }
}
```

- [ ] **Step 4: 运行确认通过**

Run: `node --test lib/docPreview.test.mjs`
Expected: 6/6 PASS

- [ ] **Step 5: Commit**

```bash
git add lib/docPreview.js lib/docPreview.test.mjs
git commit -m "feat: buildDocPreview pure function for document preview modes"
```

---

### Task 4: DocTab 重写 + i18n + CSS

**Files:**
- Modify: `frontend/nextjs/components/app/results/DocTab.tsx`
- Modify: `frontend/nextjs/lib/app-i18n/locales/zh.ts`（`results` 对象内 `rowDocDownload: '下载',` 行之后）
- Modify: `frontend/nextjs/lib/app-i18n/locales/en.ts`（`results` 对象内 `rowDocDownload: 'Download',` 行之后）
- Modify: `frontend/nextjs/styles/app.css`（`.results-doc-link` 规则之后追加）

**Interfaces:**
- Consumes: `buildDocPreview`（Task 3）、`/uspto/download?inline=1`（Task 2）、`row.url`（Task 1）
- Produces: 无对外接口

- [ ] **Step 1: 重写 DocTab.tsx**

整文件替换为：

```tsx
'use client'

import { useI18n } from '@/lib/app-i18n'
import { buildDocPreview } from '@/lib/docPreview'

export default function DocTab({ row }: { row: any }) {
  const { t } = useI18n()
  const preview = buildDocPreview(row.url)

  return (
    <div className="results-detail-card">
      <h3>{row.title}</h3>
      {row.meta.map((item: { label: string; value: string }) => (
        <div key={item.label} className="results-doc-meta">
          <span className="results-field-label">{item.label}</span>
          <span>{item.value}</span>
        </div>
      ))}
      {preview.mode === 'iframe' && (
        <iframe
          className="results-doc-frame"
          src={preview.src}
          title={row.title || 'document preview'}
        />
      )}
      {preview.mode === 'fallback' && (
        <div className="results-error">
          <p>{t('results.docNoPdf')}</p>
          <a href={preview.url} download rel="noopener noreferrer">
            {t('results.docPdfFallbackDownload')}
          </a>
        </div>
      )}
      {preview.mode === 'unavailable' && (
        <div className="results-error">
          <p>{t('results.docUnavailable')}</p>
        </div>
      )}
    </div>
  )
}
```

（原「查看原文」`<a className="results-doc-link">` 链接删除。）

- [ ] **Step 2: i18n 文案（两份）**

`zh.ts` 在 `rowDocDownload: '下载',` 后加：

```ts
    docNoPdf: '该文档无 PDF 版本，请下载后查看',
    docPdfFallbackDownload: '下载原文',
    docUnavailable: '文档不可用',
```

`en.ts` 在 `rowDocDownload: 'Download',` 后加：

```ts
    docNoPdf: 'No PDF version available — download the original file',
    docPdfFallbackDownload: 'Download original',
    docUnavailable: 'Document unavailable',
```

- [ ] **Step 3: CSS**

在 `styles/app.css` 的 `.results-doc-link { ... }` 规则块之后追加：

```css
.results-doc-frame {
  width: 100%;
  height: 70vh;
  margin-top: 10px;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  background: #525659;
}
```

- [ ] **Step 4: 验证**

Run（cwd `frontend/nextjs`）：
1. `node --test lib/*.test.mjs` — Expected: 全部 PASS（含既有 16 个测试文件）
2. `npx tsc --noEmit` — Expected: 无输出、退出码 0
3. `npm run build` — Expected: 构建成功（nginx 部署需要 out/ 产物）

- [ ] **Step 5: Commit**

```bash
git add components/app/results/DocTab.tsx lib/app-i18n/locales/zh.ts lib/app-i18n/locales/en.ts styles/app.css
git commit -m "feat: inline PDF reader in document detail tab"
```

---

## 手动验证清单（测试环境）

部署（后端 + 前端都要）：

```bash
# 服务器
git pull
docker compose --profile backend up -d --build && docker compose restart celery-worker
cd frontend/nextjs && npm run build   # nginx root 指向 out/
# 浏览器 Disable cache + Ctrl+Shift+R
```

1. 检索文档列表 → 点击文档行 → doc tab 显示内嵌 PDF，可直接翻页滚动，无「查看原文」按钮
2. 列表行「下载」按钮仍正常下载 PDF
3. （可选）构造只有 DOCX 的文档 → doc tab 显示「该文档无 PDF 版本」+ 下载原文链接
4. 旧会话恢复（刷新页面）→ 文档行 doc tab 仍可渲染（持久化裁剪保留 downloadUrl 列）
