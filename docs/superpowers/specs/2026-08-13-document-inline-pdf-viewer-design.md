# 文档详情内嵌 PDF 阅读器设计（DocTab 改造）

**日期:** 2026-08-13
**分支:** feature/search-results-split-view
**前置:** 分屏视图（[search-results-split-view-design](./2026-08-13-search-results-split-view-design.md)）已上线；下载按钮修复（1ff1c14）已验证

---

## 一、背景与问题

结果页文档行的详情面板（doc tab，`DocTab.tsx`）当前只展示标题/元数据 + 「查看原文」链接，点击在新标签打开或触发下载。

用户反馈：

1. 该按钮与列表行上的「下载」按钮功能重复
2. 期望详情面板**直接展示文档内容**，而不是跳走或下载

**关键事实**：USPTO 文档 PDF 多数为扫描件，文本提取（qpdf/pdftotext）大概率失败。因此放弃"提取文本分段展示"路线，采用**浏览器内嵌 PDF 阅读器**——浏览器原生渲染 PDF（含扫描件），成功率最高。

## 二、目标

1. 删除 doc tab 的「查看原文/view original」按钮
2. doc tab 内嵌 PDF 阅读器，直接渲染文档内容
3. 非 PDF 文档诚实降级：提示 + 「下载原文」链接兜底（仅降级时出现）

## 三、反目标

- 不做 PDF 文本提取 / OCR（扫描件提取不可靠、慢）
- 不改列表行「查看 / 下载」按钮（用户仅要求删详情里的）
- 不改专利行逻辑、spec/claims tab
- 不破坏 CSV/XLSX 下载行为。注：`_lift_download_url` 作用于所有格式，CSV/XLSX 会同步**新增**一列 `downloadUrl`（加性变更，不删除任何原有列）——2026-08-13 终审发现后经用户确认接受
- 不新增 SSE 事件类型

## 四、设计

### 4.1 后端：`/uspto/download` 支持内嵌模式

`api_routes/uspto.py` 增加可选查询参数 `inline: bool = False`：

- `inline=True` → `Content-Disposition: inline; filename="..."`（iframe 内渲染）
- 缺省 / `False` → 保持现状 `attachment`（下载行为不变）

理由：iframe 里的 `attachment` 会强制触发下载而非渲染。其余逻辑（代理下载、400/502 错误处理）完全不动。

### 4.2 后端：`_lift_download_url` 偏好 PDF

`sources/result_export.py::_lift_download_url` 当前取 `downloadOptionBag` 第一个非空 `downloadUrl`。改为：

1. 优先选 PDF 选项：`mimeType` / `mimeTypeIdentifier` 含 `pdf`，或 `downloadUrl` 以 `.pdf` 结尾
2. 无 PDF 选项 → 回退第一个非空 `downloadUrl`
3. 无任何选项 → 不产生列（现状）

行「下载」按钮与 iframe 共用此 URL —— 下载 PDF 对用户同样合理。

### 4.3 前端：DocTab 重写

- **删除**「查看原文」链接
- 顶部一行保留：文档标题 + 日期元数据
- 主体：内嵌阅读器，类 `results-doc-frame`，高约 70vh、宽 100%、内部滚动
- 预览决策抽为纯函数 `lib/docPreview.js::buildDocPreview(url)`（便于测试）：
  - **URL 形态注意**：行上的 `row.url` 是代理形式 `https://api.copiioai.com/uspto/download?url=<编码后的上游 URL>`，`.pdf` 后缀在编码后的内层参数里，必须解码后判断
  - 判定逻辑：
    1. `url` 为空 → `{ mode: 'unavailable' }`
    2. url 含 `/uspto/download` → 解码内层 `url` 参数，取其 pathname（忽略 query）：
       - 以 `.pdf` 结尾 → `{ mode: 'iframe', src: url + '&inline=1' }`
       - 否则 → `{ mode: 'fallback', url }`
    3. 普通 URL（未走代理的少数情况）→ 取自身 pathname：
       - 以 `.pdf` 结尾 → `{ mode: 'iframe', src: url }`（不追加 inline，参数无意义）
       - 否则 → `{ mode: 'fallback', url }`
- `mode: 'iframe'` → 渲染 `<iframe src={preview.src}>`
- `mode: 'fallback'` → 提示「该文档无 PDF 版本」+ 「下载原文」链接（`download` 属性）+ 重试按钮
- `mode: 'unavailable'` → 提示「文档不可用」
- iframe 加载失败无法跨域检测 → 不做失败检测，仅 URL 启发式 + 降级

### 4.4 i18n 与样式

- `lib/app-i18n/locales/zh.ts` / `en.ts` 新增：
  - `results.docNoPdf`：「该文档无 PDF 版本，请下载后查看」/ "No PDF version available — download the original file"
  - `results.docPdfFallbackDownload`：「下载原文」/ "Download original"
  - `results.docUnavailable`：「文档不可用」/ "Document unavailable"
- 全局 CSS 增加 `.results-doc-frame`（宽 100%、高 70vh、边框圆角、深色背景）

## 五、数据流

```
列表点击文档行 → DetailPanel(doc tab) → DocTab 挂载
  → iframe 加载 /uspto/download?url=<encoded>&inline=1
  → 后端代理下载 USPTO 文件（X-API-KEY）
  → Content-Disposition: inline 流式返回
  → 浏览器原生 PDF 查看器渲染
```

## 六、错误处理

| 场景 | 行为 |
|---|---|
| 后端下载失败（400/502） | iframe 内浏览器错误页（无法自定义）→ 可接受 |
| 文档只有 DOCX/XML 选项 | `_lift` 回退第一个 URL → 前端启发式识别扩展名 → 降级视图 |
| row.url 为空 | 降级视图「文档不可用」 |
| 扫描件 PDF | 浏览器直接渲染，无提取环节，天然支持 |

## 七、测试

- 后端 `tests/test_uspto_routes.py`（或既有 uspto 测试文件）：`inline=1` → `Content-Disposition: inline`；缺省 → `attachment`
- 后端 `tests/test_result_export_roles.py`：PDF 偏好（PDF 选项排后仍被选中）、无 PDF 回退第一个、选项为空跳过
- 前端 `lib/docPreview.test.mjs`：`buildDocPreview`（代理 URL 内层 .pdf / 代理 URL 内层 .docx / 普通 .pdf URL / 空 url / 代理 URL 无内层参数）
- 手动：测试环境验证 PDF 内嵌渲染 + 非 PDF 降级视图

## 八、影响面

| 文件 | 改动 |
|---|---|
| `api_routes/uspto.py` | +`inline` 参数（~6 行） |
| `sources/result_export.py` | `_lift_download_url` PDF 偏好（~20 行）；CSV/XLSX 同步新增 `downloadUrl` 列（加性，经确认） |
| `frontend/nextjs/components/app/results/DocTab.tsx` | 重写（~70 行） |
| `frontend/nextjs/lib/docPreview.js` | 新增纯函数（~15 行） |
| `frontend/nextjs/lib/app-i18n/locales/{zh,en}.ts` | +3 条文案 |
| 全局 CSS | `.results-doc-frame` |
| 测试 | 后端 2 处 + 前端 1 处 |

无数据库/配置/部署脚本变更。前端改动需重新 `npm run build`；后端改动需重建 docker。
