# 佰腾（Baiten）开放平台 — 中国专利自然语言检索接入技术方案

> 日期: 2026-08-24 | 状态: 待评审（含 6 项待用户确认事项） | 前置: 需要用户提供 BAITEN_APP_KEY/APP_SECRET

---

## 一、背景与目标

现有自然语言专利检索以 USPTO 为唯一数据源（对话 → 检索式阶梯 → USPTO applications/search → 结果页四 tab 查看）。已有中国专利通道为 DI 平台（zldsj，2026-06 上线，`docs/china-patent-launch-plan.md`），但 **zldsj 通道缺少「说明书/权利要求原始文档」查看能力**（结果页 spec/claims tab 仅支持 uspto/google_patents）。

目标：利用佰腾开放平台接口，在现有自然语言检索框架内增加**中国专利检索 + 原始文档查看**能力：

1. 复用现有自然语言检索链路（概念提取 → 检索式阶梯 → 场景工具）
2. 支持查看说明书（说明书查询）与权利要求（权利要求查询）原始文档
3. **不做**审查历史分析（FSWX 复审无效/审查意见）——law 的 FLZT 法律状态可用于展示/过滤，属可选

## 二、现状链路（代码事实）

```
对话 → long task PHASE0
  ├─ search_query_builder.py: Flash LLM 提取概念+英文关键词+载体词（强制英文）
  │    → _assemble_ladder 确定性组装由紧到松检索式阶梯（直译词版+载体词版交错）
  │    → 检索工具调用 USPTO
  ├─ scene_tools.py:279 「专利来源: cnipa → 优先 zldsj/cnipa 工具」路由提示
  ├─ core.py:127-166 infer_patent_source: 文本关键词/公司名 → uspto|cnipa|auto
  └─ 详情: api_routes/patent_detail.py VALID_SOURCES={"uspto","google_patents"}
       spec → PDF 代理 URL（/uspto/download lazy-download 模式）
       claims → 结构化列表或 PDF URL（前端内嵌查看）
```

关键复用点：
- **前端零改动**：`frontend/nextjs/services/api.ts:335-344` 按 `source` 拼接 `/patent/{source}/{id}/spec|claims`，加 `"baiten"` 即全通
- **签名客户端已存在**：`sources/baiten_client.py` 完整实现 TOP 式 MD5 签名 + 路径路由（`POST http://open.baiten.cn/router/{methodPath}`，method 参与签名但不进 body），当前仅 law 方法、**未接线**
- **配置已支持**：`sources/long_task/config.py:145-166` BAITEN_APP_KEY/APP_SECRET/GATEWAY_URL（env 或 config.ini [BAITEN] 段）

## 三、佰腾接口能力清单

| 接口 | SDK Request | 入参 | 返回关键字段 | 本方案 |
|---|---|---|---|---|
| 基础检索 | CubeOpenSearchRequest | queryString, source=15(中国) | fieldValues{id,an,ad,pn,pd,ti,pa} + hlFieldValues | ✅ 必接 |
| 基础信息 | CubeOpenGetDocRequest | docId | 完整著录项（ab/ic1/ic2/agt/agc/co/pat/type） | ✅ 必接（补 an/pd 给 PDF） |
| 权利要求 | CubeOpenGetClaimsRequest | appNum, patType(APP/AUTH) | patentClaimses[] 层级{claim, claimsNum, claimsParentNum} | ✅ 必接 |
| 说明书 | CubeOpenGetSpecRequest | docId（CN 用申请号） | 说明书内容 | ✅ 必接 |
| PDF 全文 | CubeOpenFileRequest | pubNum, pubDate(YYYYMMDD), fileCategory=PDF | base64 fileByte | ✅ 必接（PDF 查看） |
| 法律信息 | CubeOpenGetLawRequest | appNum, lawCategory | 法律状态等 6 类 | 🔵 已实现（law），FLZT 接线可选 |
| 增值信息 | CubeOpenIncrementRequest | — | 当前权利人/代理机构 | 🔵 可选 |
| 同族/相似/统计分析类 | — | — | — | ❌ 不需要（同族可用现有 family 逻辑替代） |

检索式语法（基础检索）：字段前缀 `ti:/ab:/clm:/pa:/in:/an:/pn:/ad:/pd:/ic1:/ic2:`，逻辑符 AND/OR/-/`""`/`()`，词尾通配符 `auto*`。

## 四、总体数据流

```
用户提问（自然语言，中/英）
  │
  ▼
search_query_builder（CN 变体，见 5.4）→ 检索式阶梯（中文同义词 OR 组）
  │
  ▼
场景工具「中国专利检索（佰腾）」（MySQL tools 条目，URL=/baiten/search）
  │ POST /baiten/search { query_string }
  ▼
BaitenClient.search() → source=15 中国库
  │ fieldValues{id,an,ad,pn,pd,ti,pa}
  ▼
映射 candidate 结构 { patent_id=pn, source="baiten", title=ti, pub_date=pd, ... }
  │
  ▼
结果页四 tab（前端零改动）
  ├─ details: 检索返回著录项直出（或 getDoc 补全）
  ├─ spec:  GET /patent/baiten/{pn}/spec
  │           → getDoc 补 an/pd → getSpec(an) 或 getFile(pubNum,pubDate,PDF)
  │           → 说明书文本 / PDF 代理 URL（/baiten/download，流式）
  ├─ claims: GET /patent/baiten/{pn}/claims
  │           → getDoc 补 an → getClaims(an, AUTH→APP 回退)
  │           → 扁平化映射现有 claims payload（number/text/independent）
  └─ doc:    PDF 原文 / 说明书查询文本
```

## 五、详细设计

### 5.1 BaitenClient 扩展（`sources/baiten_client.py`）

重构 `_build_top_params` 为通用签名构造器，新增 5 个方法：

```python
async def search(self, query_string: str, source: str = "15", page: int = 1,
                 page_size: int = 20, api_level: str = "ONE") -> dict
    # method="/openService/search", 参数: queryString, source, page, pageSize, apiLevel
async def get_doc(self, doc_id: str) -> dict
    # method="/openService/getDoc", 参数: docId
async def get_claims(self, app_num: str, pat_type: str = "APP") -> dict
    # method="/openService/claims", 参数: appNum, patType
async def get_spec(self, doc_id: str) -> dict
    # method="/openService/spec", 参数: docId
async def get_file(self, pub_num: str, pub_date: str) -> bytes 流
    # method="/openService/file", 参数: pubNum, pubDate, fileCategory=PDF, patentCategory
    # ⚠ 返回 base64 fileByte，必须流式解码（见 5.6）
```

- method 路径按 `/openService/law` 惯例推测（`/openService/search|getDoc|claims|spec|file`），**需 APP_KEY 实测确认**（见待确认 ①）
- 复用 `_top_sign` / `_encode_params` 不变；新增响应统一解析（code!=200 抛错，与现有 law 一致）
- **网关统一**：client 现用 `http://open.baiten.cn/router`，config.py 默认 `https://open.patexplorer.com/api/gateway`——两者不一致，需确认当前有效网关（见待确认 ②），最终以 config 为准注入

### 5.2 新路由 `api_routes/baiten.py`

沿用 `register_*_routes(logger, config)` 工厂模式，`api.py` 注册：

| 端点 | 用途 | 说明 |
|---|---|---|
| `POST /baiten/search` | 场景工具入口 | body `{query_string, page, page_size}`；内部调 BaitenClient.search → 映射 candidate 列表返回 |
| `GET /baiten/download` | PDF 代理 | 复用 uspto lazy-download 模式：入参 pubNum+pubDate（或预签名 token），流式拉取佰腾 PDF → 临时文件 → `FileResponse(application/pdf)`；供前端 PDF viewer 内嵌 |

场景工具调用走现有 `execute_backend_tool_request` 链路（MySQL tools 表条目，tool_url=`https://api.copiioai.com/baiten/search`，tool_params 模板含 queryString 示例），**无需改造 agent 工具执行框架**。

### 5.3 patent_detail.py 扩展（`api_routes/patent_detail.py`）

- `VALID_SOURCES` 加入 `"baiten"`
- 路由层 `patent_id` 语义：**patent_id = 公开号 pn**（如 CN112345678A，≤40 字符约束满足）
- `_fetch_spec_pdf(source="baiten")`:
  1. `get_doc(pn)` 补全 appNum + pubDate（一次调用）
  2. 优先说明书查询 `get_spec(appNum)` → 返回文本/文档 URL
  3. 或 `get_file(pubNum, pubDate)` → 流式解码 base64 写临时文件 → 返回 `/baiten/download?...` 代理 URL（前端内嵌不变）
- `_fetch_claims(source="baiten")`:
  1. `get_doc(pn)` 补 appNum
  2. `get_claims(appNum, "AUTH")` 失败/空 → `get_claims(appNum, "APP")` 回退
  3. `patentClaimses[]` 层级结构**扁平化**映射现有 payload：`{number, text, independent}`（claimsParentNum 为空/无引用 → independent=true），复用 `build_claims_payload` 契约（200 + success:false 优雅降级，注释已说明 Cloudflare 5xx 换页的坑）

### 5.4 CN 检索式组装（`sources/long_task/search_query_builder.py` CN 变体）

核心差异：中文无词形变化 → **不用词尾通配符，同义词用 OR**；检索式带字段前缀 `ti:/ab:/clm:`；**不能剥离 CJK**（现 `sanitize_uspto_query` 剥离中文字符，CN 版需新 sanitize：保留中文+英文+数字，限制长度）。

新增（与现有纯函数风格一致，可单测）：

```python
def assemble_baiten_query(groups: list[list[str]], field: str) -> str
    # 单字段版: ti:(A or B) and ti:(C or D)
    # 多词短语加双引号（中文短语含空格才需要，中文一般不加）

def _assemble_baiten_ladder(groups) -> list[str]
    # 阶梯（由紧到松）:
    #   L1 全概念 ti: 组（标题字段最精确）
    #   L2 全概念 ab: 组（摘要）
    #   L3 全概念 clm: 组（权利要求）
    #   L4+ 逐级丢最弱概念（复用现有丢概念逻辑），每级 ti→ab→clm 交替
    # 复用 MAX_KEYWORDS_PER_GROUP / MAX_ASSEMBLED_QUERY_CHARS 长度预算逻辑
```

LLM 概念提取 prompt **CN 变体**（新 `REWRITE_SYSTEM_PROMPT_CN`，不动现有英文版）：
- 概念抽取规则同现有（2-4 概念、按重要性排序、禁止合并要素）
- 关键词改为**中文为主**（3-8 个同义/近义，含全称/简称/上位/下位/俗名；允许少量英文术语，CN 专利文献常见中英混用）
- 载体词规则保留（中文载体词）
- ⚠ **通用性约束**（memory: reject-query-specific-synonym-hardcoding）：prompt 不得写入任何示例提问/具体技术词，仅描述抽取规则；阶梯组装纯代码化，不依赖 LLM 输出检索式
- 返回 JSON 结构不变（concepts/keywords/carriers），下游 `build_search_queries` 增加 `mode="baiten"` 分支走 CN 组装器

`format_ladder_guidance` 增加 CN 版提示文案（字段前缀语义说明：ti=标题 ab=摘要 clm=权利要求）。

### 5.5 场景工具注册（MySQL knowledge/tools 表）

新增工具条目（与管理台录入方式一致，不写死 SQL）：
- tool_title: 中国专利检索（佰腾）
- tool_url: `https://api.copiioai.com/baiten/search`
- scene 绑定 + knowledge_question 检索词（触发场景）
- tool_params 模板: `{"query_string": "<示例检索式格式>", "page": 1, "page_size": 20}`（示例值仅展示格式，不写具体测试词）

配套 `core.py:180` 场景工具 source 推断加 baiten URL 分支（`"baiten" in url_text → cnipa`）；`infer_patent_source` 文本关键词已覆盖 cnipa，无需新增硬编码关键词。

### 5.6 流式与内存约束（服务器可用内存 <1GB）

佰腾 PDF 返回 base64 fileByte（说明书 PDF 可达几十 MB，base64 膨胀 1.33x）。**禁止整包 json.loads + decode**：

- 下载：httpx `AsyncClient.stream` 逐块读取 → **ijson** 流式解析出 fileByte 字段 → 每块 base64 解码 → 追加写 tempfile（`tempfile.SpooledTemporaryFile(max_size=…)`）
- 峰值内存 <150MB（满足 memory: server-memory-constraint）
- 临时文件生命周期：响应流式回传后清理（或按 /uspto/download 现有缓存/清理策略对齐）

### 5.7 检索结果映射

`fieldValues{id,an,ad,pn,pd,ti,pa}` → candidate 结构：

```
{ patent_id: pn, source: "baiten", title: ti, pub_date: pd,
  app_num: an, apply_date: ad, ... }
```

- 与现有 zldsj 工具返回结构对齐（`extract_patent_ids` 的 patent_id 字段约定），LLM 格式化链路无需改动
- hlFieldValues 高亮 → 可映射 title 高亮（Phase 2）

## 六、分阶段实施计划

| 阶段 | 内容 | 依赖 |
|---|---|---|
| **P0 实测前置** | 用户提供 APP_KEY/APP_SECRET；curl 冒烟确认网关域名、method 路径、apiLevel 返回字段、source 枚举 | 用户 |
| **P1 核心** | BaitenClient 5 方法（含 _build_top_params 重构）→ /baiten/search + /baiten/download → patent_detail baiten source → CN 组装器 + prompt 变体 → 场景工具条目 | P0 |
| **P2 增强** | law(FLZT) 法律状态接线展示/过滤、family 同族去重、like 相似召回 | P1 |
| **P3 可选** | 与 zldsj 结果同屏合并（多源去重）、hlFieldValues 高亮 | P1 |

## 七、测试方案

- **单测**（pytest，纯函数）：
  - `assemble_baiten_query` / `_assemble_baiten_ladder`：字段前缀、引号、长度预算、丢概念逻辑
  - CN sanitize：保留 CJK、禁通配符规则
  - claims 扁平化映射：层级→number/text/independent
  - 流式 base64 解码：内存峰值断言（<150MB）
- **集成**（httpx mock 佰腾网关）：search→candidate→spec/claims 全链路
- **签名回归**：现有 `_top_sign` 已知向量测试保留
- **实测冒烟**（P0）：用户 key 到位后 3 个接口 curl 验证（search/claims/file）
- **前端回归**：`npm run build` + 结果页四 tab 手测（baiten 源 spec PDF 内嵌、claims 结构化展示）

## 八、待确认事项（阻塞 P0/P1）

| # | 事项 | 说明 | 建议动作 |
|---|---|---|---|
| ① | **APP_KEY/APP_SECRET** | 实测必需 | 用户从 open.baiten.cn 控制台获取 |
| ② | **网关域名** | client 用 `open.baiten.cn/router`，config 默认 `open.patexplorer.com/api/gateway`，不一致 | 用户确认/实测 |
| ③ | **method 路径** | `/openService/search` 等为推测（SDK 只给 Java 类名） | 实测或要 SDK JAR 反编译 |
| ④ | **apiLevel 档次** | 决定检索返回字段集，能否拿 an/pd 直接影响 spec/PDF | 控制台查已购档次 |
| ⑤ | **source 枚举** | 示例 15=中国，需确认 US/EP/PCT 完整枚举 | 文档/实测 |
| ⑥ | **计费/配额** | search/claims/spec/file 是否分项计费（PDF 逐篇下载量大） | 控制台查计费规则 |

## 九、风险与对策

| 风险 | 对策 |
|---|---|
| method 路径猜错 → 全部请求失败 | P0 先 curl 冒烟再写代码；单测覆盖签名，路径错误时快速定位 |
| 检索字段集缺 an/pd（apiLevel 低） | patent_id 改 an 或 getDoc 补一次（多 1 次调用，计费注意） |
| PDF 下载量大超配额 | spec/claims 优先说明书/权利要求接口（文本/轻量），PDF 仅「原始文档」tab 按需拉取 |
| CN 检索式质量（中文分词/同义词） | 同义词 OR 组 + 载体词机制 + 阶梯多级回退，与 US 链路同策略 |
| 与 zldsj 通道功能重叠 | 并存定位：zldsj 管检索结果流，佰腾补原始文档查看；P3 可选合并 |
