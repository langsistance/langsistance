# 设计：未指定国别时中美国专利双源并行检索（佰腾接入）

> 日期: 2026-08-24 | 状态: 待评审 | 前置: 需用户提供 BAITEN_APP_KEY/APP_SECRET 并实测接口路径（P0）

---

## 一、背景与目标

现有自然语言专利检索以 USPTO 为唯一数据源。用户要求：**当用户没有明确指定要中国还是美国专利时，同时检索两个国家**——美国走现有 USPTO 逻辑，中国走佰腾开放平台。

本设计是 `docs/technical/baiten-cn-search-proposal.md`（2026-08-24 技术方案）的**修订与扩展**：

| 变更点 | 原提案 | 本设计 |
|---|---|---|
| 检索入口 | 场景工具条目 + 新端点 `/baiten/search` | **内置 StructuredTool**（agent 进程内直接并行调两源，无新检索端点） |
| 检索范围 | 仅中国专利（明确指定时） | 未指定国别 → **US + CN 双源并行**；明确指定 → 对应单源 |
| cnipa/zldsj 通道 | 与佰腾并存（zldsj 管结果流） | **不使用**——CN 检索统一佰腾，原有 cnipa 逻辑确保不触发、代码不改 |

约束（沿用 CLAUDE.md 与既有 memory）：
- 检索增强/prompt 修改必须**通用**，禁止把测试提问的具体词汇写进代码或 prompt 示例（memory: reject-query-specific-synonym-hardcoding）
- 服务器可用内存 <1GB，PDF base64 等大响应必须**流式低内存**（峰值 <150MB，memory: server-memory-constraint）
- 前端零改动（结果页已按 `source` 拼接 `/patent/{source}/{id}/spec|claims`）

---

## 二、需求决策（已与用户确认）

1. 未指定国别（`auto`）→ **双源并行**：USPTO（现有逻辑）+ 佰腾（中国库）
2. 明确美国（`uspto`）→ 现有 USPTO 逻辑，行为与现状**逐字节一致**
3. 明确中国 → 佰腾单源检索（复用双源工具的执行函数，只走 CN 参数）
4. **cnipa/zldsj 通道不再使用**：原有代码（`patent_token.py`、`patent_analyzer.py`、zldsj 场景工具）**不修改**，但新链路保证其**不会被触发**（见 §4.4）
5. 双源执行形态：**内置工具**（`build_tool_set` 注册 python `StructuredTool`），不新增检索端点、不新增 MySQL 工具条目
6. 结果页原始文档查看（spec/claims/PDF）仍需佰腾端点——这是前端浏览器直接调用的，与 agent 工具无关（前端零改动的代价）
7. 本轮**只写设计，不实现**

---

## 三、触发逻辑（国别判定与路由矩阵）

`_detect_patent_source`（`api_routes/core.py:104`）现有三态判定**代码不改**（关键词表不动）。改的是**消费方**：其返回值不再直接注入 agent，而是经一层映射：

```
_detect_patent_source 输出          映射后语义                检索入口
─────────────────────           ──────────────            ─────────────
uspto（明确美国）                 uspto                    现有 USPTO 场景工具（现状不变，不注册内置检索工具）
cnipa（明确中国）                 cn → 佰腾                注册内置「中国专利检索」工具（佰腾单源）
auto（未指定）                    dual → 双源              注册内置「综合专利检索」工具（USPTO + 佰腾并行）
```

映射层落在两处消费方（见 §4.3、§4.4），`_detect_patent_source` 本身与 zldsj 相关代码零改动。

---

## 四、组件设计

### 4.1 BaitenClient 扩展（`sources/baiten_client.py`）

现有 `BaitenClient`（`baiten_client.py:108`）已完整实现 TOP 式 MD5 签名（`_top_sign` `baiten_client.py:338`）+ 路径路由（`POST {gateway}/{methodPath}`，method 参与签名不进 body），仅 `query_law_infos` 一个方法。新增 5 个方法：

```python
async def search(self, query_string: str, source: str = "15", page: int = 1,
                 page_size: int = 20, api_level: str = "ONE") -> dict
    # method="/openService/search"，参数: queryString, source, page, pageSize, apiLevel
    # 返回 fieldValues{id, an, ad, pn, pd, ti, pa} + hlFieldValues（高亮，Phase 2 用）

async def get_doc(self, doc_id: str) -> dict
    # method="/openService/getDoc"，参数: docId —— 完整著录项，补 an/pd 给 spec/PDF

async def get_claims(self, app_num: str, pat_type: str = "APP") -> dict
    # method="/openService/claims"，参数: appNum, patType —— patentClaimses[] 层级结构

async def get_spec(self, doc_id: str) -> dict
    # method="/openService/spec"，参数: docId（CN 用申请号）

async def get_file(self, pub_num: str, pub_date: str) -> bytes 流
    # method="/openService/file"，参数: pubNum, pubDate, fileCategory=PDF
    # ⚠ 返回 base64 fileByte（PDF 可达几十 MB），必须流式解码（见 §7）
```

- `_build_top_params`（`baiten_client.py:297`）重构为通用签名构造器（接受 method + 业务参数 dict），`query_law_infos` 行为不变（签名回归测试保留）
- method 路径 `/openService/search|getDoc|claims|spec|file` 为**推测**（SDK 只给 Java 类名），需 APP_KEY 实测（P0 阻塞项）
- 网关域名：client 现用 `http://open.baiten.cn/router`（`baiten_client.py:55`），`config.py` 默认 `open.patexplorer.com/api/gateway` 不一致——以 config 为准注入（P0 实测确认，见 §10 ②）
- 响应统一解析：code != 200 抛 `BaitenAPIError`（与现有 law 一致）

### 4.2 CN 检索式组装（`sources/long_task/search_query_builder.py`）

**关键决策：并行两次独立 LLM 调用，现有英文链路零改动。**

- 现有 `build_search_queries`（`search_query_builder.py:158`，英文概念 + 英文关键词 + 载体词 → USPTO 阶梯）**完全不动**——`uspto` 明确时行为逐字节不变
- 新增 `build_baiten_queries(query, provider)`：新 `REWRITE_SYSTEM_PROMPT_CN`
  - 概念抽取规则与现有版一致（2-4 概念、按重要性排序、禁止合并要素、载体词机制保留）
  - 关键词改为**中文为主**（3-8 个同义/近义，含全称/简称/上位/下位；允许少量英文术语——CN 专利文献常见中英混用）
  - **通用性约束**：prompt 只描述抽取规则，不得写入任何示例提问/具体技术词（memory: reject-query-specific-synonym-hardcoding）
  - 返回 JSON 结构不变（concepts/keywords/carriers）
- 纯函数组装器（可单测，与现有 `assemble_query` / `_assemble_ladder` 风格一致）：
  - `sanitize_baiten_query(q)`：**保留 CJK**（现有 `sanitize_uspto_query` 剥离中文字符，不可复用），保留中文+英文+数字，限制长度
  - `assemble_baiten_query(groups, field)` → `ti:(A or B) and ab:(A or B) and clm:(A or B)`；中文无词形变化 → **不用词尾通配符**，同义词用 OR；多词短语按需加引号
  - `_assemble_baiten_ladder(groups)`：由紧到松——L1 全概念 ti 组（标题最精确）→ L2 全概念 ab 组 → L3 全概念 clm 组 → 逐级丢最弱概念（复用现有丢概念逻辑与 `MAX_KEYWORDS_PER_GROUP` / `MAX_ASSEMBLED_QUERY_CHARS` 长度预算）
- `format_ladder_guidance`（`search_query_builder.py:213`）加双源渲染分支：auto 模式输出两套阶梯（US 阶梯 + CN 阶梯），CN 部分标注字段前缀语义（ti=标题 ab=摘要 clm=权利要求）；现有英文文案不动

### 4.3 内置「综合专利检索」工具（`sources/agents/react_tools.py`）

参照现有内置工具先例 `FETCH_PATENT_SPEC_TOOL_NAME`（`react_tools.py:306-318`，`StructuredTool.from_function` + `_fetch_patent_spec_stub`），**无 HTTP 端点、无 MySQL 工具条目**：

```python
# build_tool_set（react_tools.py:286）增加 patent_source 入参（从 long task 上下文取）
# auto → 注册「综合专利检索」；cn → 注册「中国专利检索」；uspto → 不注册

class _DualPatentSearchArgs(BaseModel):
    query_string_us: str | None = None   # USPTO 检索式（英文阶梯）
    query_string_cn: str | None = None   # 佰腾检索式（中文阶梯）
    page: int = 1
    page_size: int = 20

async def _dual_patent_search(agent, args, lang) -> dict:
    tasks = []
    if args.query_string_us:
        tasks.append(uspto_search(args.query_string_us))     # 现有 USPTO 调用模式
    if args.query_string_cn:
        tasks.append(baiten_search(args.query_string_cn))    # BaitenClient.search(source=15)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # 统一映射 candidate: {patent_id, source, title, pub_date, app_num, ...}
    #   US:  patent_id=纯数字,   source="uspto"
    #   CN:  patent_id=CN 公开号, source="baiten"
    # 任一源失败 → 该源空列表 + 状态说明，不阻塞另一源
    return {"raw_items": merged_candidates, "note": ...}
```

- 单源变体（cn 明确时）复用同一执行函数，只传 `query_string_cn`
- USPTO 检索复用现有 `outbound_http`（`sources/http_outbound.py`，带重试）+ `_build_uspto_envelope`（`react_tools.py:682`）模式；佰腾检索直接调 `BaitenClient.search()`
- 工具描述明确引导：「未指定国别时优先使用本工具，一次返回美国+中国专利结果；指定国别时也可用于单源补充检索」
- 候选池消费方（`extract_patent_ids` `scene_tools.py:352`、`_extract_patent_ids_from_items` `general_agent.py:162`）对 `patent_id` 字段天然兼容，无需改动；CN patent_id 格式（CN112345678A）由 `patent_id_utils.py`（已支持 CN 前缀解析）处理

### 4.4 cnipa/zldsj 通道确保不触发（不改原有代码）

原有 zldsj 代码（`patent_token.py`、`patent_analyzer.py`、zldsj 场景工具、`scene_tools.py:279` 的 cnipa 路由 prompt 文本）**全部不动**。通过切断触发条件保证其不被走到：

| 触发条件（现状） | 切断手段 |
|---|---|
| agent 上下文中出现「专利来源: cnipa」→ select_tool prompt 规则选中 zldsj 工具 | 消费方映射：`_detect_patent_source` 返回 cnipa 时，注入上下文的语义改写为**佰腾语义**（「专利来源: cn → 优先佰腾工具」），不再产生 cnipa 字样 → `scene_tools.py:279` 的 cnipa 规则文本仍在但**永不触发** |
| LLM 在无提示时也可能选中场景中的 zldsj 工具条目 | zldsj 工具条目从绑定场景的 tools 表中**下线**（MySQL 数据层操作，非代码改动）——需用户/管理员确认并在管理台处理 |

> 注：`_detect_patent_source` 的场景工具 URL 推断分支（`core.py:169-191`，`zldsj`/`cnipa` URL → cnipa）依赖工具条目仍在场景中；zldsj 条目下线后该分支自然失效。此分支代码同样不动。

### 4.5 结果页原始文档查看（`api_routes/patent_detail.py`）

- `VALID_SOURCES`（`patent_detail.py:30`）加 `"baiten"`；路由层 `patent_id` 语义 = 公开号 pn（如 CN112345678A，≤40 字符约束满足）
- `spec` 分支（source=baiten）：
  1. `get_doc(pn)` 补全 appNum + pubDate（一次调用）
  2. 优先 `get_spec(appNum)` → 说明书文本/文档
  3. 或 `get_file(pubNum, pubDate)` → 流式解码 base64 写临时文件 → `/baiten/download` 代理 URL（前端 PDF viewer 内嵌，复用 `/uspto/download` lazy-download 模式）
- `claims` 分支（source=baiten）：
  1. `get_doc(pn)` 补 appNum
  2. `get_claims(appNum, "AUTH")` 失败/空 → `get_claims(appNum, "APP")` 回退
  3. `patentClaimses[]` 层级结构**扁平化**映射现有 payload `{number, text, independent}`（claimsParentNum 为空/无引用 → independent=true）
- 新增 `GET /baiten/download` PDF 代理（`api_routes/baiten.py`，新建小模块，沿用 `register_*_routes(logger, config)` 工厂模式）
- 前端零改动（`frontend/nextjs/services/api.ts:335-344` 已按 source 拼接）

---

## 五、数据流（端到端）

```
用户提问（中/英，未指定国别）
  │
  ▼ PHASE0（general_agent.py:1627）
  ├─ build_search_queries(prompt)      → US 英文阶梯（现有，不改）
  ├─ build_baiten_queries(prompt)      → CN 中文阶梯（新增，asyncio.gather 并行）
  └─ format_ladder_guidance(dual)      → 两套阶梯注入 agent 系统提示
  │
  ▼ agent 循环
  调「综合专利检索」内置工具（auto 模式注册）
    ├─ USPTO applications/search（英文阶梯）──┐ asyncio.gather 并行
    ├─ BaitenClient.search(source=15)（中文阶梯）─┘ return_exceptions=True
  │
  ▼ 候选池 [US candidates + CN candidates]（source 字段标记）
  │     → extract_patent_ids / 相关性池（现有，天然兼容）
  ▼
结果页四 tab（前端零改动）
  ├─ details: 著录项直出（检索返回字段）
  ├─ spec:    GET /patent/baiten/{pn}/spec → getDoc 补 an → getSpec / PDF 代理
  ├─ claims:  GET /patent/baiten/{pn}/claims → getDoc 补 an → getClaims(AUTH→APP 回退)
  └─ doc:     PDF 原文（/baiten/download 流式代理）
```

---

## 六、错误处理与降级

| 场景 | 行为 |
|---|---|
| 佰腾 key 未配置/失效 | auto 模式 CN 源失败 → 降级为纯 US，不阻塞（与现状一致）；上下文/工具描述提示 CN 检索暂不可用 |
| 单源超时/异常 | `asyncio.gather(return_exceptions=True)`，失败源空结果 + 状态说明，另一源照常 |
| CN 中文概念提取 LLM 失败 | `build_baiten_queries` 返回空阶梯 → 跳过 CN 源（与现有 `build_search_queries` 永不抛错的降级约定一致） |
| 佰腾 method 路径错误（P0 未实测时） | `BaitenAPIError` 被聚合工具捕获 → 该源空结果，全链路不炸 |
| claims AUTH 空 | APP 回退 |
| spec 无文本 | 降级 PDF 代理（getFile） |

---

## 七、流式与内存约束（服务器 <1GB）

佰腾 PDF 返回 base64 fileByte（说明书 PDF 可达几十 MB，base64 膨胀 1.33x）。**禁止整包 json.loads + decode**：

- 下载：httpx `AsyncClient.stream` 逐块读取 → **ijson** 流式解析出 fileByte 字段 → 每块 base64 解码 → 追加写 `tempfile.SpooledTemporaryFile`
- 峰值内存 <150MB
- 临时文件生命周期：响应流式回传后清理（对齐 `/uspto/download` 现有缓存/清理策略）

---

## 八、测试方案

- **单测**（pytest，纯函数）：
  - `assemble_baiten_query` / `_assemble_baiten_ladder` / `sanitize_baiten_query`：字段前缀、CJK 保留、长度预算、丢概念逻辑、无通配符规则
  - `build_baiten_queries`：mock provider（概念/关键词/载体词结构解析、失败返回空）
  - 聚合工具执行函数：mock USPTO + BaitenClient——双源并行、单源失败降级、candidate 映射（source 标记、CN 公开号格式）
  - 佰腾 claims 扁平化映射：层级 → number/text/independent
  - 流式 base64 解码：内存峰值断言（<150MB）
- **回归**：现有 `_top_sign` 已知向量测试保留；现有英文链路测试全绿（`build_search_queries` 不改）
- **实测冒烟（P0）**：用户 key 到位后 curl 验证 search/claims/spec/file + method 路径
- **前端回归**：`npm run build` + 结果页四 tab 手测（baiten 源 spec PDF 内嵌、claims 结构化展示）

---

## 九、分阶段实施计划

| 阶段 | 内容 | 依赖 |
|---|---|---|
| **P0 实测前置** | 用户提供 BAITEN_APP_KEY/APP_SECRET；curl 冒烟确认网关域名、method 路径、apiLevel 返回字段、source 枚举、计费 | 用户 |
| **P1 核心** | BaitenClient 5 方法（含 `_build_top_params` 重构）→ CN 组装器 + `build_baiten_queries` → 内置工具注册（build_tool_set 扩展 + 消费方映射层）→ patent_detail baiten + `/baiten/download` → 单测 | P0 |
| **P2 增强（可选）** | law(FLZT) 法律状态展示/过滤、同族去重、hlFieldValues 高亮 | P1 |

---

## 十、待确认事项（阻塞 P0/P1）

| # | 事项 | 说明 | 建议动作 |
|---|---|---|---|
| ① | **APP_KEY/APP_SECRET** | 实测必需 | 用户从 open.baiten.cn 控制台获取 |
| ② | **网关域名** | client 用 `open.baiten.cn/router`，config 默认 `open.patexplorer.com/api/gateway`，不一致 | 用户确认/实测 |
| ③ | **method 路径** | `/openService/search` 等为推测（SDK 只给 Java 类名） | 实测或要 SDK JAR 反编译 |
| ④ | **apiLevel 档次** | 决定检索返回字段集，能否拿 an/pd 直接影响 spec/PDF | 控制台查已购档次 |
| ⑤ | **source 枚举** | 示例 15=中国，需确认完整枚举 | 文档/实测 |
| ⑥ | **计费/配额** | search/claims/spec/file 是否分项计费（PDF 逐篇下载量大） | 控制台查计费规则 |
| ⑦ | **zldsj 工具条目下线** | 需在 MySQL 管理台从绑定场景下线（数据操作，非代码） | 用户/管理员确认 |

---

## 十一、风险与对策

| 风险 | 对策 |
|---|---|
| method 路径猜错 → 全部请求失败 | P0 先 curl 冒烟再写代码；聚合工具捕获异常降级，不阻塞 US 源 |
| 检索字段集缺 an/pd（apiLevel 低） | patent_id 改 an 或 getDoc 补一次（多 1 次调用，计费注意） |
| PDF 下载量大超配额 | spec/claims 优先说明书/权利要求接口（文本/轻量），PDF 仅「原始文档」tab 按需拉取 |
| CN 检索式质量（中文分词/同义词） | 同义词 OR 组 + 载体词机制 + 阶梯多级回退，与 US 链路同策略 |
| 双源结果量翻倍 → agent 上下文膨胀 | 复用现有候选池截断/摘要机制（`_items_digest` / 相关性池）；必要时按源分别截断 |
| zldsj 工具仍被 LLM 选中 | 工具条目下线（数据层）+ 佰腾工具描述引导；若仍有风险，`select_tool` 候选按 URL 排除 zldsj（一行过滤，P2 备选） |
