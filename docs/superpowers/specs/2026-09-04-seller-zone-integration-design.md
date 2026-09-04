# CopiioAI 统一专利情报首页 · 卖家线结合设计（v2）

> 交互原型：`docs/prototypes/2026-09-04-seller-zone-prototype.html`（v3：统一首页单页模型，标题 → 模式切换 → 提问框 → 模式内容区 → 共享营销区）
>
> 上游文档：《copiioai-seller-product-design.html》v1.0（卖家安全台产品设计，2026-09-03）、《copiioai-oa-product-design.html》（案件指挥台主线）。本文档回答"卖家线如何与现产品结合"并记录 2026-09-04 三轮定稿。

## 1. 决策记录（2026-09-04 三轮）

| 轮次 | 决策 | 备注 |
|---|---|---|
| ① | 主站内结合，不建独立站点/独立前端 | 单仓库单账号，共享检索/解析/报告底层 |
| ② | 卖家 ⇄ 专业 = **双场景一等公民切换**，参照 DeepSeek instant-expert / WorkBuddy；不做小号"模式开关" | 场景页签显眼、实心主色高亮 |
| ③（本轮定稿） | **卖家首页与现有落地页合并为单一首页**："CopiioAI 专利情报，一问即得"；模式切换钮在标题正下方；切换只变**提问框下方内容区**，**URL 不变**（场景存状态 + cookie）；登录墙后同 URL 原位出结果；五环节/六模块等营销内容放**页面下方固定共享区**（上变下不变）；**Agent 策略 = 单 agent + 场景配置层，不上多 agent 框架** | 全部已入原型 v3 |
| ③-补 | 登录后首页即主工作区（快速查 + 历史入口）；聊天长对话/知识库等沿用现有页面但场景状态记忆 | — |

## 2. 现状事实清单（代码核实）

### 前端 frontend/nextjs（Cloudflare Pages）
- 现有 `(landing)` 主页 = 营销落地 + SEO 基建（JsonLd/sitemap/robots/LandingSeoContent/`examples/*` 示例子页）；`(auth)` 产品层 = chat / results / knowledge / community / messages / devtools / share
- 视觉体系（LandingHeader.tsx）：fixed 白底 header、logo、灰描边按钮 + **teal-600 实心主按钮**、rounded-lg、max-w-7xl、Tailwind 默认灰阶
- 登录引导已有：`SceneAutoOnboard` + `PatentOnboardingWizard`（匿名跳过）

### 后端
- 场景机制为**数据驱动**：`mysql/init/init.sql` 已种 `scenes`（scene_id=1 美国专利检索）；用户订阅存 `user_scenes`（local_user 自动订阅默认场景）；场景知识经 `sources/long_task/scene_tools.py` **以工具形式挂进 agent react_loop**——切场景 = 换知识挂载，不改推理循环
- 引擎族在 `sources/long_task/`（query_builder / recall_sources / 双库 / rerank / relevance_gate / candidate_metadata / report_generator / prosecution 系 / `query_mode.py` 模式分发）
- 明细与解析：`patent_detail.py`、`patent_number_parser.py`、`patent_source_detect.py`；长任务异步：`long_task.py` 全套（submit/status/report/user_queue）
- 对话：`core.py /query_stream`（SSE agent 循环）；统计埋点 `analytics.py track_event`（已含 scene_id 参数）

### 明确缺口（文档称"复用"但代码不存在）
1. 图搜（产品图→外观专利）——无图像引擎
2. 盯一盯（订阅/邮件推送）——无订阅表无推送任务
3. 专利卡所需**法律状态/到期数据源**（US 年费状态；CN 佰腾字段核对）——M1.5 spike
4. 报告**署名水印分享链接**——share 链路在，水印规格未实现

## 3. 统一首页模型（单页多场景）

### 3.1 页面结构（一个 URL，自上而下）
```
┌ 顶部（现有 LandingHeader 壳复用）┐
├───────────────────────────────────┤
│ H1：CopiioAI 专利情报，一问即得      │  ← 统一标题（前缀品牌名）
│ [🛒 卖家安全台 │ 🔬 专业工作台]      │  ← 模式切换：标题正下方、提问框正上方，
│ 大提问框（占位随模式变化）           │    显眼大按钮（实心 teal 高亮激活态）
│ ── 模式内容区（随切换变化，URL 不变）── │
│   卖家：模块快捷键卡 ×3 + 示例 chips   │
│   专业：检索类型卡 ×3 + 示例 chips     │
│   （查询后：结果卡片流 / 专利卡原位展开） │
├───────────────────────────────────┤
│ 共享营销区（两模式一致，SEO/转化留存）   │
│   卖家的每一步…五环节 · 六大模块 · 免责  │
└ 页脚 ┘
```

### 3.2 关键交互规格
1. **模式切换**：按钮在 H1 下（DeepSeek/WorkBuddy 式）；切换仅重渲染模式内容区与提问框占位；**URL 不变**；场景写入 cookie/本地偏好；深链 `?scene=` 仅在落地时生效一次，不随切换改写 URL
2. **登录承接**：未登录输入 → 登录墙 → 成功后**同 URL 原位执行**，结果出现在提问框下方；登录后首页即主工作区
3. **查询自识别**：形如专利号（容忍裸号/回执号）→ 专利卡；产品名/描述/ASIN → 查一查结果流；结果区提供两者互切
4. **专利卡四件套**：① 保护什么（图+一句人话）② 状态与到期 ③ 撞不撞（产品比对，M2+）④ 下一步建议带（可上架/需规避/授权询价/已过期）
5. **不确定性可视化**：结果流顶部固定检索范围声明（"已检索 644 万件 US + 佰腾库 → 候选 N 件"）；专利卡/营销区固定免责
6. **追问**：模式内容区内嵌追问 → 落现有对话（带当前场景包）；结果可"换专业视角看这件"（同 URL 切模式）
7. **文案红线**：卖家口径不出现专利法术语；营销区仅"保护什么/撞不撞/到期了吗/能不能卖"

### 3.3 视觉与内容原则（2026-09-04 反馈）
- 与 copiioai.com 完全一致：复用 landing 组件与 Tailwind 体系（teal/gray/rounded/max-w-7xl），不引入第二套视觉
- 一屏一主题：首屏只有标题+切换+提问框+模式内容；营销内容不挤进首屏
- 用户认可的保留元素：提问框位置（首屏居中）、模块卡片说明、"卖家的每一步，都有人替你盯着专利"

## 4. Agent 策略与技术架构（2026-09-04 定稿：**单 agent，不上多 agent 框架**）

### 4.1 为什么场景差异不需要新 agent
卖家 vs 专业的差异 = 输入口径 / 输出口径 / 知识范围 / 表述约束，四项全部可参数化：
| 差异 | 卖家 | 专业 | 承担层 |
|---|---|---|---|
| 输入口径 | 产品名/图/ASIN/裸号 | 权利要求语言/公开号 | 输入适配（parser 现成 + 前端） |
| 输出口径 | 人话结论卡片 | 对比文件/术语 | 输出契约 typed schema + 提示词包 |
| 知识范围 | 侵权 FAQ/被诉流程/术语翻译 | 检索/OA/同族深度 | 场景包（scenes 知识挂载，现成） |
| 表述约束 | 禁术语 | 术语自由 | system prompt 场景包 |

代码证据：`scene_tools.py` 已实现"场景知识以工具形式挂进 react_loop"（换场景=换挂载）；`long_task/query_mode.py` 已是同引擎多档位分发。AgentRouter 只在**能力差异**时换 agent（coder/browser 需不同执行环境），卖家/专业共用同一检索工具链。

### 4.2 不上多 agent 框架的代价账
多 agent 编排 = 额外 LLM 跳数（延迟+成本）、跨 agent 状态协议、调试面翻倍；服务器内存 <1GB、卖家为高频低客单流量，均不可承受。现有等价物：工具挂载 + long_task 异步 + Celery（监控推送本就是独立任务）已覆盖并发/异步/长跑。

**多 agent 触发器（出现才考虑，当前均不成立）**：两场景需并发长跑且互相决策依赖；或需按场景独立模型+预算+失败域且单进程装不下——届时加第二个 worker/池，非 LangGraph 式框架。

### 4.3 落地架构（文件级）
```
前端：统一首页（单路由）
  场景切换组件 SceneSwitcher + 模式内容区（两套空态渲染 + 共享结果/专利卡组件）
  ↓ 带 scene_id + scene 偏好 cookie
api_routes/seller.py（薄适配层，新）
  ① /seller/patent_card  专利号 → 解析 → patent_detail → 状态源(spike) → LLM 人话(缓存) → 卡片 schema
  ② /seller/search       产品描述 → 现有检索引擎族直连（长任务异步 submit/status） → 卡片化结果
  ③ /seller/ask          追问 → 复用 /query_stream + scene 参数
GeneralAgent：请求级 scene_id 参数 → 装配场景 prompt 包 + scene_tools 知识挂载 + 频控额度；agent 池照旧
scenes/知识种子：INSERT scenes('卖家查专利') + 卖家场景知识行（幂等）→ user_scenes 订阅后自动可用
```
- **卖家知识场景包内容**（种子数据）：被投诉怎么办 / 供应商话术验证 / 术语人话翻译 / 亚马逊 IP 投诉流程
- **新增代码面 M1**：后端 `api_routes/seller.py` + seeds（后端约 1 文件 + 种子 SQL）；前端统一首页重构（合并原 landing hero 与卖家工作台为单页：模式内容区两套视图 + 结果/专利卡组件 + SceneSwitcher + utm cookie）；**不动**：auth、session、knowledge、long_task 引擎、react_loop、agent 池主链路、现有 (auth) 页面
- **缓存与成本**：专利卡人话按（专利号 × 语言）缓存；报告/深分析仅按需；检索结果走 long_task 现有缓存
- **明确不做**：图搜（无引擎不做假接口）、投诉应对包/监控订阅（主线共用底层先建后包装）、新增第二个 agent

## 5. 能力复用映射与分期

| 卖家模块 | 代码现状 | 改造点 | 分期 |
|---|---|---|---|
| 查一查·文字版 | 引擎现成 | seller/search 卡片化 | M1 |
| 专利卡 | 明细现成；状态/到期缺 | 状态源 spike + 人话缓存 | M1（状态 M1.5） |
| 风险报告 | long_task report 现成 | 模板参数化 + 水印分享 + 收费开关 | M2 |
| 供应商验真 | — | 专利卡换皮参数 | M2 |
| 图搜 | 无 | 调研先行（内存约束），定位粗筛 | M3 |
| 盯一盯 | 无 | 与主线共用底层一次建设 | M3 |
| 投诉应对包 | 依赖审查历史解读 | 案件指挥台 MVP 之后复用 | M3+ |

### M1（2-4 周）验证口径
统一首页上线（两场景切换 + 登录墙 + 原位出卡）；冷启动周定基线：注册转化率 / 7 日回访 / 专利卡使用次数，达标才放量（utm 归因同期接入）。

## 6. 风险与边界
1. 漏检责任 → 免责 + 检索范围声明为硬性 UI 要求（三处：营销区/结果流/专利卡）
2. 图搜一期不做、不做首屏；结论交给报告环节锚点兜底
3. 账号质量 → 重资源功能轻门槛（邮箱验证 + 频控）
4. 协同次序 → 监控/报告/审查历史底层先建组件后分场景包装，避免双实现
5. 合规 → CN 数据与生成走境内链路（沿用主线原则）
6. 专利卡"状态与到期"数据源未验证 → M1.5 spike 前置（US 法律状态接口；CN 佰腾字段核对）

## 7. 交付清单（文件级）
- **前端 nextjs**：统一首页（重构 `(landing)/page.tsx` 与卖家工作台为单页模型）、SceneSwitcher 与模式内容区组件、结果流/专利卡组件、utm cookie 工具、`examples/seller-*` SEO 子页（长尾）
- **后端薄改**：`api_routes/seller.py`（3 端点）、scenes/卖家知识种子（幂等 SQL）、GeneralAgent scene_id 参数、analytics utm 字段
- **后端新建（M2/M3）**：报告水印分享参数、监控订阅底层、法律状态源 client、图搜调研
- **零改动**：auth、session、knowledge、long_task 引擎、react_loop、现有 (auth) 产品页
