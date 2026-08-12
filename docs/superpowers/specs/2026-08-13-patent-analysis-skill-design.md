# 专利审查分析 Claude Code Skill — 产品设计文档

> 日期: 2026-08-13 | 分支: `feature/china-patent-analysis` | 状态: 设计已确认，待实现

---

## 1. 背景与目标

将已有的专利分析能力（USPTO 审查策略分析 + 跨国同族 US/CN/JP/EP 对比报告）打包为 Claude Code skill，通过 MCP server 连接自托管后端 `api.copiioai.com`，实现商业变现。

**核心价值主张**：5 分钟生成一份专利律师级的审查策略报告（含 Claim 对照表、审查员攻防记录、无效风险分析），替代初级律师 4-8 小时的手工 file wrapper 阅读。

**目标用户**：
- 主：有技术背景的专利代理人/律师（会用 AI 工具的早期采用者）
- 次：企业 IP 部门的工程师、做 FTO/无效分析的团队
- 特色：中国涉外代理所（中英双语报告 + 跨国同族对比是刚需）

---

## 2. 架构总览

```
┌──────────────────────┐
│  用户 Claude Code     │
│  "分析 US16/123456     │
│   的审查历史"          │
└────────┬─────────────┘
         │ MCP protocol (stdio)
         ▼
┌──────────────────────┐
│  npm: @copiioai/patent-mcp   │  ← 用户本地 npx 启动，薄代理
│  (TypeScript, 无状态)         │
│                              │
│  tools:                      │
│  - patent_prosecution_analyze│  提交→立即返回 job_id
│  - patent_family_analyze     │  提交→立即返回 job_id
│  - patent_job_status         │  查进度
│  - patent_job_result         │  取完整报告
│  - patent_usage              │  查剩余配额
└────────┬─────────────┘
         │ HTTPS + X-API-Key
         ▼
┌──────────────────────┐
│  api.copiioai.com     │  ← 现有机房，新增路由
│  /api/v1/patent/*     │
│  - POST /jobs/submit  │  创建 Celery 任务
│  - GET  /jobs/{id}    │  读 Redis 进度
│  - GET  /jobs/{id}/result │ 取报告
│  - GET  /usage        │  配额查询
│  - 中间件: API key 验证 + 限额   │
└────────┬─────────────┘
         │
    ┌────┴────┐
    │  Redis  │  ← job status / 部分结果 / 配额计数
    │  Celery │  ← 现有 prosecution_analyzer pipeline
    │  MySQL  │  ← api_keys 表
    └─────────┘
```

**关键决策**：
- MCP server 是 npm 包（`npx` 启动），跑在用户本地，无需额外运维
- 后端复用现有机房，只新增 `/api/v1/patent/*` 路由；分析 pipeline 不变
- 异步作业模式：submit → poll(60s) → fetch（MCP tool call 不能阻塞 5 分钟）

---

## 3. MCP Tools 定义

### 3.1 提交分析（2 个）

```
patent_prosecution_analyze
├── patent_id: string        # 美国专利申请号，如 "US16/123456"
├── query: string            # 用户想搞清楚的问题
├── lang: "zh" | "en"        # 报告语言
└── 返回: { job_id, status: "queued", estimated_seconds: 300 }

patent_family_analyze
├── patent_id: string        # 任意国家专利号，自动解析同族
├── query: string
├── lang: "zh" | "en"
└── 返回: { job_id, status: "queued", estimated_seconds: 480 }
```

### 3.2 查进度（1 个）

```
patent_job_status
├── job_id: string
└── 返回: {
      status: "running" | "done" | "failed",
      progress_pct: 78,
      current_step: "正在撰写核心审查洞察...",
      estimated_remaining_seconds: 90
    }
```

### 3.3 取结果 + 查配额（2 个）

```
patent_job_result
├── job_id: string
└── 返回: { report: "## 审查策略简报\n\n..." }

patent_usage
└── 返回: { tier: "free", credits_remaining: 2, total_used: 1 }
```

### 3.4 Claude Code 交互流程

```
用户: "分析 US16/123456，为什么这个专利能授权？"
  → [1] patent_prosecution_analyze(...) → job_id
  → 告诉用户: "分析已提交，预计 5 分钟"
  → [2] patent_job_status(job_id) ← 每 60 秒轮询，向用户报告进度
  → [3] 完成后 patent_job_result(job_id) → 展示完整报告
```

SKILL.md 指令指导 Claude Code 自动执行此流程。

---

## 4. 后端 API 设计

### 4.1 新增路由（`api_routes/patent_v1.py`）

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/patent/jobs/submit` | 创建分析作业。body: `{patent_id, query, lang, analysis_type}`，其中 `analysis_type ∈ {prosecution, family}`。验证 API key → 检查配额 → 提交 Celery 任务 → 返回 job_id |
| GET | `/api/v1/patent/jobs/{job_id}` | 读 Redis：status / progress_pct / current_step |
| GET | `/api/v1/patent/jobs/{job_id}/result` | 作业完成后返回完整 Markdown 报告 |
| GET | `/api/v1/patent/usage` | 当前 key 的 tier / 剩余次数 |

**MCP tool → 路由映射**：
- `patent_prosecution_analyze` → POST /jobs/submit（analysis_type=prosecution）
- `patent_family_analyze` → POST /jobs/submit（analysis_type=family）
- `patent_job_status` → GET /jobs/{job_id}
- `patent_job_result` → GET /jobs/{job_id}/result
- `patent_usage` → GET /usage

### 4.2 认证中间件

- 请求头 `X-API-Key` → SHA-256 哈希 → 查 `api_keys` 表
- 无效 key → 401；有效 key → 注入 user 上下文
- 配额检查：submit 时扣减 credits（失败则 402 + 购买引导）
- 与现有 Firebase 认证（`passport.py`）并行，互不影响

### 4.3 作业生命周期（Redis）

```
Key: patent:job:{job_id}
Value: {
  "status": "queued" | "running" | "done" | "failed",
  "progress_pct": 0-100,
  "current_step": "正在分析 Office Action...",
  "report": "...",          # done 后写入
  "error": "...",           # failed 时写入
  "created_at": 1755600000,
  "expires": 86400          # TTL 24h
}
```

Celery 任务运行现有 `prosecution_analyzer` pipeline，通过 `summary_updater` 的 `push()` 回调把进度写进 Redis（改造 `push()` 同时写 SSE 和 Redis，或新增适配器）。

### 4.4 API key 表（MySQL）

```sql
CREATE TABLE api_keys (
    id INT AUTO_INCREMENT PRIMARY KEY,
    key_hash VARCHAR(64) UNIQUE NOT NULL,
    user_email VARCHAR(255),
    tier VARCHAR(20) DEFAULT 'free',        -- free / paid
    credits_remaining INT DEFAULT 1,          -- 免费 1 次
    created_at TIMESTAMP DEFAULT NOW()
);
```

管理方式（手动阶段）：付钱 → SQL INSERT → 邮件发 key。1 天开发量。

---

## 5. 计费与变现

### 5.1 定价（初版）

| 档位 | 价格 | 内容 |
|---|---|---|
| 免费试用 | $0 | 1 次深度分析 |
| 按次 | $29/次 | 单次深度分析 |
| 优惠包 | $99/6 次 | 6 次深度分析 |
| 月付 | $149/月 | 20 次深度分析 |

### 5.2 手动收款流程（第一阶段）

```
[1] 用户免费额度用完，继续提交分析
    → MCP 返回: "credits 已用完。购买: https://copiioai.com/keys
      或邮件 contact@copiioai.com"
[2] 用户发邮件: "想买 6 次分析包"
[3] 你回复付款链接（PayPal.me / Wise / Stripe Payment Link）
[4] 用户付款（1 分钟）
[5] 你收到付款通知，打开自己的管理后台页面
    → 填 user_email + credits 数量 → 生成
[6] 后端 SQL UPDATE 给该 email 的 key 增加 credits
    （或第一次购买时 INSERT 新 key）
[7] 你回复邮件: "已到账，你的 key 还是原来那个，直接继续用"
```

**关键体验**：老用户的 key 不变——充值是给现有 key 加 credits，不是发新 key。用户付完钱回来，Claude Code 里什么都不用改，直接继续分析。

**管理后台**：v1 做一个极简内部页面（或复用现有 admin 能力）：输入 email + credits → 按钮 → SQL 更新。不需要给用户看的 portal。

**不接 Stripe 的原因**：先验证 10 个付费用户。手动收款强迫与用户对话，获取反馈；Stripe 需要美国主体/Atlas（成本 $500+ 且合规复杂）。跑通后再自动化。Stripe 的过渡产品是 Stripe Payment Link——不建 billing 系统，只是一个收款链接，保持"手动"性质。

### 5.3 免费额度耗尽提示

```
patent_usage → credits_remaining: 0
submit → 402: "免费次数已用完，购买分析次数: https://copiioai.com/keys
或邮件 contact@copiioai.com"
```

---

## 6. 分发与推广

### 6.1 分发渠道

- **npm**: `@copiioai/patent-mcp`（MCP server）
- **GitHub**: 开源 repo，含 SKILL.md、README、演示 GIF、示例报告
- **Claude Code 安装**：
  ```json
  // .mcp.json
  {
    "mcpServers": {
      "patent-mcp": {
        "command": "npx",
        "args": ["-y", "@copiioai/patent-mcp"],
        "env": { "PATENT_API_KEY": "pk_xxx" }
      }
    }
  }
  ```

### 6.2 推广矩阵（按优先级）

| # | 渠道 | 动作 | 成本 |
|---|---|---|---|
| 1 | MCP 目录站（mcp.so / PulseMCP / Glama） | 提交 listing | 1 小时 |
| 2 | 示例报告内容营销 | 挑 3 个明星专利（Apple Vision Pro / Tesla / 中国出海公司），生成报告做截图 | 0 |
| 3 | r/ClaudeAI + r/patents | "Drop a patent number, get a strategy report" 帖 | 0 |
| 4 | GitHub + awesome 列表 | 开源 repo，PR 进 awesome-claude-code / awesome-mcp | 0 |
| 5 | Show HN / X (#claudecode #mcp) | 硬核工具帖 + 演示视频 | 0 |
| 6 | 中文社区（掘金 / V2EX / 知乎） | 涉外专利人群，双语报告是钩子 | 0 |
| 7 | LinkedIn 定向 BD | patent agent + AI 关键词，附示例报告 | 0 |

**核心原则**：推广的是"5 分钟生成专利审查策略报告"这个结果，不是 skill 本身。

---

## 7. 交付物清单

| # | 交付物 | 位置 | 预估工作量 |
|---|---|---|---|
| 1 | npm 包 `@copiioai/patent-mcp` | `patent-mcp/`（TypeScript，~200 行） | 2-3 天 |
| 2 | SKILL.md | 同 GitHub repo | 0.5 天 |
| 3 | 后端路由 `api_routes/patent_v1.py` | langsistance | 2 天 |
| 4 | 认证中间件 + api_keys 表 | langsistance | 1 天 |
| 5 | Celery 任务 + Redis 进度适配器 | langsistance | 2 天 |
| 6 | 测试（后端 pytest + MCP 冒烟） | langsistance | 1-2 天 |
| 7 | 部署（docker-compose / api.py 注册） | langsistance | 0.5 天 |

总计约 9-11 天开发。

---

## 8. 测试策略

- **后端路由测试**：API key 验证（有效/无效/配额耗尽）、submit → status → result 全流程（mock Celery）
- **pipeline 集成测试**：现有 `test_patent_analyzer.py` / `test_patent_family.py` 保持绿
- **MCP server 冒烟测试**：stdio 协议握手、5 个 tool 的 schema 验证、错误路径（无 key / 402）
- **端到端手动验证**：真实专利号跑通全流程，报告质量人工 review

---

## 9. 安全

- API key 存哈希（SHA-256），不存明文
- 每个 key 限流（如 10 req/min）
- `patent_id` 输入格式校验（防注入）
- 报告内容本身已有法律措辞红线（现有 prompt 已实现）
- 不新增任何面向公网的无认证端点

---

## 10. 明确不做（v1 范围外）

- ❌ Stripe 自动计费（验证 10 个付费用户后再做）
- ❌ GPT Store 分发（以后可复用同一后端，OAuth 接入，1-2 天）
- ❌ 快速/深度两档分析模式（Claude Code agent 自动轮询，等待体验可接受；以后可加）
- ❌ 网页前端（需要时直接调 API 即可）
- ❌ 批量分析、团队 workspace、私有部署
- ❌ OpenAI 官方分成（需要美国税务身份，收入预期低）

---

## 11. 里程碑

| 里程碑 | 验收标准 |
|---|---|
| M1: 后端 API 可用 | curl 提交真实专利号，24h 内拿到完整报告 |
| M2: MCP + SKILL.md 可用 | Claude Code 里一句话触发全流程 |
| M3: 内部测试 | 3 份示例报告质量通过人工 review |
| M4: 公开上线 | npm 发布 + GitHub 开源 + MCP 目录提交 |
| M5: 首个付费用户 | 有人为第 4 次分析付费 |
