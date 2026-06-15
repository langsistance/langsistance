# 中国专利检索 — 上线方案

> 分支: `feature/china-patent` | 日期: 2026-06-15 | 提交数: 13

---

## 一、功能概述

基于 DI 开放平台 (open.zldsj.com) 的中国专利检索能力，用户通过自然语言提问即可检索中国专利数据库，结果经 LLM 格式化后流式输出。

### 1.1 改动文件

| 文件 | 类型 | 说明 |
|---|---|---|
| `api_routes/patent.py` | 新增 | OAuth 回调接口 & Token 管理 API（4 个端点） |
| `sources/patent_token.py` | 新增 | Token 生命周期管理（Redis 持久化、自动刷新、线程安全锁） |
| `api_routes/models.py` | 新增 | Pydantic 请求/响应模型 |
| `scripts/analyze_usage.py` | 新增 | 用量分析脚本 |
| `sources/dynamic_tool_params.py` | 修改 | ZLDJS 域名识别、OAuth 参数注入、`context.records` 提取 |
| `sources/agents/general_agent.py` | 修改 | 小列表快速路径、dict fallback 格式化、`_prune_item_for_llm` 裁剪逻辑 |
| `api.py` | 修改 | 注册 patent 路由 |
| `api_routes/core.py` | 修改 | 日志补充 user_id |

### 1.2 数据流

```
用户提问 "人工智能相关专利"
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ LLM 调用 dynamic_backend_tool_function              │
│ → POST https://open.zldsj.com/api/patent/search/... │
│ → query 参数自动注入: client_id, access_token, scope│
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ DI 平台返回:                                         │
│ { errorCode, total: 134193,                         │
│   context: { records: [10条专利] } }                 │
│                                                      │
│ → _extract_raw_items 提取 context.records           │
│ → raw_items = [10条专利]  传给 general_agent        │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ Phase 2: _stream_raw_items                          │
│ → 逐条 _prune_item_for_llm 裁剪过大字段             │
│ → 分 batch 送 LLM 格式化为 Markdown                  │
│ → SSE 流式输出到前端                                 │
│ → 生成 artifacts (JSON 下载)                         │
└──────────────────────────────────────────────────────┘
```

---

## 二、新增 API 端点

| 方法 | 路径 | 说明 | 认证 |
|---|---|---|---|
| `POST` | `/patent/callback` | DI 平台 OAuth 回调，接收 access_token + refresh_token | 无（DI 平台调用） |
| `GET` | `/patent/token` | 查询当前 token 状态 | 建议加 Admin |
| `POST` | `/patent/refresh` | 手动触发 refresh_token 刷新 | 建议加 Admin |
| `POST` | `/patent/clear` | 清除存储的 token（调试） | 建议加 Admin |

---

## 三、依赖与配置

### 3.1 外部依赖

| 依赖 | 说明 | 状态 |
|---|---|---|
| DI 开放平台 (open.zldsj.com) | 专利数据源，需注册应用获取 OAuth 凭证 | **需上线前完成** |
| Redis | Token 持久化存储（复用现有 Redis） | 已有 |
| DI 平台回调可达性 | `https://api.copiioai.com/patent/callback` 需公网可达 | **需确认** |

### 3.2 config.ini 新增

```ini
[PATENT]
; DI 开放平台 OAuth 配置
; 在 DI 开放平台 (open.zldsj.com) 注册应用后获取
; 也可通过环境变量 PATENT_CLIENT_ID / PATENT_CLIENT_SECRET 覆盖
client_id =
client_secret =
; DI 平台 refresh_token 接口地址
refresh_url = https://open.zldsj.com/oauth2/token
```

### 3.3 .env 新增

```bash
# DI 开放平台 OAuth（优先级高于 config.ini）
PATENT_CLIENT_ID=your-client-id
PATENT_CLIENT_SECRET=your-client-secret
```

### 3.4 docker-compose.yml 变更

需要在 `environment` 段新增：

```yaml
- PATENT_CLIENT_ID=${PATENT_CLIENT_ID}
- PATENT_CLIENT_SECRET=${PATENT_CLIENT_SECRET}
```

### 3.5 Redis Key 设计

| Key | 说明 | TTL |
|---|---|---|
| `patent:access_token` | 当前 access_token | 7 天 |
| `patent:refresh_token` | 当前 refresh_token | 30 天 |
| `patent:token_expires_at` | access_token 过期时间戳 | 7 天 |
| `patent:refresh_expires_at` | refresh_token 过期时间戳 | 30 天 |

---

## 四、上线步骤

### Step 1 — DI 平台准备（提前 1 天）

- [ ] 在 [DI 开放平台](https://open.zldsj.com) 注册应用
- [ ] 获取 `client_id` 和 `client_secret`
- [ ] 配置回调地址: `https://api.copiioai.com/patent/callback`
- [ ] 完成 DI 平台授权，确认 callback 能到达我们的服务器
- [ ] 验证 token 已存入 Redis: `GET /patent/token` 返回有效 access_token
- [ ] 验证 token 自动刷新逻辑（可调短 TTL 模拟过期场景）

### Step 2 — 服务器配置

- [ ] 更新服务器 `.env`，添加 `PATENT_CLIENT_ID` / `PATENT_CLIENT_SECRET`
- [ ] 更新 `config.ini` 的 `[PATENT]` section（或直接用 env 覆盖）
- [ ] 更新 `docker-compose.yml`，添加 patent 相关 environment 变量
- [ ] 确认 `REDIS_HOST` / `REDIS_PORT` 在容器内可达

### Step 3 — 代码部署

```bash
# 1. 合并到 main
git checkout main
git merge feature/china-patent

# 2. 推送
git push origin main

# 3. 服务器拉取
ssh <server>
cd /path/to/langsistance
git pull origin main

# 4. 重建容器
docker compose --profile backend up -d --build
```

### Step 4 — 部署后验证

- [ ] `GET /patent/token` → 返回有效 access_token
- [ ] `POST /patent/refresh` → 刷新成功
- [ ] 前端提问 "检索人工智能相关中国专利" → 返回格式化专利列表
- [ ] 验证 SSE 流式输出正常
- [ ] 验证 artifacts（JSON 下载）正常
- [ ] 检查 `general_agent.log` 无异常（`[SMALL-LIST]` 路径不触发 — 专利结果数 > 3）
- [ ] 检查 `patent_token.log` Token 刷新日志正常
- [ ] 检查 `dynamic_tool_params.log` ZLDJS 请求/响应日志正常

---

## 五、回滚方案

如上线后出现严重问题：

```bash
# 代码回滚
git revert <merge-commit>
git push origin main

# 服务器重建
ssh <server>
cd /path/to/langsistance
git pull origin main
docker compose --profile backend up -d --build
```

或仅关闭专利功能：
```bash
# 从 api.py 注释掉 patent 路由注册
# api.include_router(patent.router, tags=["patent"])
# 重启容器即可
```

---

## 六、已知限制 & 注意事项

| 事项 | 说明 |
|---|---|
| Token 失效 | refresh_token 30 天过期后需手动在 DI 平台重授权；建议加监控告警 |
| 回调依赖 | DI 平台到期前自动刷新依赖 `/patent/callback` 公网可达；如回调失败，`ensure_valid_access_token` 的备用路径会直接读取 refresh API 返回值 |
| 日志截断 | `[ZLDJS RESPONSE]` 日志截断为前 10000 字符，仅影响日志排查，不影响业务数据 |
| 大结果集 | patent API 单页返回 10 条，如有分页需求需后续迭代 |
| LLM 模型 | 推荐使用 `openrouter` 系模型（已调优 prompt）；小型 Flash 模型可能在格式化长专利文本时截断 |
| 网络 | open.zldsj.com 需从服务器可达（如服务器在国内则无问题） |
| `_prune_item_for_llm` | 单个字段值 > 10000 字符会被裁剪，> 15000 字符的整个 item 会触发截断；专利摘要通常不会超过此阈值 |

---

## 七、后续优化建议

1. **分页支持** — 目前只取第 1 页 10 条，total 134193 条只能看到前 10 条
2. **Token 过期监控** — 加 Cron 定时检查 `/patent/token`，access_token < 1 天时告警
3. **字段映射表** — LLM prompt 中的字段翻译依赖模型知识，建议加显式映射表提升准确性
4. **结果缓存** — 相同检索条件可缓存 Redis 5 分钟，减少对 DI 平台的调用
5. **日志截断移除** — 稳定后把 `[:10000]` 去掉或改为按记录数限制
