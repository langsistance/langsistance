# CopiioAI 中国部署 + 微信小程序 迁移总方案与执行手册

- 版本：v1.0（2026-09-08 锁定）｜ 状态：待评审
- 关联文档：
  - `deploy/china/model-config.deepseek.md`（中国实例模型配置）
  - `deploy/china/RUNBOOK_DB_MIGRATION.md`（数据迁移+双端切流，分钟级执行手册）
  - `deploy/china/RUNBOOK_GO_LIVE.md`（上线编排/灰度/回滚矩阵）
- 原则：**每一步有执行、有验证、有回滚**；代码改动在方案评审通过后开始

---

## 1. 背景与目标

1. 推出**微信小程序**（Taro 原生），服务**国内微信用户**，并与现有用户体系（数字 user_id 单键）兼容。
2. **数据迁回国内**（MySQL 单主放国内腾讯云），美国现网（web/浏览器插件）在过渡期内**直连国内库**，最终收敛到国内为主。
3. 中美服务器**全部只使用中国模型**（deepseek-v4-flash / vision-exp；Embedding 已是 siliconflow）。
4. 网页端/插件**现有功能与 Firebase 登录先不迁移、不回归**。

## 2. 已锁定决策（2026-09-08）

| # | 决策 | 备注 |
|---|---|---|
| D1 | 小程序国内微信用户 + Taro(React) 原生 | 不套 web-view |
| D2 | 身份兼容：微信挂 `users.oauth_provider='wechat' / oauth_provider_id=openid|unionid`，与邮箱/Google 同表同 `user_id` | 需 users 表 3 条 ALTER |
| D3 | 登录：`POST /auth/wechat`（code→code2session→wx token 存 Redis 7d）；passport 加 `wx_` 分支 | code2session 已实测美国可达 |
| D4 | 数据：MySQL **单主迁国内**；美国直连（**不做代码连接池**，过渡期接受降级） | 备选：美国机 ProxySQL/只读从库（Phase 4 闸门） |
| D5 | 模型：中国实例全 DeepSeek（chat/interpret/stream/vision）；**美国生产暂不切** | 模板已备 |
| D6 | 中国实例 = 完整业务后端镜像（backend+celery+Redis），小程序流量直连它 | 非薄网关 |
| D7 | 国内出站：容器级 `HTTPS_PROXY` + `NO_PROXY`（无代码）；美国机跑 tinyproxy | 全 DeepSeek 后无 LLM 出境 |
| D8 | 访客埋点：自托管 page_view beacon 已上线生产；CF Web Analytics 待 token | 独立线程 |
| D9 | 增长统计：`growth_daily_stats.py` 按北京时间出日表，小程序用户自动并入（auth:wechat 需加口径） | 已含访客列 |

## 3. 目标架构

```
【国内 · 腾讯云】(新建)                        【美国 · 5.78.157.14】(现网不动)
mini.<备案域名> (小程序 request 合法域名)         api.copiioai.com (web/插件照旧)
   ↓ 仅小程序流量                                  ↓
中国实例 = backend + celery + Redis(本地)         美国 backend + celery + Redis(本地)
   │ 读/写 ~5ms                                    │ 跨境直连(无池,过渡期)
   ▼                                               ▼
国内 MySQL【主库·唯一权威】◄────── 跨境读写(P2 同窗切流后, P4 决策性能层)────────┘
users 单库: 微信 + Firebase + Google 全部同表同 user_id
```
- 中国实例对话模型 deepseek（国内 ~30ms）；美国实例暂 openrouter（保持现状）
- 中国实例出站：`NO_PROXY` 直连（微信/DeepSeek/SiliconFlow/佰腾/CNIPA/SearXNG…），其余走美国机 tinyproxy（Google Patents；USPTO 待实测定）

---

## 4. 阶段总览（执行顺序 + 并行关系）

| 阶段 | 内容 | 前置 | 负责人 | 工期 |
|---|---|---|---|---|
| **P0 前置准备** | 腾讯云账号/主体/备案/小程序 AppID/服务器开通 | 无 | **你** | 1-4 周（备案，全案最长路径，立即启动） |
| **P1 中国实例搭建** | 同构部署 + 模型配置 + 出站代理 | P0 服务器 | 我+你 | 2-3 天 |
| **P2 数据库迁移** | 全库 dump→国内主库→校验→**双端同窗切流**（详见 RUNBOOK_DB_MIGRATION） | P1 | 我 | <1h（含窗口，库仅 12MB） |
| **P3 微信登录服务端** | users ALTER + `/auth/wechat` + passport 分支（W3 代码） | P1/P2 | 我 | 1-2 天（代码可先写） |
| **P4 美国侧过渡** | 切流后性能决策：3 天探针 → 闸门（直连接受/ProxySQL/只读从库）；web 回归 | P2 | 我 | 观测期 |
| **P5 小程序端** | Taro M1 登录→M2 对话+专利面板→M3 查重/分享→提审 | P0/P3 | 我+你 | 3-6 周 |
| **P6 收尾** | 监控/日志归档/旧资源下线 | P5 | 我 | 0.5 天 |

并行线：P0 备案 与 P3 代码 与 P5 小程序 UI 可并行推进（代码不依赖备案）。

---

## 5. 分步执行手册

### P0 · 前置准备（你，立即启动，可与一切并行）

| 步骤 | 动作 | 验证 |
|---|---|---|
| P0.1 | 确认**公司主体**（小程序+域名备案同一主体） | 主体材料齐 |
| P0.2 | 开通**腾讯云账号**；建议地域：广州或上海 | 控制台可登录 |
| P0.3 | **域名备案** `mini.copiioai.com`（或新购域名）；备案需大陆服务器接入（可先用 P0.5 的机器作为接入） | 工信部备案号下发 |
| P0.4 | 注册**微信小程序**（企业认证），拿到 AppID/AppSecret | 后台可见 |
| P0.5 | 开通服务器：规格建议 4C8G + 50-100G SSD（与现网同构，DB 可先同机）；**安全组**：80/443 对公网、3306 仅对美国机 IP `5.78.157.14` 与国内机内网开放 | ssh 可达 |
| P0.6 | 给我：服务器公网 IP、AppID/Secret（或我先用占位符）、备案域名 | 记录进 .env |

> ⚠️ 全部后续阶段的**硬前置**是 P0.3（备案）与 P0.4（AppID）；P0.5 机器一开即可先跑 P1/P2 不等待备案（备案只需域名解析时生效）。

### P1 · 中国实例搭建（P0.5 完成后）

| 步骤 | 动作 | 验证/回滚 |
|---|---|---|
| P1.1 | 服务器装 Docker；同步代码仓库（与现网同 commit） | `docker ps` |
| P1.2 | 写 `.env`：从生产 .env 复制 + `DEEPSEEK_API_KEY` + `WECHAT_APPID/SECRET` + 出站变量（见 P1.4） | 无缺 key 报错 |
| P1.3 | 写 `config.ini`：**套用 `deploy/china/model-config.deepseek.md` 模板**（[MAIN]+[MODEL]+[LONG_TASK]） | 冒烟①文字提问走 deepseek |
| P1.4 | **出站代理**：美国机起 tinyproxy 容器（仅对中国 IP 放行）；中国 `.env` 设 `HTTPS_PROXY` + `NO_PROXY=api.weixin.qq.com,api.deepseek.com,api.siliconflow.cn,*.baiten.cn,searxng…` | 冒烟②：curl Google 走代理通；curl 佰腾直连通 |
| P1.5 | docker-compose 起 backend+celery+redis | 启动日志无错；`registered routes` 行出现 |
| P1.6 | **冒烟清单**（model-config.deepseek.md §4 四条）：文字提问 / 图片外观比对 / 知识库问答 / 检索架构理解(确认 deepseek 非 openrouter) | 全绿才进 P2 |

回滚：P1 全部可逆（机器未接流量，重装/改配置即可）。

### P2 · 数据库迁移（P1 绿后，选低峰窗口）

> **完整分钟级手册见 `RUNBOOK_DB_MIGRATION.md`**（含实测数据规模、逐条命令、校验清单、Redis 键级复制、回滚预案）。
> 关键设计：**单主一致性**——切流瞬间起全部写只指向国内主库，美国应用与中国实例**同窗口切换**（杜绝双主分裂窗口）。

摘要步骤：
1. 预检与快照（表清单/字符集/应用账号/白名单/TLS；记 binlog 锚点）
2. 停写：`docker stop backend celery`（美国机，分钟级）
3. 备份：`mysqldump --single-transaction --routines --triggers` + gzip + md5 → 传输 → 导入国内（12MB，秒级）
4. 校验（切流前必须全绿）：逐表 COUNT + 抽样 + `CHECKSUM TABLE` 两端一致
5. 同窗切流：国内实例与美国实例 .env 同时指向国内主库 → 重启 → 双端写同一库验证
6. 观测 15 分钟（错误日志/接口探活/双端新行可见）
7. 保留美国旧库冻结为回滚源 7 天

**回滚**：整窗回滚——停写 → 美国 .env 指回旧库 → 重启；窗口内新增数据导出留档（量级极小）。详见手册 §6。

### P3 · 微信登录服务端（代码先行，P0.4 后联调）

| 步骤 | 动作 | 验证 |
|---|---|---|
| P3.1 | W3 代码落地（见 §8 工件清单）：`wechat_login.py` + `wechat_auth.py` + api.py include + 单测 | pytest 过（PYTHONUTF8=1） |
| P3.2 | 部署到**中国实例**（美国生产不动） | 语法/重启无错 |
| P3.3 | 填真实 AppID/Secret → 小程序开发者工具发起登录 | `/auth/wechat` 返回 token+user_id |
| P3.4 | 兼容性验证：新微信用户建档；**老用户场景**用已有邮箱账号同一手机测试后续绑定流程（二期） | users 表可见 oauth_provider='wechat' 行 |
| P3.5 | growth_daily_stats.py 加 auth:wechat → 重新出表 | 日表含小程序注册 |

### P4 · 美国侧过渡（P2 同窗切流后的性能决策；有闸门）

> P2 切流时美国 .env 已直连国内主库（无池）。P4 = 依据实测数据决定是否加性能层。

| 闸门 G4 | 内容 |
|---|---|
| 依据 | **延迟探针**：美国机→国内库 3 天实测（建连耗时/p95 查询/丢包/整轮模拟）；另测网页端真实链路降级幅度 |
| 三选一 | ① 直连（接受无池降级，仅抬连接超时）② 美国机跑 ProxySQL 本地代理（代码零改动，推荐）③ 美国只读从库（读本地、写跨境） |

| 步骤 | 动作 | 验证/回滚 |
|---|---|---|
| P4.1 | 切流后立即启动 3 天探针 | 探针报告 |
| P4.2 | 按闸门三选一实施（默认 ② ProxySQL 若探针超标） | web 冒烟：登录/检索/落库 |
| P4.3 | 双端用户统一抽验：同一 user_id 在小程序与 web 均可与会话 | 会话记录互通 |
| P4.4 | 观测 3-7 天（错误率/延迟）；不达标按回滚矩阵处理 | 监控日志 |

### P5 · 小程序端（Taro，与 P0/P3 并行）

| 里程碑 | 内容 | 出口 |
|---|---|---|
| M1 | 工程骨架 + 登录（wx.login + 手机号可选）+ 会话列表 | 测试号可登录 |
| M2 | 对话页（WebSocket 流式→后端加 `/ws/chat`）+ 专利结果面板（复用检索接口与展示架构） | 真机可问答 |
| M3 | 上传查重（wx.uploadFile）、历史会话、分享、隐私政策 | 可提审 |
| M4 | 提审发布（类目：工具-信息查询；注意文案避开"专利代理"资质表述） | 线上 |

> 注：小程序 request 域名 = P0.3 备案域名；WSS 与上传域名同一备案域名下即可。

### P6 · 收尾

| 步骤 | 动作 |
|---|---|
| P6.1 | 监控：新增页面/接口错误率看板；日志轮转与归档（国内实例 .logs 容量策略） |
| P6.2 | 美国旧库/旧资源按保留期下线；更新本手册状态为已执行 |
| P6.3 | 增长表双端统一（访客/注册/登录列在 GA4 token 到位后合并出完整漏斗） |

---

## 6. 代码/配置工件清单（全部经评审后开始写）

| 工件 | 类型 | 阶段 | 状态 |
|---|---|---|---|
| `deploy/china/MIGRATION_PLAN.md` | 总方案 | — | ✅ 已备(待评审) |
| `deploy/china/model-config.deepseek.md` | 配置模板 | P1 | ✅ 已备 |
| `deploy/china/RUNBOOK_DB_MIGRATION.md` | 迁移执行手册 | P2 | ✅ 已备(待评审) |
| `deploy/china/RUNBOOK_GO_LIVE.md` | 上线编排手册 | L1-L8 | ✅ 已备(待评审) |
| `deploy/china/w3_users_wechat_migration.sql` | SQL | P2/P3 | ⏳ |
| `sources/user/wechat_login.py` | 代码 | P3 | ⏳ |
| `api_routes/wechat_auth.py` | 代码 | P3 | ⏳ |
| `api.py` include（本地+生产双版） | 代码 | P3 | ⏳ |
| `tests/test_wechat_login.py` | 测试 | P3 | ⏳ |
| `growth_daily_stats.py` +auth:wechat | 分析 | P3 | ⏳ |
| tinyproxy 配置 + `.env` 出站模板 | 配置 | P1 | ⏳ |
| 探针脚本(可选) | 工具 | P4 闸门 | ⏳ |
| Taro 工程 + WSS `/ws/chat` | 代码 | P5 | ⏳ |

## 7. 风险登记

| 风险 | 等级 | 缓解 |
|---|---|---|
| 备案周期超预期(1-4周) | 高 | P3/P5 全并行；备案不阻塞代码 |
| 美国→国内库直连降级致 web 投诉 | 高 | P4 闸门三选一(ProxySQL 首选) |
| users ALTER 影响存量代码(非空假设) | 中 | SQL 只在中国主库执行；回归冒烟；代码审计 email 用法 |
| 小程序提审类目/资质 | 中 | 文案边界；先咨询客服 |
| 跨境探针未测就切 | 中 | P4.1 强制 3 天数据 |
| 微信 code2session 从国内实例调用失败 | 低 | 国内直连微信=最优路径；失败降级提示重试 |

## 8. 待确认事项（评审时回答）

1. P0 主体/备案由谁操作、预计主体名称与所属云账号？
2. P5 小程序 MVP 功能范围（仅对话检索 vs 含上传查重）？
3. P4 闸门倾向（直连降级 / ProxySQL / 从库）？
4. 小程序是否需要手机号快捷登录（二期绑定）？
5. 评审通过后按 §8 工件清单从 W3 代码开始，对吗？
