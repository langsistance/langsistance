# 运行手册：数据库主迁移 + 双端切流（P2）

> 配套总方案：`MIGRATION_PLAN.md` §P2 / §P4
> 实测数据规模（2026-09-08 生产盘点，MySQL 8.0.33，库总 ~12MB）：

| 表 | 行数 | 大小 | | 表 | 行数 | 大小 |
|---|---|---|---|---|---|---|
| conversations | 120 | 8.06MB | | feedback | 1 | 0.05MB |
| long_tasks | 135 | 3.58MB | | messages | 2 | 0.05MB |
| knowledge | 298 | 0.25MB | | user_scenes | 73 | 0.05MB |
| tools | 186 | 0.14MB | | knowledge_share | 6 | 0.03MB |
| users | 77 | 0.06MB | | scenes | 0 | 0.02MB |

Redis：**219 keys**（~200 个 `knowledge_embedding_NNN` + 2 个 patent 缓存；无会话/计数残留——会话已持久化 MySQL，计数带 TTL 自灭）。
**结论：全量 dump+导入 ≈ 秒级；整窗含校验 < 15 分钟；无需 binlog 增量追赶。**

---

## 0. 设计原则（先读）

1. **单主一致性**：切换完成瞬间起，**全部写流量只允许指向国内主库**。美国应用与中国实例**同窗口切**，杜绝"双主各写一半"的数据分裂窗口。
2. 迁移窗口选**低峰**（用户 77/日活 ~10，北京时间凌晨 2-5 点为佳）；窗口内美国 backend/celery **短暂停机**（<10 分钟），旧库保持原样作为回滚源。
3. 回滚=整窗回滚：停写 → 美国 .env 指回美国库 → 重启 → 国内停。窗口内中国实例产生的新数据导出留档，明确"可接受丢失"口径（见 §6）。
4. Redis：219 keys **整库键级复制**（DUMP/RESTORE），向量保持原值（同模型 bge-m3，免重嵌）。

---

## 1. 前置检查（T-1 天）

| 步骤 | 命令/动作 | 通过标准 |
|---|---|---|
| 1.1 | 国内 MySQL 8.0.33 就绪（版本需 ≥ 美国同版本，避免 8.0 差异） | `SELECT VERSION()` = 8.0.33 |
| 1.2 | 字符集一致：两端库均 `utf8mb4/utf8mb4_unicode_ci` | 对比 `information_schema` |
| 1.3 | 建应用账号：`CREATE USER 'app'@'<美国机IP>' IDENTIFIED BY '<强密码>'; CREATE USER 'app'@'<国内机IP>' ...; GRANT ALL ON copiioai.* TO ...;` | 两端 IP 均可用账号登录 |
| 1.4 | 安全组：3306 仅放行 5.78.157.14 + 国内机内网 IP；**非标准端口 + 仅白名单**（TLS 见 1.5） | 外部扫描不通 |
| 1.5 | 跨境加密：启用 MySQL TLS（自签 CA 即可）或腾讯云 CDB 自带 SSL；应用连接串加 `ssl_ca` | `SHOW STATUS LIKE 'Ssl_cipher'` 非空 |
| 1.6 | 慢查询/错误日志落盘路径确认（国内实例 .logs 策略） | 文件可写 |
| 1.7 | 预演一次完整 dry-run（在美国→国内内网测试库跑 §2-§4 全流程） | 演练记录 |

## 2. 迁移窗口执行（分钟级，T 为切流时刻）

### T-30 —— 预检与快照
| 步骤 | 动作 |
|---|---|
| 2.1 | 确认低峰：`analytics.log` 近 1h 无 query_stream / 或用户确认可停机 |
| 2.2 | 记录基线：`SHOW MASTER STATUS;`（binlog 文件名+pos，作回滚对账锚点）；记 `MAX(user_id)`、各表 `MAX(id)` |

### T-5 —— 停写
| 步骤 | 动作 |
|---|---|
| 2.3 | `docker stop backend celery`（美国机）——写流量归零 |
| 2.4 | 确认无连接：`SHOW PROCESSLIST;` 无 app 连接（SYSTEM 除外） |

### T-4 —— 备份与传输（美 → 国）
| 步骤 | 命令（美国机执行） |
|---|---|
| 2.5 | `mysqldump -h127.0.0.1 -u"$MYSQL_USER" -p"$MYSQL_PASSWORD" --single-transaction --routines --triggers --events --hex-blob --set-gtid-purged=OFF "$MYSQL_DATABASE" \| gzip > /tmp/copiioai_$(date +%Y%m%d_%H%M).sql.gz` |
| 2.6 | `md5sum /tmp/*.sql.gz`（记录） |
| 2.7 | 传输：`scp` 直传或经 COS 中转；12MB 秒级 |
| 2.8 | 国内机解压后 `md5sum` 对比 = 2.6 值 |

### T-2 —— 导入（国内机）
| 步骤 | 命令/动作 |
|---|---|
| 2.9 | 建库（若未建）：`CREATE DATABASE copiioai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;` |
| 2.10 | 导入：`mysql -h127.0.0.1 -uapp -p copiioai < dump.sql`（<15s 预期；如含外键先 `SET FOREIGN_KEY_CHECKS=0` 再置 1） |
| 2.11 | users 迁移 ALTER（若建库时未并入）：执行 `deploy/china/w3_users_wechat_migration.sql` | 

### T-1 —— 校验（只读，切流前必须全绿）
| 步骤 | 动作 |
|---|---|
| 2.12 | **逐表行数对比**：两端各跑 `SELECT table_name, COUNT(*) ...`（10 表全比） |
| 2.13 | **抽样比对**：users 全表 dump 对比；conversations/knowledge 各抽样最新 20 行关键列 |
| 2.14 | `CHECKSUM TABLE users, conversations, knowledge, ...;` 两端输出一致（表小，全表 checksum 秒级） |
| 2.15 | 自增锚点：`MAX(user_id)` 等与 2.2 记录一致 |

**任一校验失败 → 中止切流，执行 §6 回滚（此时国内库未接流量，丢弃重导即可）。**

### T0 —— 切流（双端同时指向国内主库）
| 步骤 | 动作 |
|---|---|
| 2.16 | 国内实例 .env：`DB_HOST=127.0.0.1(国内)` 已配置 → 启动国内 backend/celery/redis |
| 2.17 | 美国实例 .env：`DB_HOST=<国内机公网IP>` `DB_PORT=<端口>` + 连接超时上调（`connect_timeout=15, read_timeout=30`）→ 启动美国 backend/celery |
| 2.18 | 验证写路径：小程序侧（国内）发 1 条提问 → 落库国内；web 侧（美国）发 1 条 → 落库**同一国内库**（`SELECT` 可见双端各自新行） |

### T+15 —— 观测（窗口正式结束前）
| 步骤 | 动作 |
|---|---|
| 2.19 | 双端错误日志 15 分钟零 DB 报错（backend.log / 新实例日志） |
| 2.20 | 关键接口探活：`/auth/*`、提问、知识库检索 curl 各 200 |
| 2.21 | 确认国内库持续有新行、美国库不再增长（旧库只读冻结 = 天然回滚源） |

**观测失败 → 整窗回滚（§6）。观测通过 → 通知全量恢复，记录"切换锚点"，保留美国旧库 7 天。**

---

## 3. Redis 键级复制（219 keys，可与 §2 并行）

| 步骤 | 动作 |
|---|---|
| 3.1 | 美国机导出清单：`docker exec redis redis-cli --scan` 存文件（确认 219 条） |
| 3.2 | 逐键迁移（小脚本，DUMP+RESTORE）：`for k in $(cat keys.txt); do docker exec redis redis-cli --raw DUMP "$k" | ... RESTORE 国内 "$k" 0 ... ; done`；RESTORE 前 `FLUSHDB` 国内实例确认干净 |
| 3.3 | 校验：两端 `DBSIZE` = 219；抽样 5 键 `STRLEN`/`TYPE` 一致 |
| 3.4 | 不迁移的键类型说明：计数/缓存类（`api_usage_*` 等）过期即弃，无需迁；`firebase_uid_*` 映射为可重建缓存（用户请求时自动回填） |

## 4. 切流后（P4 关联）性能观测

| 步骤 | 动作 |
|---|---|
| 4.1 | 美国→国内直连跑 **3 天探针**（脚本待写）：记录建连耗时、p95 查询、丢包、整轮模拟耗时 |
| 4.2 | 按 `MIGRATION_PLAN.md` §P4 闸门三选一：直连接受 / 美国机 ProxySQL / 只读从库 |

## 5. 收尾

| 步骤 | 动作 |
|---|---|
| 5.1 | 美国旧库 `mysqldump` 二次全备归档（含 binlog 到切换点），冷存 |
| 5.2 | 7 天后确认稳定：删除/停用美国旧库容器（或保留只读备查） |
| 5.3 | 更新 `MIGRATION_PLAN.md` 状态为"已迁移"，记录实际窗口时长 |

## 6. 回滚预案（整窗回滚）

| 触发 | 动作 | 数据影响 |
|---|---|---|
| 2.12-2.15 校验失败 | 国内库丢弃重导（未接流量，无影响） | 无 |
| 观测期错误率超阈 / 核心功能不可用 | ① `docker stop` 双端 backend/celery ② 美国 .env 指回美国库并启动 ③ 国内实例停止 ④ 报告窗口内国内产生的新数据清单（可接受：窗口极短，通常为 0-数十行，导出留档） | 窗口内新增数据丢失（留档） |
| 7 天内严重问题 | 同左；美国库为切换点快照+冻结状态，数据完整 | 同上 |
