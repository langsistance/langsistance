# 长任务上线步骤与操作流程

## 1. 数据库变更

### 1.1 更新知识描述（已启用的能力卡片）

```bash
mysql -u <user> -p -h <host> <database> < sql/update_deep_research_description.sql
```

执行前确认 `scene_id` 和 `type`：
```sql
SELECT id, scene_id, `type`, question, description FROM knowledge WHERE scene_id = 1 AND `type` = 3;
```

### 1.2 确认知识库中的工具配置

检查场景 1 下所有工具是否正确配置（URL、params、超时）：

```sql
SELECT k.id, k.question, k.answer, k.`type`, t.title, t.url, t.timeout
FROM knowledge k
LEFT JOIN tools t ON k.tool_id = t.id AND t.status = 1
WHERE k.status = 1 AND k.scene_id = 1
ORDER BY k.`type` DESC;
```

关键工具必须存在：
- `search_patent_by_assignee`（按受让人搜索）
- `search_patent_by_key_word`（按关键字搜索，仅用户明确要求时使用）
- `get_patent_documents_application_number`（下载专利说明书）

### 1.3 创建 long_tasks 表（如果不存在）

```sql
CREATE TABLE IF NOT EXISTS long_tasks (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    task_id VARCHAR(64) NOT NULL UNIQUE,
    session_id VARCHAR(32),
    user_id BIGINT NOT NULL,
    scene_id INT,
    task_type VARCHAR(32) DEFAULT 'patent_analysis',
    input_params JSON,
    status VARCHAR(16) DEFAULT 'pending',
    current_phase VARCHAR(32),
    progress INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_status (user_id, status),
    INDEX idx_task_id (task_id),
    INDEX idx_session (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

## 2. 配置文件

### 2.1 config.ini

```ini
[LONG_TASK]
provider_family = deepseek          # LLM 提供商：deepseek / minimax
max_patents = 20                     # 全局最大专利数（实际由代码根据来源限制）
vision_enabled = true                # 是否启用 vision LLM（扫描件 PDF 分析）
vision_provider = minimax            # Vision 模型提供商
vision_model = MiniMax-M3            # Vision 模型名称
```

### 2.2 环境变量

```bash
# USPTO API
USPTO_API_KEY=your_uspto_api_key

# Redis（Celery broker + 状态存储）
REDIS_HOST=redis_test
REDIS_PORT=6379

# MySQL
MYSQL_HOST=your_mysql_host
MYSQL_PORT=3306
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=your_database

# COS / 对象存储（报告文件存储）
COS_SECRET_ID=your_secret_id
COS_SECRET_KEY=your_secret_key
COS_REGION=your_region
COS_BUCKET=your_bucket
```

## 3. 部署

### 3.1 重新构建镜像并启动

代码变更后，需要重新构建 backend 和 celery 镜像，然后重启服务。

```bash
# 构建并启动（推荐）
docker compose up -d --build backend celery

# 或分步执行
docker compose build backend celery
docker compose up -d backend celery
```

**Celery 并发说明**：`--concurrency=1` 是因为 USPTO API Key 有限流，同时只能发一个请求（`http_outbound.py` 里已有 Semaphore 控制并发）。如果 docker-compose.yml 中没设置，可在 command 中加上。

### 3.2 仅重启（无代码变更时）

```bash
docker compose restart backend celery
```

### 3.3 验证服务

```bash
# Celery Worker 是否在线
docker compose logs celery | grep "ready"

# Backend 是否正常
curl http://localhost:7777/health
```

## 4. 功能验证

### 4.1 搜索分析场景

1. 新建对话
2. 输入：「帮我看看特斯拉近期的专利在自动驾驶领域有什么新进展」
3. 预期行为：
   - 卡片显示「深度分析进行中」
   - 进度条逐步更新（检索 → 分析框架 → 逐篇分析 → 撰写报告 → 导出文件）
   - 完成后显示下载按钮（DOCX + PDF）
4. 检查日志：
   ```bash
   tail -f .logs/long_task_pipeline.log
   ```
   关键日志：
   ```
   MODE=search_extract
   PHASE0 select_tool → tool_title=search_patent_by_assignee
   PHASE0 llm_params → body.q 只含公司名，不含技术关键词
   PHASE2 COMPLETE → successful=X, failed=0
   ```

### 4.2 指定专利分析场景

1. 输入：「分析专利 17429113、18012525、18331482」（纯数字）
2. 预期行为：
   - MODE=direct_ids
   - patent_source 自动检测为 uspto
   - 逐篇下载分析
3. 检查：分析数据表第一列「专利号」有值

### 4.3 文件上传分析场景

1. 上传 3-5 个 PDF/XML/DOCX 专利说明书文件
2. 输入：「从这几条中筛选出XX相关专利」
3. 预期行为：
   - MODE=file_upload
   - FILE_EXTRACT 日志显示每个文件的提取字符数
   - 提取失败的文件走 vision/OCR fallback

### 4.4 追问场景

1. 在完成一次分析后，在同一对话中继续追问
2. 输入：「这里面哪些已授权？」
3. 预期行为：
   - scenario=conversation_refs
   - 自动引用历史对话中的专利数据

### 4.5 无关联追问（隔离验证）

1. 在已完成文件上传分析的对话中
2. 输入新问题：「帮我看看特斯拉近期的专利在自动驾驶领域有什么新进展」
3. 预期行为：
   - **不应该**引用历史文件上传的数据
   - MODE=search_extract
   - patent_ids=[] 进入搜索流程

### 4.6 页面刷新恢复

1. 长任务进行中时刷新页面
2. 预期行为：
   - 页面加载后显示进度卡片
   - 卡片进度持续更新（不卡在某个百分比）
   - 完成后自动显示下载按钮

## 5. 故障排查

### 5.1 任务一直排队

```bash
# 查看用户队列锁
docker compose exec redis redis-cli GET "lt:user:<user_id>:running"
docker compose exec redis redis-cli LRANGE "lt:user:<user_id>:queue" 0 -1

# 清锁（强制释放）
docker compose exec redis redis-cli DEL "lt:user:<user_id>:running"
docker compose exec redis redis-cli DEL "lt:user:<user_id>:queue"
```

### 5.2 查看任务状态

```bash
# Redis 中的任务状态
docker compose exec redis redis-cli GET "lt:<task_id>:status" | python -m json.tool

# MySQL 中的任务记录
mysql> SELECT task_id, status, current_phase, progress FROM long_tasks WHERE task_id = 'lt_xxx';
```

### 5.3 关键日志文件

| 日志 | 内容 |
|------|------|
| `.logs/long_task_pipeline.log` | 长任务 pipeline 全流程 |
| `.logs/dynamic_tool_params.log` | USPTO API 请求/响应 |
| `.logs/text_extractor.log` | Vision/OCR 提取详情 |
| `.logs/backend.log` | 任务提交、排队、错误 |
| `.logs/provider.log` | LLM 调用详情 |

### 5.4 常见问题

| 问题 | 可能原因 | 解决 |
|------|---------|------|
| 全部失败 `fallback path not implemented` | patent_source 未正确设为 uspto | 检查 auto-detect 逻辑、params 传递 |
| PDF 提取失败 `uspto_spec_extract_empty` | pypdf 无法提取扫描件文字 | Vision/OCR 自动 fallback，检查 vision_provider 配置 |
| MySQL `InterfaceError: (0, '')` | 连接超时断开 | 已加固 `get_db_connection()`，需部署最新代码 |
| 卡片进度不更新（刷新后） | SSE 消息保存时 taskId 丢失 | 已通过 load 阶段过滤修复，需部署最新代码 |
| SSP API 返回 0 结果 | 搜索参数过于严格（附加了标题过滤） | select_tool prompt 已加固，仅用公司名搜索 |

## 6. 模型配置

### 6.1 当前模型分配

| 阶段 | 模型 | 用途 |
|------|------|------|
| Phase 0 工具选择 | deepseek-chat | 选工具 + 生成参数 |
| Phase 1 分析框架 | deepseek-chat | 生成表格列定义 |
| Phase 2 逐篇分析 | deepseek-reasoner | 专利文本深度分析 |
| Phase 2 小结 | deepseek-reasoner | 单篇专利一句话总结 |
| Phase 3 报告撰写 | deepseek-reasoner | 逐章节生成报告 |
| Vision fallback | MiniMax-M3 | 扫描件 PDF 图片分析 |
| 场景分类 (_classify_long_task_async) | deepseek-chat | 判断 scenario + 提取 ID |

### 6.2 模型切换

如需切换到 GPT，修改 `config.ini`：
```ini
[LONG_TASK]
provider_family = openai
```

并根据 Provider 配置设置对应的 API Key 环境变量。

## 7. 配置调优

| 配置项 | 位置 | 默认值 | 说明 |
|--------|------|--------|------|
| 中国专利上限 | `celery_worker.py:_get_max_patents_for_source` | 10 | CNIPA 单次最多 10 条 |
| 美国专利上限 | `celery_worker.py:_get_max_patents_for_source` | 50 | USPTO 单次最多 50 条 |
| 任务超时 TTL | `user_queue.py` | 86400s (24h) | Redis 锁 TTL |
| 心跳超时 | `user_queue.py:_is_task_terminal` | 300s | 任务无响应 5 分钟视为死锁 |
| 轮询间隔 | `page.tsx:startLongTaskPolling` | 3000ms | 前端轮询任务状态间隔 |
| LLM 超时 | `patent_analyzer.py:analyze_single_patent` | 60s | 单篇专利分析超时 |
