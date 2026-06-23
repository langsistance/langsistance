# 批量专利分析功能 — 技术设计文档

**版本**: v1.0
**日期**: 2026-06-23
**状态**: 待评审

---

## 一、产品概述与目标

### 1.1 背景

CopiioAI 已接入中国专利（DI 平台 / CNIPA）和美国专利（USPTO）数据源，用户可以通过自然语言检索专利。当前缺失的能力是：对检索到的一批专利进行批量分析，生成结构化报告。

### 1.2 目标

新增**批量专利分析**能力，作为一个"长任务（Long Task）"场景：

- 用户通过关键词检索、上传 PDF、提交专利 ID 三种方式指定一批专利
- 系统按用户问题动态生成分析表格，逐专利串行深度分析，最终生成 Word/PDF 报告
- 长任务在后台独立执行，用户可关闭网页后返回查看进展
- 支持 DeepSeek V4-Flash/Pro 和 MiniMax 2.7-highspeed 两套模型，后端手动切换
- MVP 阶段最多处理 20 个专利

### 1.3 反目标

- MVP 不支持用户上传 PDF（Phase 3）
- MVP 不处理复杂 USPTO 集成（先聚焦中国专利）
- 不在前端暴露模型切换选项（对用户透明）
- 不引入新的基础设施（不拆分新服务）

---

## 二、核心概念

### 2.1 Session 懒创建

```
用户检索专利（多轮对话） → 不创建 session
用户说"分析这些专利"     → 触发长任务 → 创建 session
                              ↓
                         前几轮对话写入 session
                         长任务过程写入 session
                              ↓
                         用户后续对话继续写入
                         关闭网页 → 打开 → 恢复 session
```

**关键规则：**
- 只有触发长任务时才创建 session
- 触发前的对话由前端携带到长任务请求中
- 长任务创建后，该 session 内后续无论短任务还是长任务都记录
- Session 按 user_id 隔离

### 2.2 长任务（Long Task）

长任务是 Knowledge 的新 type（type=3），表示一个需要后台异步执行、有明确进度和最终产物的任务流程。

| 属性 | 说明 |
|------|------|
| 入口 | 意图识别匹配到 type=3 knowledge |
| 上下文 | 关联 scene，可用场景下全部 knowledge + tools |
| 生命周期 | pending → running → completed / failed |
| 执行单元 | Celery Worker（独立进程） |
| 进度模型 | 多阶段串行执行，每阶段有状态和百分比 |
| 最终产物 | Word + PDF 报告 |

---

## 三、数据流全景

```
┌─────────────────────────────────────────────────────────────┐
│                        用户浏览器                            │
│                                                             │
│  1. 用户在对话页正常发消息: "分析这50个专利"                   │
│     POST /query_stream（走现有 SSE 流式接口）                 │
│     {query, conversation_history: [...]}                    │
│                                                             │
│  2. SSE 流返回:                                             │
│     event: long_task_created                                │
│     data: {task_id: "lt_...", session_id: "sess_..."}       │
│                                                             │
│  3. 前端停止 SSE 监听，切换到轮询模式                         │
│     GET /long_task/{task_id}/status                         │
│     ← {phase, step, progress, table_columns, table_rows}    │
│                                                             │
│  4. ... 用户关闭网页，回来，继续轮询 ...                       │
│                                                             │
│  5. 任务完成 → GET /long_task/{task_id}/report → 下载        │
└─────────────────────────────────────────────────────────────┘

                         │
                         ▼

┌─────────────────────────────────────────────────────────────┐
│                  FastAPI（US 服务器）                         │
│                                                             │
│  POST /query_stream（现有接口，增强意图识别）                  │
│  ├── 现有流程：embedding 检索 → LLM 路由                     │
│  ├── 匹配到 type=3 knowledge → 确认是长任务                  │
│  ├── 创建 session（MySQL conversations 表）                  │
│  ├── 将 conversation_history 写入 session                   │
│  ├── 生成 task_id                                           │
│  ├── 提交 Celery 任务（Redis broker）                        │
│  └── SSE 推送: {event: "long_task_created", data: {task_id}}│
│                                                             │
│  GET /long_task/{task_id}/status                            │
│  └── 读 Redis → 返回当前状态、步骤列表、表格行                 │
│                                                             │
│  GET /long_task/{task_id}/report?format=pdf|docx            │
│  └── 返回报告文件                                            │
└─────────────────────────────────────────────────────────────┘

                         │
             Redis Broker │
                         ▼

┌─────────────────────────────────────────────────────────────┐
│                Celery Worker（同一台 US 服务器）              │
│                --pool=solo --concurrency=1                  │
│                                                             │
│  execute_long_task(task_id, params):                        │
│                                                             │
│  Phase 1: 生成表格列定义（Flash）                            │
│           ├── 根据用户问题动态确定分析维度                     │
│           └── 输出: {columns: [...], patent_count: N}       │
│                                                             │
│  Phase 2: 逐专利串行处理（每个专利 4 步）                     │
│           for patent in patents:                            │
│               ├── 2a: 下载专利说明书（DI 平台 / USPTO）       │
│               ├── 2b: Pro 分析专利，按 columns 逐列填充       │
│               ├── 2c: 更新表格行 + 生成该专利简单总结          │
│               ├── 2d: 写 Redis checkpoint + 前端轮询可见       │
│               └── 继续下一个专利                               │
│                                                             │
│  Phase 3: 生成报告（Pro，全动态）                            │
│           ├── 3a: 根据用户问题 + 表格内容 → 生成报告大纲       │
│           ├── 3b: 按大纲逐章节撰写                            │
│           └── 输出完整报告文本                                │
│                                                             │
│  Phase 4: 导出文件                                           │
│           ├── 生成 Word (.docx)                              │
│           ├── 生成 PDF                                       │
│           └── 写入持久化存储                                  │
└─────────────────────────────────────────────────────────────┘

                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   DeepSeek API     MiniMax API      DI 平台
   (中国模型)        (中国模型)       (下载专利)
```

---

## 四、数据库 Schema 变更

### 4.1 conversations 表（新增）

```sql
CREATE TABLE conversations (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id      VARCHAR(64) NOT NULL UNIQUE COMMENT 'session 唯一标识',
    user_id         BIGINT UNSIGNED NOT NULL COMMENT '用户 ID',
    scene_id        BIGINT UNSIGNED DEFAULT NULL COMMENT '关联场景',
    title           VARCHAR(256) DEFAULT '' COMMENT 'session 标题（可由 LLM 生成）',
    messages        JSON NOT NULL COMMENT '对话消息数组 [{"role":"user|assistant","content":"...","patent_data":null|[...]}]',
    long_task_ids   JSON DEFAULT NULL COMMENT '关联的长任务 task_id 列表',
    status          TINYINT UNSIGNED DEFAULT 1 COMMENT '1=active, 2=archived',
    create_time     DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time     DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_id (user_id),
    INDEX idx_create_time (create_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

**messages JSON 结构示例：**
```json
[
  {
    "role": "user",
    "content": "用关键词'深度学习图像识别'检索专利",
    "patent_data": null,
    "timestamp": "2026-06-23T10:00:00"
  },
  {
    "role": "assistant",
    "content": "检索到 50 个相关专利...",
    "patent_data": [
      {"patent_id": "CN202310001234A", "title": "...", ...},
      {"patent_id": "CN202310001235A", "title": "...", ...}
    ],
    "timestamp": "2026-06-23T10:00:30"
  }
]
```

### 4.2 knowledge 表变更

无需 DDL 变更。type 值扩展：

| type | 含义 |
|------|------|
| 1 | 普通知识（system prompt 模板） |
| 2 | workflow（多步工具链） |
| **3** | **长任务入口（新增）** |

type=3 的 knowledge 记录示例：
```json
{
  "question": "批量分析专利",
  "description": "用户要求对一批专利进行综合分析、对比、生成报告",
  "answer": "这是一个长任务入口，用于触发批量专利分析流程",
  "type": 3,
  "params": {
    "type": "long_task",
    "agent": "patent_analysis",
    "output_formats": ["pdf", "docx"]
  }
}
```

> `max_patents` 不在 knowledge 里硬编码，而是从 `config.ini` 的 `[LONG_TASK]` 段读取。修改配置后重启 Worker 生效。

```
}
```

### 4.3 long_tasks 表（新增）

```sql
CREATE TABLE long_tasks (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    task_id         VARCHAR(64) NOT NULL UNIQUE COMMENT 'Celery task UUID',
    session_id      VARCHAR(64) DEFAULT NULL COMMENT '关联 session',
    user_id         BIGINT UNSIGNED NOT NULL,
    scene_id        BIGINT UNSIGNED DEFAULT NULL,
    task_type       VARCHAR(64) NOT NULL DEFAULT 'patent_analysis' COMMENT '任务类型',
    input_params    JSON NOT NULL COMMENT '输入参数（query, patent_ids, model_family 等）',
    status          VARCHAR(32) NOT NULL DEFAULT 'pending' COMMENT 'pending|running|completed|failed',
    progress        INT DEFAULT 0 COMMENT '进度百分比 0-100',
    current_phase   VARCHAR(64) DEFAULT NULL COMMENT '当前阶段标识',
    current_step    VARCHAR(512) DEFAULT NULL COMMENT '当前步骤描述（自然语言）',
    phases          JSON DEFAULT NULL COMMENT '阶段详情 [{phase, steps: [{description, status}]}]',
    table_columns   JSON DEFAULT NULL COMMENT 'Phase1 生成的列定义',
    table_rows      JSON DEFAULT NULL COMMENT 'Phase2 逐行填充的数据',
    result_summary  TEXT DEFAULT NULL COMMENT 'Phase3 生成的报告文本',
    report_files    JSON DEFAULT NULL COMMENT '生成的文件 [{format, path, size, created_at}]',
    error_message   TEXT DEFAULT NULL COMMENT '失败原因',
    celery_task_id  VARCHAR(128) DEFAULT NULL COMMENT 'Celery 内部 task id',
    create_time     DATETIME DEFAULT CURRENT_TIMESTAMP,
    update_time     DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    complete_time   DATETIME DEFAULT NULL,
    INDEX idx_user_id (user_id),
    INDEX idx_session_id (session_id),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

---

## 五、API 设计

### 5.1 长任务的触发：复用现有查询接口

长任务没有独立的触发 API。用户像平常一样发消息到现有的查询接口，后端内部做意图识别。

```
POST /query_stream（现有接口，无需前端改动）
Authorization: Bearer {firebase_token}
```

**Request（跟现有格式一致）：**
```json
{
  "query": "分析这些专利的技术分布、创新点和竞争风险",
  "conversation_history": [
    {"role": "user", "content": "用关键词'深度学习图像识别'检索专利"},
    {"role": "assistant", "content": "检索到50个相关专利...", "patent_data": [...]}
  ],
  "user_id": "..."
}
```

> `conversation_history` 是现有 `general_agent.py` 里 `create_agent()` 已经接收的参数，前端在发送请求时把最近几轮对话一起带上，不需要后端维护短对话临时缓存。

**SSE 响应 — 当识别为长任务时，推送特殊事件：**
```
data: {"type": "status", "message": "正在启动批量专利分析...", "transient": false}

data: {"type": "long_task_created", "task_id": "lt_20260623_a1b2c3d4", "session_id": "sess_20260623_e5f6g7h8"}

data: {"type": "status", "message": "长任务已提交，task_id=lt_20260623_a1b2c3d4，可通过该 ID 查询进度"}

data: [DONE]
```

**后端逻辑（在 `create_agent()` 的流程中）：**
1. 现有流程：embedding 检索 → LLM 路由选择 knowledge
2. LLM 路由匹配到 type=3 knowledge → 确认是长任务
3. `create_agent()` 返回特殊标记（不走正常 agent 调用路径）
4. `run_pipeline()` 检测到长任务标记：
   a. 如果用户此时没有活跃 session，创建新 session
   b. 将 `conversation_history` 写入 session
   c. 提取对话历史中的 `patent_ids`
   d. 生成 `task_id`
   e. 创建 `long_tasks` 记录（status=pending）
   f. **从场景配置中读取 `model_family`（DeepSeek / MiniMax）**
   g. `long_task_execute.delay(task_id, params)` 提交 Celery
   h. SSE 推送 `long_task_created` 事件
   i. SSE 推送最终状态消息，关闭流

**前端收到 `long_task_created` 事件后的行为：**
1. 停止当前 SSE 监听
2. 记录 `task_id` 和 `session_id`
3. 切换到长任务轮询视图
4. 开始轮询 `GET /long_task/{task_id}/status`

### 5.2 查询任务状态

```
GET /long_task/{task_id}/status
```

**Response (200):**
```json
{
  "success": true,
  "task_id": "lt_20260623_a1b2c3d4",
  "status": "running",
  "progress": 45,
  "current_phase": "analyzing",
  "current_step": "正在分析第 9/20 个专利（CN202310001234A）：提取核心技术方案...",
  "phases": [
    {
      "phase": "generating_columns",
      "label": "生成分析框架",
      "status": "completed",
      "steps": [
        {"description": "根据问题确定分析维度", "status": "completed"},
        {"description": "生成表格列定义", "status": "completed"}
      ]
    },
    {
      "phase": "analyzing",
      "label": "逐专利分析",
      "status": "running",
      "steps": [
        {"description": "分析 CN202310001234A", "status": "completed"},
        {"description": "分析 CN202310001235A", "status": "completed"},
        ...
        {"description": "分析 CN202310001239A", "status": "running"}
      ]
    },
    {
      "phase": "generating_report",
      "label": "生成报告",
      "status": "pending",
      "steps": []
    },
    {
      "phase": "exporting",
      "label": "导出文件",
      "status": "pending",
      "steps": []
    }
  ],
  "table_columns": ["专利号", "技术领域", "核心技术方案", "创新点", "相关度"],
  "table_rows": [
    {
      "patent_id": "CN202310001234A",
      "技术领域": "G06V 计算机视觉",
      "核心技术方案": "双流注意力机制 + 残差网络",
      "创新点": "解决了小样本下过拟合问题",
      "相关度": "★★★★☆"
    },
    ...
  ]
}
```

### 5.3 获取报告文件（持久化，可反复下载）

```
GET /long_task/{task_id}/report?format=pdf
GET /long_task/{task_id}/report?format=docx
```

**Response:** 文件下载（`Content-Disposition: attachment`）

**持久化保证：**
- 报告文件存储在服务端文件系统 `/opt/workspace/reports/{task_id}/`
- 任务完成后文件永久保留，不设过期
- 用户关闭网页、换设备、清缓存后，只要知道 `task_id`（从 session 历史中可以查到），就能重新下载
- `GET /session/{session_id}` 返回的 session 数据中包含关联的 `long_task_ids`，前端可据此构造下载链接

**报告文件存储路径结构：**
```
/opt/workspace/reports/
├── lt_20260623_a1b2c3d4/
│   ├── report.pdf
│   ├── report.docx
│   └── metadata.json
├── lt_20260624_e5f6g7h8/
│   ├── report.pdf
│   └── report.docx
└── ...
```

### 5.4 Session 相关 API

```
GET /session/{session_id}
        → 获取 session 详情和对话历史
        → 返回中包含 long_task_ids，前端据此构造下载链接
GET /sessions?user_id={uid}
        → 获取用户所有 session 列表
        → 每个 session 摘要包含 long_task_ids 和对应报告是否可下载
POST /session/{session_id}/message
        → 追加消息到 session（短任务对话时调用）
DELETE /session/{session_id}
        → 归档 session（不删除报告文件）
```

---

## 六、Celery Worker 设计

### 6.1 Worker 配置

```python
# celery_worker.py
app = Celery('patent_tasks', broker=REDIS_URL, backend=REDIS_URL)

app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_track_started=True,
    task_acks_late=True,          # 任务完成后才 ack，防止崩溃丢失
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # concurrency=1 时只预取一个
)

# Worker 启动时从 config.ini 读取：
#   [LONG_TASK] provider_family → 决定用哪套 Provider
#   [LONG_TASK] max_patents     → 单次分析最大专利数
# 修改后重启 Worker 生效

# 启动命令:
# celery -A celery_worker worker --pool=solo --concurrency=1 --loglevel=info
```

### 6.2 防遗漏、防重复机制

```
三个层面保证：

┌─ 层面 1: 输入去重 ───────────────────────────────────┐
│  params['patent_ids']                                 │
│  → sorted(set(patent_ids))   // 去重 + 排序（确定性）    │
│  → [:max_patents]            // 按配置截断              │
│  保证：同一个专利不会分析两次，顺序固定                    │
└──────────────────────────────────────────────────────┘
                        │
                        ▼
┌─ 层面 2: 逐专利状态跟踪（Redis checkpoint）────────────┐
│  每分析完一个专利，写入 checkpoint：                    │
│                                                       │
│  lt:{task_id}:checkpoint  →  {                        │
│    "completed": ["CN...001", "CN...003"],  // 已完成  │
│    "current":   "CN...005",                // 正在处理 │
│    "pending":   ["CN...007", ...],         // 待处理   │
│    "failed":    []                         // 失败的   │
│  }                                                    │
│                                                       │
│  保证：任何时候 Worker 崩溃，重启后知道哪些已完成         │
└──────────────────────────────────────────────────────┘
                        │
                        ▼
┌─ 层面 3: Worker 崩溃恢复（idempotent 续算）──────────┐
│  Worker 重启 → 检查 lt:{task_id}:checkpoint          │
│                                                       │
│  如果 checkpoint 存在 + task 未完成:                    │
│    pending = checkpoint.pending   // 跳过已完成的      │
│    table_rows = checkpoint.completed_rows              │
│    从 pending[0] 继续，不重复分析已完成的专利            │
│                                                       │
│  保证：崩溃不会导致重头再来，已分析的专利不会浪费          │
└──────────────────────────────────────────────────────┘
```

**Redis checkpoint 写入时机（Worker 内）：**
```
每个专利分析完成后，立即写入 checkpoint：
  1. 将当前 patent_id 从 pending 移到 completed
  2. 将新行追加到 checkpoint.completed_rows
  3. 批量写入 Redis（原子操作）

单专利超时（60s）：
  1. 标记该专利为 failed（不阻塞后续专利）
  2. failed 的专利在表格中标注"分析失败"
  3. 继续处理下一个
```

### 6.3 任务编排（全流程串行）

```python
@app.task(bind=True, max_retries=3, default_retry_delay=30)
def execute_patent_analysis(self, task_id: str, params: dict):
    """批量专利分析长任务 — 全流程串行，支持断点续算"""
    try:
        # === 输入去重 + 排序 + 截断 ===
        patent_ids = sorted(set(params['patent_ids']))
        max_patents = get_max_patents_from_config()
        patent_ids = patent_ids[:max_patents]

        # === 检查是否有 checkpoint（崩溃恢复）===
        checkpoint = load_checkpoint(task_id)
        if checkpoint and checkpoint['pending']:
            # 恢复：从上次中断处继续
            table_rows = checkpoint.get('completed_rows', [])
            pending = checkpoint['pending']
        else:
            table_rows = []
            pending = patent_ids

        # === Phase 1: 生成表格列定义 ===
        update_status(task_id, 'generating_columns', 0,
                     '正在生成分析框架...')
        columns = generate_table_columns(
            query=params['query'],
            patent_count=len(patent_ids),
            model=get_flash_model(params['model_family'])
        )
        update_columns(task_id, columns)
        update_progress(task_id, 5)

        # === Phase 2: 逐专利串行处理（下载 → 分析 → 填表 → 总结）===
        for i, patent_id in enumerate(pending):
            try:
                # 2a: 下载专利说明书
                update_single_patent_status(task_id, patent_id, 'downloading',
                    f'正在下载专利文件（{i+1}/{total}）...')
                patent_text = download_patent_document(
                    patent_id=patent_id,
                    source=params.get('patent_source', 'cnipa')
                )

                # 2b: 分析专利
                update_single_patent_status(task_id, patent_id, 'analyzing',
                    f'正在分析（{i+1}/{total}）...')
                row = analyze_single_patent(
                    patent_id=patent_id,
                    patent_text=patent_text,
                    columns=columns,
                    query=params['query'],
                    model=get_pro_model(params['model_family']),
                    timeout=60
                )

                # 2c: 生成该专利简单总结
                row['_summary'] = generate_patent_summary(
                    patent_id=patent_id,
                    row=row,
                    query=params['query'],
                    model=get_pro_model(params['model_family'])
                )

            except PatentDownloadError:
                row = build_failed_row(patent_id, "下载失败")
            except PatentAnalysisTimeout:
                row = build_failed_row(patent_id, "分析超时")
            except Exception as e:
                row = build_failed_row(patent_id, str(e))

            table_rows.append(row)

            # 立即写入 checkpoint（每个专利完成后）
            save_checkpoint(task_id, {
                'completed': [r['patent_id'] for r in table_rows],
                'current': patent_id,
                'pending': pending[i+1:],
                'completed_rows': table_rows,
                'failed': [r['patent_id'] for r in table_rows if r.get('_failed')]
            })

            # 进度: 占总任务 5% ~ 75%，按实际专利总数均分
            total = len(patent_ids)
            progress = 5 + int((i + 1 + len(patent_ids) - len(pending)) / total * 70)
            update_status(task_id, 'analyzing', progress,
                         f'正在分析第 {len(table_rows)}/{total} 个专利（{patent_id}）',
                         table_rows=table_rows)
            write_message_to_session(params['session_id'], ...)

        # === Phase 3: 动态生成报告（Pro）===
        # Step 3a: 生成报告大纲
        update_status(task_id, 'generating_report', 80,
                     '正在根据分析结果规划报告结构...')
        report_outline = generate_report_outline(
            query=params['query'],
            columns=columns,
            table_rows=table_rows,
            model=get_pro_model(params['model_family'])
        )
        update_report_outline(task_id, report_outline)
        update_progress(task_id, 82)

        # Step 3b: 按大纲逐章节撰写
        report = ""
        total_sections = len(report_outline['sections'])
        for idx, section in enumerate(report_outline['sections']):
            section_progress = 82 + int((idx + 1) / total_sections * 8)
            update_status(task_id, 'generating_report', section_progress,
                         f'正在撰写：{section["heading"]}...')
            section_text = generate_report_section(
                section=section,
                query=params['query'],
                columns=columns,
                table_rows=table_rows,
                model=get_pro_model(params['model_family'])
            )
            report += f"## {section['heading']}\n\n{section_text}\n\n"

        update_result_summary(task_id, report)
        update_progress(task_id, 90)

        # === Phase 4: 导出 Word/PDF ===
        update_status(task_id, 'exporting', 92,
                     '正在生成 PDF 文件...')
        pdf_path = export_pdf(report, table_rows, columns)
        update_status(task_id, 'exporting', 96,
                     '正在生成 Word 文件...')
        docx_path = export_docx(report, table_rows, columns)
        update_report_files(task_id, [pdf_path, docx_path])
        update_progress(task_id, 100)

        # 完成
        update_status(task_id, 'completed', 100, '分析完成')

    except Exception as e:
        logger.exception(f'长任务失败: {task_id}')
        update_status(task_id, 'failed', error=str(e))
        raise self.retry(exc=e)
```

### 6.3 状态写入

所有状态写 Redis（快速读写），关键数据写 MySQL（持久化）：

| 数据 | Redis | MySQL |
|------|-------|-------|
| 当前阶段、步骤、进度 | `lt:{task_id}:status` (TTL=24h) | `long_tasks` 表 |
| 表格列 + 行 | `lt:{task_id}:table_columns`, `lt:{task_id}:table_rows` | `long_tasks` 表 |
| 报告文件路径 | — | `long_tasks.report_files` |
| 任务状态转换 | — | `long_tasks` 表 |

**Redis key 设计：**
```
lt:{task_id}:status       → JSON {status, phase, step, progress, phases, ...}
lt:{task_id}:table_cols   → JSON ["列1", "列2", ...]
lt:{task_id}:table_rows   → JSON [{...}, {...}, ...]
lt:{task_id}:checkpoint   → JSON {completed[], current, pending[], completed_rows, failed[]}
```

**TTL 策略：** Redis 数据设 24 小时 TTL。MySQL 为 source of truth。

---

## 七、意图识别设计

### 7.1 核心原则

**意图识别完全由后端完成，前端不感知。** 前端像平常一样把用户消息发到 `/query_stream`，后端通过 embedding 向量检索 + LLM 语义路由判断是否为长任务。

**关键：意图识别是语义理解，不是关键词匹配。** 用户表达"分析"意图的措辞多种多样（"分类"、"对比"、"检索"、"分析"、"看一下"），embedding 检索负责召回候选，LLM 路由负责精确判断语义意图。

### 7.2 用户可能的各种表达方式

同一个分析意图，用户可以用完全不同的话说出来：

| 语义意图 | 用户可能的表达 | 为什么是长任务 |
|---------|-------------|--------------|
| 分类分析 | "帮我把这 20 个专利智能分个类" | 需要逐个理解专利内容后归类 |
| 对比分析 | "华为和腾讯在这些专利上的技术路线有什么不同" | 需要逐个理解再横向比较 |
| 深度检索 | "语义检索一下，哪些专利用到了注意力机制" | 需要理解每个专利的技术方案再匹配 |
| 趋势分析 | "这些专利的技术演进方向是什么" | 需要跨专利归纳找脉络 |
| 风险评估 | "我的产品会不会侵犯这些专利" | 需要逐权利要求深度判断 |
| 布局挖掘 | "这个方向还有哪些技术空白可以布局" | 需要找出已有专利未覆盖的点 |
| 综合报告 | "给我出一份完整分析报告" | 明确的综合分析请求 |

**共同特征：** 需要逐专利语义理解 + 跨专利综合分析。这是 embedding 能捕获的语义信号——不以"分析"二字为必要条件。

### 7.3 Embedding 检索：多入口覆盖

同一场景下建多个 type=3 knowledge，每个的 `question` + `description` 覆盖一类表达方式：

```json
[
  {"question": "批量分析专利",      "description": "对一批专利进行技术分布、趋势、风险等综合分析"},
  {"question": "智能分类专利",      "description": "根据专利技术内容自动理解并归类、标注、分组"},
  {"question": "语义检索技术方案",   "description": "深入理解每个专利的技术方案描述，匹配合特定技术特征的专利"},
  {"question": "专利对比分析",      "description": "横向对比不同专利或不同申请人的技术差异和布局策略"},
  {"question": "专利侵权风险评估",   "description": "将产品特征与专利权利要求逐项对照分析，评估侵权风险"},
  {"question": "技术空白挖掘",      "description": "分析已有专利覆盖范围，识别未被覆盖的创新机会"}
]
```

用户说"帮我分下类" → embedding 跟"智能分类专利"的 description 高相似 → 召回。用户说"搜一下有没有用到 transformer 的" → embedding 跟"语义检索技术方案"相似 → 召回。**多个入口收敛到同一个长任务流程。**

### 7.4 LLM 语义路由

embedding 检索召回候选后，LLM 做最终判断。路由 prompt：

```
你是意图路由器。判断当前用户消息是否需要触发"长任务"。

## 什么情况触发长任务：
- 需要对一批专利用大模型逐个深入理解
- 不是简单的关键词匹配或列表展示
- 需要跨专利的综合分析、对比、归类、判断
- 最终产物是结构化分析结果或报告

## 什么情况不触发（正常短任务）：
- 用户只是在检索/查找/浏览专利
- 只查看单个专利的详情
- 简单问答（"这个专利的申请人是谁"）
- 用户明确说不需要分析

## 对话历史
{conversation_history}

## 当前消息
{current_query}

## 候选 knowledge
{candidates with type labels}

请选择最匹配的 knowledge_id，不触发长任务返回 null。
```

### 7.5 意图识别后的处理

`create_agent()` 的返回值扩展：
- 正常 agent → 返回 LangChain agent 对象，继续流式 LLM 路径
- 长任务 → 返回 `{"intent": "long_task", "knowledge": {...}}`

`run_pipeline()` 检测到长任务标记后，不走 `invoke_agent()` 流式路径，而是提交 Celery 并通过 SSE 推送 `long_task_created`。

### 7.6 边界处理

| 场景 | 处理 |
|------|------|
| 用户表达分析意图但未明确专利列表 | LLM 路由判断：若对话历史有检索结果，触发长任务；否则走普通 agent 让用户澄清 |
| 用户意图模糊 | LLM 路由返回 null → 正常短任务 |
| 用户明确说"不用分析，列出就行" | LLM 路由匹配 type=1 检索类 knowledge → 不触发 |

---

## 八、模型方案

### 8.1 多 Provider 实例

Worker 启动时根据 `config.ini` 配置创建 Provider 实例：

```ini
# config.ini
[LONG_TASK]
provider_family = deepseek        # deepseek | minimax
max_patents = 20                  # 单次分析最大专利数，超出截断
```

```
provider_family = deepseek:
  ├── flash_provider = Provider("deepseek", model="deepseek-chat")    # V4-Flash
  └── pro_provider   = Provider("deepseek", model="deepseek-reasoner") # V4-Pro

provider_family = minimax:
  ├── flash_provider = Provider("minimax", model="minimax-2.7-highspeed")
  └── pro_provider   = Provider("minimax", model="minimax-2.7-highspeed")
```

切换方式：修改 config.ini → 重启 Worker 生效。对用户透明。

### 8.2 模型分工

| 阶段 | 模型 | 说明 |
|------|------|------|
| Phase 1: 动态列定义 | Flash | 根据用户问题生成列，1 次调用 |
| Phase 2a: 下载专利文件 | — | 通过 DI 平台 / USPTO API 下载，无 LLM 调用 |
| Phase 2b: 单专利分析 | Pro | 每个专利 1 次调用，按列分析 |
| Phase 2c: 单专利总结 | Pro | 每个专利 1 次调用，生成简短总结 |
| Phase 3a: 生成报告大纲 | Pro | 根据用户问题 + 完整表格动态决定报告结构 |
| Phase 3b: 逐章节撰写 | Pro | 每章 1 次调用，按大纲顺序 |
| Phase 4: 导出 | — | python-docx / weasyprint |

---

## 九、报告生成

### 9.1 报告结构：LLM 动态生成

报告结构**不写死**，由 LLM 在 Phase 3 根据用户的问题和表格分析结果动态决定。

```
Phase 3 内部拆为两步：

Step 3a: 生成报告大纲（Pro）
  输入：
    - 用户原始问题："分析技术分布和创新趋势，重点关注华为的布局"
    - Phase 1 生成的列定义
    - Phase 2 生成的完整表格 + 各专利简单结论

  LLM 输出（JSON）：
  {
    "title": "深度学习图像识别专利技术分析报告",
    "sections": [
      {"heading": "一、分析概览",    "description": "专利集统计画像，技术领域分布总览"},
      {"heading": "二、技术分布对比", "description": "基于对比表格，解读各技术方向的专利布局差异"},
      {"heading": "三、华为技术布局", "description": "聚焦华为的 8 件专利，分析其技术策略和重点方向"},
      {"heading": "四、创新趋势分析", "description": "关键技术演进趋势、新兴方向识别"},
      {"heading": "五、机会与建议",   "description": "技术空白点、潜在研发方向、风险提示"}
    ]
  }

Step 3b: 按大纲逐节撰写（Pro）
  for section in outline.sections:
      撰写该章节内容
  → 输出完整报告文本
```

**示例：同一个专利集，不同问题 → 不同报告结构**

| 用户问题 | LLM 生成的报告结构 |
|---------|------------------|
| "分析技术分布和创新点" | 概览 → 技术领域分布 → 各领域代表专利 → 创新点总结 |
| "对比华为和腾讯的专利策略" | 概览 → 华为布局分析 → 腾讯布局分析 → 策略对比 → 差异化建议 |
| "评估侵权风险" | 概览 → 我的产品特征 → 逐专利风险对照 → 风险等级总览 → 规避建议 |
| "做一份完整分析报告" | 概览 → 技术分布 → 申请人分析 → 创新趋势 → 技术空白 → 核心专利 → 建议 |

### 9.2 报告存储（可替换后端）

报告文件不直接写本地文件系统，而是通过一个**存储抽象层**写入。MVP 默认用本地文件系统，未来切对象存储只需改配置。

```python
# sources/long_task/storage.py — 存储抽象

class ReportStorage(ABC):
    """报告文件存储抽象接口"""
    @abstractmethod
    async def put(self, task_id: str, filename: str, content: bytes) -> str:
        """存储文件，返回访问路径"""
        ...

    @abstractmethod
    async def get(self, task_id: str, filename: str) -> bytes:
        """读取文件内容"""
        ...

    @abstractmethod
    async def delete(self, task_id: str) -> None:
        """删除某个任务的全部报告"""
        ...


class LocalReportStorage(ReportStorage):
    """MVP：本地文件系统"""
    def __init__(self, base_dir: str = "/opt/workspace/reports"):
        self.base_dir = base_dir

    async def put(self, task_id, filename, content):
        path = os.path.join(self.base_dir, task_id, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        async with aiofiles.open(path, 'wb') as f:
            await f.write(content)
        return path

    async def get(self, task_id, filename):
        path = os.path.join(self.base_dir, task_id, filename)
        async with aiofiles.open(path, 'rb') as f:
            return await f.read()

    ...

class S3ReportStorage(ReportStorage):
    """未来扩展：S3 兼容对象存储（MinIO / AWS S3 / 阿里云 OSS）"""
    ...
```

**配置切换：**
```ini
# config.ini
[STORAGE]
backend = local                    # local | s3
local_base_dir = /opt/workspace/reports
# s3_endpoint = https://s3.example.com
# s3_bucket = copiioai-reports
# s3_access_key = ...
```

Worker 导出文件时调用 `storage.put(task_id, filename, content)`，API 下载时调用 `storage.get(task_id, filename)`。上层代码不感知底层存储。

| 存储后端 | 场景 |
|---------|------|
| `local` | MVP 单机，当前 2C2G |
| `s3` | 分布式 / 多副本 / Worker 和 API 不在同一台机器时 |

### 9.3 导出方案

| 格式 | 工具 | 方式 |
|------|------|------|
| PDF | `weasyprint` 或 `wkhtmltopdf` | HTML 模板 → PDF |
| Word | `python-docx` | 程序化构建，表格 + 段落 |

导出完成后通过 `storage.put()` 写入，后续下载走 `storage.get()`。

---

## 十、前端交互设计

参考 Manus 的交互模式：

### 10.1 交互流程

```
1. 用户在对话页正常输入 "分析这些专利的技术分布和创新点" → 点击发送
   （前端无需区分是否长任务，跟普通消息一样走现有对话接口）

2. 前端 POST /query_stream（现有 SSE 流式接口）
   ← SSE 流开始返回，跟正常对话一样的 token 流...

3. SSE 流中返回特殊事件：
   ← event: long_task_created
     data: {"task_id": "lt_...", "session_id": "sess_..."}

4. 前端检测到 long_task_created 事件 → 切换 UI：
   ┌──────────────────────────────────────────┐
   │ 🔄 批量专利分析  task_id: lt_...          │
   │                                          │
   │ 阶段: ████████░░░░░░░░░░ 45%             │
   │                                          │
   │ ✓ 生成分析框架                            │
   │ ✓ 确定分析维度：技术领域 | 核心方案 | 创新点  │
   │                                          │
   │ ⟳ 逐专利处理（9/20）                      │
   │   ✓ CN...234A 下载✓ 分析✓ 总结✓           │
   │   ✓ CN...235A 下载✓ 分析✓ 总结✓           │
   │   ⟳ CN...239A 下载✓ 分析中...             │
   │   ○ CN...240A                            │
   │   ...                                    │
   │                                          │
   │ ○ 生成报告                               │
   │ ○ 导出文件                               │
   └──────────────────────────────────────────┘

5. 用户可关闭网页，回来后打开 session 继续查看

6. 任务完成 → 前端展示报告预览 + 下载按钮 [PDF] [Word]
```

### 10.2 SSE 事件约定

在现有 SSE 事件类型（`token`, `status`, `artifact_*`）基础上，新增一种：

| 事件类型 | 触发时机 | data 格式 |
|---------|---------|----------|
| `long_task_created` | 后端识别为长任务后，SSE 流中推送 | `{"task_id": "lt_...", "session_id": "sess_..."}` |

前端需在 SSE 流监听中增加对 `long_task_created` 的处理：
```javascript
eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'long_task_created') {
    // 停止 SSE，切换到轮询
    eventSource.close();
    startPolling(data.task_id, data.session_id);
  }
  // ... 其他事件处理保持不变
};
```

### 10.3 轮询策略

```javascript
// 前端轮询逻辑
let status = 'running';
const pollInterval = 2000; // 2 秒

while (status === 'running') {
  const res = await fetch(`/long_task/${taskId}/status`);
  const data = await res.json();
  status = data.status;
  renderProgress(data);
  if (status === 'running') {
    await sleep(pollInterval);
  }
}
// status === 'completed' 或 'failed'
```

### 10.3 状态展示

- 每个步骤有图标：✓（完成）、⟳（进行中，带动画）、○（等待）、✗（失败）
- 当前步骤有高亮和详细描述
- 表格行实时追加，新行有进入动画
- 完成后，报告文本支持 Markdown 渲染预览

---

## 十一、部署架构

### 11.1 MVP 阶段：单机部署

```
US 服务器 2C2G
├── FastAPI（Uvicorn, 端口 7777）
│   ├── 短任务 Agent Pool（max=3 改为 3，原为 10）
│   └── SSE 流式响应
├── Celery Worker（pool=solo, concurrency=1）
│   └── 长任务执行
├── 外部 Redis（已有）
├── 外部 MySQL / AWS RDS（已有）
└── DeepSeek / MiniMax API（中国，跨境调用）
```

### 11.2 内存预算

| 组件 | 内存（MB） |
|------|-----------|
| OS + 基础服务 | ~400 |
| FastAPI + Agent Pool | ~300-500 |
| Celery Worker（solo） | ~300-500 |
| 系统缓冲 | ~500-1000 |
| **总计（2GB 上限）** | **~1500-2400** |

**缓解措施：**
- Agent pool 从 10 降到 3
- 配置 2GB swap
- 专利文件流式处理，不全量加载
- Celery solo pool（不 fork 子进程）

### 11.3 后续扩展路径

```
Phase 2（需要时）: 升配至 4C4G
Phase 3（需要时）: Worker 拆到国内服务器
  └── 理由：专利下载延迟低、中国 API 调用更稳定、数据合规
```

---

## 十二、风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| 2C2G OOM | 进程被 Kill，任务丢失 | swap 2GB + Celery ack_late + 任务重试 |
| 中国 API 跨境网络不稳定 | LLM 调用失败 | Celery 重试 3 次 + 指数退避 |
| 单个专利分析时间过长 | 20 专利超预期等待 | 单专利超时 60s，超时跳过 |
| DI 平台 token 过期 | 专利下载失败 | `ensure_valid_access_token()` 提前刷新（已有） |
| Worker 进程崩溃 | 当前任务丢失 | Celery ack_late 机制，任务重新入队 |
| 同时多个长任务 | 资源抢占 | concurrency=1 串行执行，后续任务排队 |

---

## 十三、需要新增/修改的文件

### 新增文件

| 文件 | 用途 |
|------|------|
| `api_routes/long_task.py` | 长任务 API 路由 |
| `api_routes/session.py` | Session API 路由 |
| `celery_worker.py` | Celery Worker 入口 |
| `sources/long_task/__init__.py` | 长任务模块 |
| `sources/long_task/patent_analyzer.py` | 专利分析核心逻辑 |
| `sources/long_task/report_generator.py` | 报告生成（Word/PDF） |
| `sources/long_task/storage.py` | 报告存储抽象（本地 / S3） |
| `sources/long_task/status_manager.py` | Redis 状态管理 |
| `mysql/init/add_conversations.sql` | conversations 表 DDL |
| `mysql/init/add_long_tasks.sql` | long_tasks 表 DDL |

### 修改文件

| 文件 | 变更 |
|------|------|
| `api.py` | 注册 long_task 和 session 路由 |
| `sources/knowledge/type_utils.py` | 增加 type=3 识别 |
| `sources/knowledge/selection.py` | 增强路由 prompt（跨轮上下文） |
| `sources/agents/general_agent.py` | 长任务入口分支处理 |
| `api_routes/core.py` | Agent pool 从 10 降到 3 |
| `config.ini` | 新增 `[LONG_TASK]` 配置段 |
| `config.ini.example` | 新增配置示例 |
| `requirements.txt` | 新增 celery, python-docx, weasyprint |

---

## 十四、开发阶段

### Phase 1: 基础设施（第 1-2 周）

- [ ] 数据库：conversations 表、long_tasks 表
- [ ] Celery Worker 启动和配置
- [ ] Redis 状态管理模块
- [ ] API 路由：`/long_task/execute`, `/{task_id}/status`
- [ ] API 路由：session CRUD
- [ ] Worker 能接收任务并写入状态

### Phase 2: 分析流程（第 3-4 周）

- [ ] Phase 1: Flash 动态生成表格列定义
- [ ] Phase 2: Pro 串行逐专利分析 + 填表
- [ ] Phase 3a: Pro 动态生成报告大纲
- [ ] Phase 3b: Pro 按大纲逐章节撰写
- [ ] Phase 4: 导出 Word + PDF
- [ ] 意图识别：type=3 knowledge + conversation_history 上下文
- [ ] SSE 新增 `long_task_created` 事件推送

### Phase 3: 前端联调（第 5-6 周）

- [ ] SSE 流中 `long_task_created` 事件监听
- [ ] 收到事件后切换到轮询视图
- [ ] 轮询状态展示（进度条 + 步骤列表 + 表格逐行实时更新）
- [ ] 报告大纲预览 + 全文 Markdown 渲染 + 下载按钮
- [ ] Session 列表 + 恢复
- [ ] 完整端到端测试

---

## 十五、附录：设计决策记录

| # | 决策 | 原因 |
|---|------|------|
| 1 | Knowledge type=3 + Scene 容器 | 最小改动，复用 embedding 检索 + LLM 路由 |
| 2 | MySQL conversations 表 | 可靠、持久、可查询 |
| 3 | Session 懒创建 | 不浪费存储，只有长任务触发时才建 |
| 4 | 后端意图识别，前端通过 SSE 通知 | 前端不感知长任务判断；用户走现有 `/query_stream` 发消息，后端 LLM 路由识别后 SSE 推送 `long_task_created` 事件 |
| 5 | Celery solo worker | 进程级任务隔离，连接断了任务不丢 |
| 6 | 多 Provider 实例，config.ini 手动切换 | 后端重启切换 DeepSeek/MiniMax，对用户透明 |
| 7 | 长任务状态走轮询（非 SSE） | 用户关闭网页后可恢复查看 |
| 8 | 全流程串行 | 保证结果顺序和一致性 |
| 9 | 动态列定义：表格列由 Flash 根据用户问题生成 | 灵活适配不同分析问题 |
| 10 | 动态报告结构：大纲由 Pro 根据用户问题 + 表格内容生成 | 不同问题 → 不同报告结构，不是固定模板 |
| 11 | 2C2G 单机 MVP | 先验证流程，数据驱动扩容 |
| 12 | max_patents 在 config.ini 中配置 | 修改重启即生效，不需要改代码 |
| 13 | 报告持久化 + 存储抽象层 | MVP 用本地文件系统，通过 `ReportStorage` 接口隔离；未来加一行配置切 S3，上层代码不动 |
| 14 | 报告可反复下载 | task_id 持久有效，关页重开、换设备都能下载 |
| 15 | 输入去重 + 排序 | `sorted(set(patent_ids))`，同一专利不分析两次，顺序确定 |
| 16 | Redis checkpoint 断点续算 | 每个专利完成后立即写 checkpoint；Worker 崩溃后从 pending 继续，已完成的不重复 |
| 17 | 单专利超时不阻塞 | 超时 60s 标记 failed，表格显示"分析失败"，继续下一个 |
