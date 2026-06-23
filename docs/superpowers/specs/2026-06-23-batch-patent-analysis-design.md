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
│  [上一轮检索结果] + [用户输入"分析这50个专利"]                │
│  └──→ POST /long_task/execute                              │
│       {query, patent_ids, conversation_history}             │
│                                                             │
│  ←── {task_id} 立即返回                                      │
│                                                             │
│  轮询 GET /long_task/{task_id}/status                       │
│  ←── {phase, step, progress, table_rows, cards}             │
│                                                             │
│  ... 用户关闭网页，回来，继续轮询 ...                          │
│                                                             │
│  任务完成 → GET /long_task/{task_id}/report → 下载           │
└─────────────────────────────────────────────────────────────┘

                         │
                         ▼

┌─────────────────────────────────────────────────────────────┐
│                  FastAPI（US 服务器）                         │
│                                                             │
│  POST /long_task/execute                                    │
│  ├── 意图识别：匹配 type=3 knowledge → 确认是长任务          │
│  ├── 创建 session（MYSQL conversations 表）                  │
│  ├── 写入前端带来的对话历史                                   │
│  ├── 生成 task_id                                           │
│  ├── 提交 Celery 任务（Redis broker）                        │
│  └── 返回 task_id                                           │
│                                                             │
│  GET /long_task/{task_id}/status                            │
│  └── 读 Redis → 返回当前状态、步骤列表、表格行、卡片          │
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
│           ├── state → Redis: {phase:1, progress:0%}         │
│           └── 输出: {columns: [...], patent_count: N}       │
│                                                             │
│  Phase 2: 逐专利串行分析（Pro）                              │
│           for patent in patents:                            │
│               ├── 分析单个专利                                │
│               ├── 填入表格行                                  │
│               ├── 生成简单结论                                │
│               └── 写入 Redis → 前端轮询可见                   │
│                                                             │
│  Phase 3: 汇总生成报告（Pro）                                │
│           ├── 读取完整表格 + 所有结论                          │
│           ├── LLM 综合分析                                   │
│           └── 输出报告文本                                    │
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
    "max_patents": 20,
    "output_formats": ["pdf", "docx"]
  }
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

### 5.1 触发长任务

```
POST /long_task/execute
Authorization: Bearer {firebase_token}
```

**Request:**
```json
{
  "query": "分析这些专利的技术分布、创新点和竞争风险",
  "patent_ids": ["CN202310001234A", "CN202310001235A", ...],
  "conversation_history": [
    {"role": "user", "content": "用关键词'深度学习图像识别'检索专利"},
    {"role": "assistant", "content": "检索到50个相关专利...", "patent_data": [...]}
  ],
  "scene_id": 1
}
```

**Response (200):**
```json
{
  "success": true,
  "task_id": "lt_20260623_a1b2c3d4",
  "session_id": "sess_20260623_e5f6g7h8",
  "message": "长任务已提交，正在准备分析"
}
```

**后端逻辑：**
1. 验证 auth token
2. LLM 意图识别：查询匹配 type=3 knowledge，确认是长任务
3. 检查 usage limit
4. 如果用户此时没有活跃 session，创建新 session
5. 将 `conversation_history` 写入 session
6. 创建 `long_tasks` 记录（status=pending）
7. `long_task_execute.delay(task_id, params)` 提交 Celery
8. 返回 `task_id` + `session_id`

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

### 5.3 获取报告文件

```
GET /long_task/{task_id}/report?format=pdf
GET /long_task/{task_id}/report?format=docx
```

**Response:** 文件下载（`Content-Disposition: attachment`）

### 5.4 Session 相关 API

```
GET /session/{session_id}              → 获取 session 详情和对话历史
GET /sessions?user_id={uid}            → 获取用户所有 session 列表
POST /session/{session_id}/message    → 追加消息到 session（短任务对话时）
DELETE /session/{session_id}           → 归档 session
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

# 启动命令:
# celery -A celery_worker worker --pool=solo --concurrency=1 --loglevel=info
```

### 6.2 任务编排（全流程串行）

```python
@app.task(bind=True, max_retries=3, default_retry_delay=30)
def execute_patent_analysis(self, task_id: str, params: dict):
    """批量专利分析长任务 — 全流程串行"""
    try:
        # === Phase 1: 生成表格列定义 ===
        update_status(task_id, 'generating_columns', 0,
                     '正在生成分析框架...')
        columns = generate_table_columns(
            query=params['query'],
            patent_count=len(params['patent_ids']),
            model=get_flash_model(params['model_family'])
        )
        update_columns(task_id, columns)
        update_progress(task_id, 5)

        # === Phase 2: 串行逐专利分析 ===
        patents = params['patent_ids']
        table_rows = []
        for i, patent_id in enumerate(patents):
            row = analyze_single_patent(
                patent_id=patent_id,
                columns=columns,
                query=params['query'],
                model=get_pro_model(params['model_family'])
            )
            table_rows.append(row)
            table_rows_sorted = table_rows.copy()  # 保持顺序

            # 进度: 5% ~ 75%，按专利数量均分
            progress = 5 + int((i + 1) / len(patents) * 70)
            update_status(task_id, 'analyzing', progress,
                         f'正在分析第 {i+1}/{len(patents)} 个专利（{patent_id}）',
                         table_rows=table_rows_sorted)
            write_message_to_session(params['session_id'], ...)

        # === Phase 3: 汇总生成报告 ===
        update_status(task_id, 'generating_report', 80,
                     '正在基于完整分析结果生成报告...')
        report = generate_report(
            query=params['query'],
            columns=columns,
            table_rows=table_rows,
            model=get_pro_model(params['model_family'])
        )
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
```

**TTL 策略：** Redis 数据设 24 小时 TTL。MySQL 为 source of truth。

---

## 七、意图识别设计

### 7.1 长任务触发条件

| 触发场景 | 识别方式 |
|---------|---------|
| 用户直接说"用 XX 关键字找专利并分析" | LLM 路由匹配 type=3 knowledge |
| 用户上传了一批专利文件 | 前端标记 `intent=long_task` |
| 用户上一轮检索了专利，这轮说"分析这些" | LLM 路由时传入上一轮上下文（前端携带），LLM 判断用户意图是长任务 |

### 7.2 LLM 路由 Prompt 增强

在 `choose_knowledge_candidate()` 的 prompt 中增加以下上下文：

```
## 上一轮对话
用户: 用"深度学习"检索了 50 个中国专利
助手: 已检索到 50 个专利，以下是列表...

## 当前轮
用户: 分析这些专利的技术分布

## 判断
如果当前轮的用户意图是对上一轮检索结果做分析，应该选择 type=3（长任务）的知识入口。
```

---

## 八、模型方案

### 8.1 多 Provider 实例

Worker 启动时根据 `config.ini` 配置创建 Provider 实例：

```ini
# config.ini
[LONG_TASK]
provider_family = deepseek        # deepseek | minimax
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
| Phase 1: 生成列定义 | Flash | 快速分类，一次调用 |
| Phase 2: 逐专利分析 | Pro | 每个专利一次深度调用 |
| Phase 3: 汇总报告 | Pro | 综合所有分析结果 |
| Phase 4: 导出 | — | python-docx / wkhtmltopdf |

---

## 九、报告生成

### 9.1 报告结构

```
1. 分析概览
   - 专利集统计（数量、时间范围、IPC 分布）
   - 分析问题与方法说明

2. 对比分析
   嵌入完整对比表格（Phase 1-2 产物）
   关键差异解读（Phase 3 产物）

3. 重点专利深度解读
   Phase 3 挑选 3-5 个最关键专利进行 300-500 字解读

4. 技术趋势与洞察
   技术演进方向、竞争格局、创新空白

5. 建议与行动方向
   研发建议、专利申请建议、风险提示

附录：全量专利对比表格
```

### 9.2 导出方案

| 格式 | 工具 | 方式 |
|------|------|------|
| PDF | `weasyprint` 或 `wkhtmltopdf` | HTML 模板 → PDF |
| Word | `python-docx` | 程序化构建，表格 + 段落 |

报告文件存储在本机文件系统 `/opt/workspace/reports/{task_id}/`，前端通过下载 API 获取。

---

## 十、前端交互设计

参考 Manus 的交互模式：

### 10.1 交互流程

```
1. 用户在对话页检索专利 → 看到结果列表（正常短任务流程）

2. 用户输入"分析这些专利的技术分布和创新点" → 点击发送

3. 前端发送 POST /long_task/execute
   ← 返回 {task_id, session_id}

4. 前端进入"长任务视图"：
   ┌──────────────────────────────────────────┐
   │ 🔄 批量专利分析                           │
   │                                          │
   │ 阶段: ████████░░░░░░░░░░ 45%             │
   │                                          │
   │ ✓ 生成分析框架                            │
   │ ✓ 确定分析维度：技术领域 | 核心方案 | 创新点  │
   │                                          │
   │ ⟳ 逐专利分析（9/20）                      │
   │   ✓ CN202310001234A - 已完成              │
   │   ✓ CN202310001235A - 已完成              │
   │   ⟳ CN202310001239A - 正在提取核心技术... │
   │   ○ CN202310001240A                      │
   │   ...                                    │
   │                                          │
   │ ○ 生成报告                               │
   │ ○ 导出文件                               │
   └──────────────────────────────────────────┘

5. 用户可关闭网页，回来后打开 session 继续查看

6. 任务完成 → 前端展示报告预览 + 下载按钮 [PDF] [Word]
```

### 10.2 轮询策略

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

- [ ] Phase 1: Flash 生成表格列定义
- [ ] Phase 2: Pro 串行逐专利分析 + 填表
- [ ] Phase 3: Pro 汇总生成报告
- [ ] Phase 4: 导出 Word + PDF
- [ ] 意图识别：type=3 knowledge + 跨轮上下文

### Phase 3: 前端联调（第 5-6 周）

- [ ] 前端长任务触发逻辑
- [ ] 轮询状态展示（进度条 + 步骤列表 + 表格实时更新）
- [ ] 报告预览 + 下载
- [ ] Session 恢复
- [ ] 完整端到端测试

---

## 十五、附录：设计决策记录

| # | 决策 | 原因 |
|---|------|------|
| 1 | Knowledge type=3 + Scene 容器 | 最小改动，复用 embedding 检索 + LLM 路由 |
| 2 | MySQL conversations 表 | 可靠、持久、可查询 |
| 3 | Session 懒创建 | 不浪费存储，只有长任务触发时才建 |
| 4 | 前端携带跨轮上下文 | 后端无状态，不维护短对话缓存 |
| 5 | Celery solo worker | 进程级任务隔离，连接断了任务不丢 |
| 6 | 多 Provider 实例 | 显式切换 DeepSeek/MiniMax，改动小 |
| 7 | 轮询（非 SSE） | 用户关闭网页后可恢复查看 |
| 8 | 全流程串行 | 保证结果顺序和一致性 |
| 9 | 动态列定义 + 逐行填充 | 灵活适配不同分析问题，前端可视化渐进 |
| 10 | 2C2G 单机 MVP | 先验证流程，数据驱动扩容 |
