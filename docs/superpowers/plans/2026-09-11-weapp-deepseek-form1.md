# 小程序形态一（DeepSeek 式交互）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把小程序从「会话列表页 → 对话页」两页结构改成 DeepSeek App 形态一（首页即对话页 + 左上角 ☰ 抽屉收历史，抽屉内可删除/重命名），同时补全后端会话端点的鉴权与归属校验。

**Architecture:** 后端先动（前端依赖新端点）。`api_routes/session.py` 的 5 个端点补 `verify_firebase_token` + `user_id` 归属过滤，新增专用 `PUT /session/{id}/title`。前端把 `pages/chat/index` 提升为首页，新增三个纯展示组件（`NavBar` / `SessionDrawer` / `RenameModal`），chat 页持有全部状态与网络调用，最后删除 `pages/index/`。

**Tech Stack:** Python 3.14 + FastAPI + pytest（后端）／Taro 4.1 + React 18 + TypeScript + SCSS（小程序）

**依据 spec:** `docs/superpowers/specs/2026-09-10-weapp-deepseek-ux-design.md`

---

## Global Constraints

- 分支：`feat/weapp`（当前分支，先确认 `git branch --show-current` 输出 `feat/weapp`）
- **后端测试命令**（本机缺 `REDIS_PORT` 时 `passport.py:16` 会在 import 期炸，故必须带上两个环境变量）：

  ```bash
  cd E:/online/workspace/copiioai/langsistance && \
    REDIS_HOST=127.0.0.1 REDIS_PORT=6379 PYTHONUTF8=1 \
    python -m pytest tests/test_session_api.py -v
  ```

- **前端验证命令**（本项目无测试框架，tsc + 构建即验证口径）：

  ```bash
  cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc && npm run build:weapp
  ```

- `frontend/weapp/node_modules` 易丢（分支切换/误删），丢了先 `npm install`（约 47s）
- 归属校验失败统一返回 **404**，不用 403 —— 不暴露"会话存在但不属于你"
- `conversations.title` 是 `VARCHAR(256)`（`mysql/init/add_conversations.sql:9`）；客户端沿用既有的 60 字上限（`services/chat.ts:26` 同款）
- 不要提交 `firebase_service_key.json`（`.gitignore:25` 已忽略，本地那份是为跑测试放的一次性假凭据）
- 提交信息用 conventional commits，**不加 `Co-Authored-By`**（本仓库 200 条历史里 0 条带，用户全局规则也写明 attribution disabled）

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `api_routes/session.py` | 改：5 端点补鉴权/归属 + 新增 `PUT /title` |
| `tests/test_session_api.py` | 改：auth fixture + 归属/401/重命名测试 |
| `frontend/weapp/src/services/sessions.ts` | 改：加 `renameSession` / `archiveSession` |
| `frontend/weapp/src/components/NavBar/` | 新：自绘顶栏（状态栏高度 + 胶囊避让） |
| `frontend/weapp/src/components/RenameModal/` | 新：重命名弹窗（平台无原生输入弹窗，必须自绘） |
| `frontend/weapp/src/components/SessionDrawer/` | 新：会话抽屉（纯展示，只发意图） |
| `frontend/weapp/src/pages/chat/index.tsx` | 改：首页化 + 挂三个组件 + 持有状态与网络 |
| `frontend/weapp/src/pages/chat/index.config.ts` | 改：加 `navigationStyle: 'custom'` |
| `frontend/weapp/src/app.config.ts` | 改：chat 置首页，去掉 index |
| `frontend/weapp/src/pages/login/index.tsx` | 改：登录兜底 reLaunch 目标改 chat |
| `frontend/weapp/src/pages/index/` | 删：三文件 |

---

## Task 1: 后端测试基建（auth fixture + 修既有红灯）

当前基线：`tests/test_session_api.py` **5 passed, 2 failed**。两个红灯是 `test_create_session` / `test_get_user_sessions` —— 这两个端点**已经有**鉴权了，但测试没带 token。本任务把测试基建补齐，让基线回到全绿，后续任务才能拿到干净的 RED 信号。

**Files:**
- Modify: `tests/test_session_api.py`

**Interfaces:**
- Consumes: 无
- Produces: `auth` fixture（patch `api_routes.session.verify_firebase_token`）、模块常量 `AUTH`（测试用请求头）、`TEST_UID = 123`

- [ ] **Step 1: 确认基线是 5 passed / 2 failed**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  REDIS_HOST=127.0.0.1 REDIS_PORT=6379 PYTHONUTF8=1 \
  python -m pytest tests/test_session_api.py -v
```

Expected: `2 failed, 5 passed`，两个 FAILED 是 `test_create_session` 与 `test_get_user_sessions`，均为 `assert 401 == 200`。

- [ ] **Step 2: 加 auth fixture 与请求头常量**

在 `tests/test_session_api.py` 顶部 import 区补 `HTTPException`，并在 `mock_db` fixture 之后插入：

```python
from fastapi import HTTPException

# ── 鉴权替身 ──────────────────────────────────────────────────
# session.py 里 verify_firebase_token 是模块级 import，patch 目标必须是
# api_routes.session.verify_firebase_token。替身行为对齐真实实现
# （sources/user/passport.py:36-37）：无 Bearer 头抛 401。
TEST_UID = 123
AUTH = {'Authorization': 'Bearer test-token'}


def _fake_verify(auth_header, uid=TEST_UID):
    if not auth_header or not auth_header.startswith('Bearer '):
        raise HTTPException(status_code=401, detail='Missing token')
    return {'uid': uid}


@pytest.fixture
def auth():
    """把 verify_firebase_token 换成仿真替身，让测试同时覆盖 401 与已鉴权路径。"""
    with patch('api_routes.session.verify_firebase_token',
               side_effect=_fake_verify):
        yield TEST_UID
```

- [ ] **Step 3: 让 `client` fixture 依赖 `auth`**

把现有的 `client` fixture 签名从 `def client(mock_db):` 改成：

```python
@pytest.fixture
def client(mock_db, auth):
```

（函数体不变。）

- [ ] **Step 4: 给 7 个既有测试补上请求头**

对 `tests/test_session_api.py` 做以下 7 处修改（`client.请求(...)` 调用一律补 `headers=AUTH`）：

```diff
 def test_get_session_not_found(client, mock_db):
-    response = client.get("/session/nonexistent")
+    response = client.get("/session/nonexistent", headers=AUTH)
     assert response.status_code == 404

 def test_create_session(client, mock_db):
     response = client.post("/session", json={
         "user_id": 123,
         "messages": [{"role": "user", "content": "hello"}],
-    })
+    }, headers=AUTH)
     assert response.status_code == 200

 def test_get_user_sessions(client, mock_db):
-    response = client.get("/sessions?user_id=123")
+    response = client.get("/sessions?user_id=123", headers=AUTH)
     assert response.status_code == 200

 def test_append_message(client, mock_db):
     response = client.post("/session/sess_001/message", json={
         "role": "assistant",
         "content": "hi there",
-    })
+    }, headers=AUTH)
     assert response.status_code == 200

 def test_append_message_session_not_found(client, mock_db):
     response = client.post("/session/nonexistent/message", json={
         "role": "assistant",
         "content": "hi",
-    })
+    }, headers=AUTH)
     assert response.status_code == 404

 def test_archive_session(client, mock_db):
-    response = client.delete("/session/sess_001")
+    response = client.delete("/session/sess_001", headers=AUTH)
     assert response.status_code == 200

 def test_archive_session_not_found(client, mock_db):
-    response = client.delete("/session/nonexistent")
+    response = client.delete("/session/nonexistent", headers=AUTH)
     assert response.status_code == 404
```

- [ ] **Step 5: 跑测试确认全绿**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  REDIS_HOST=127.0.0.1 REDIS_PORT=6379 PYTHONUTF8=1 \
  python -m pytest tests/test_session_api.py -v
```

Expected: `7 passed`。

- [ ] **Step 6: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add tests/test_session_api.py && \
  git commit -m "test: 会话 API 测试补 auth fixture——修既有 2 个红灯(create/list 端点早已要求鉴权)"
```

---

## Task 2: 5 处会话端点补鉴权与归属校验

修 spec §1.2 的 IDOR。`PUT /messages` 那处注释声称校验归属但 SQL 里没有 `user_id`（`api_routes/session.py:190` vs `:192`），本任务一并修掉。

**Files:**
- Modify: `api_routes/session.py`
- Test: `tests/test_session_api.py`

**Interfaces:**
- Consumes: Task 1 的 `auth` fixture、`AUTH`、`TEST_UID`
- Produces: 无新签名；5 个既有端点的行为变为「缺 token → 401，非本人会话 → 404」

- [ ] **Step 1: 写失败测试**

在 `tests/test_session_api.py` 末尾追加：

```python
def _executed_sql(cursor):
    """把 cursor.execute 收到的 SQL 文本拼起来，供"归属过滤真的写进 SQL 了吗"断言用。"""
    return ' '.join(
        str(call.args[0]) for call in cursor.execute.call_args_list if call.args
    )


# 5 个本任务补鉴权的端点（方法, 路径, 合法 body）
# body 必须合法，否则 FastAPI 会在进 handler 前返回 422，测不到 401。
PROTECTED = [
    ('get', '/session/sess_001', None),
    ('get', '/session-by-id?session_id=sess_001', None),
    ('post', '/session/sess_001/message', {'role': 'user', 'content': 'x'}),
    ('put', '/session/sess_001/messages', {'messages': [], 'title': ''}),
    ('delete', '/session/sess_001', None),
]


@pytest.mark.parametrize('method,path,body', PROTECTED)
def test_endpoints_require_auth(client, method, path, body):
    """缺 Authorization 一律 401。"""
    kwargs = {'json': body} if body is not None else {}
    response = getattr(client, method)(path, **kwargs)
    assert response.status_code == 401


@pytest.mark.parametrize('method,path,body', PROTECTED)
def test_endpoints_filter_by_owner(client, mock_db, method, path, body):
    """SQL 必须真的带 user_id 过滤；查不中 → 404。

    这条断言存在的理由：PUT /messages 之前注释写着 "belongs to user" 但 SQL 里
    没有 user_id。只测状态码不够，必须测 SQL 文本。
    """
    _, cursor = mock_db
    cursor.fetchone.return_value = None
    cursor.rowcount = 0
    kwargs = {'json': body} if body is not None else {}
    response = getattr(client, method)(path, headers=AUTH, **kwargs)
    assert response.status_code == 404
    assert 'user_id' in _executed_sql(cursor)


def test_save_messages_keeps_title_behavior(client, mock_db):
    """回归：PUT /messages 带 title 时仍写 title 列（web 端在用，行为不能变）。"""
    _, cursor = mock_db
    cursor.fetchone.return_value = {'id': 1}

    response = client.put('/session/sess_001/messages',
                          json={'messages': [{'role': 'user', 'content': 'hi'}],
                                'title': '新标题'},
                          headers=AUTH)

    assert response.status_code == 200
    sql = _executed_sql(cursor).lower()
    assert 'messages = %s' in sql
    assert 'title = %s' in sql


def test_save_messages_omits_title_when_blank(client, mock_db):
    """回归：title 为空串时不动 title 列（既有语义）。"""
    _, cursor = mock_db
    cursor.fetchone.return_value = {'id': 1}

    response = client.put('/session/sess_001/messages',
                          json={'messages': [], 'title': ''},
                          headers=AUTH)

    assert response.status_code == 200
    assert 'title = %s' not in _executed_sql(cursor).lower()
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  REDIS_HOST=127.0.0.1 REDIS_PORT=6379 PYTHONUTF8=1 \
  python -m pytest tests/test_session_api.py -v -k "require_auth or filter_by_owner"
```

Expected: `require_auth` 5 条全 FAIL（现在不校验 token，返回 200/404 而非 401）；`filter_by_owner` 5 条 FAIL 在 `assert 'user_id' in ...`。

- [ ] **Step 3: 实现——`GET /session-by-id`**

把 `api_routes/session.py` 的 `get_session_by_id` 整个函数替换为：

```python
    @router.get("/session-by-id")
    async def get_session_by_id(http_request: Request,
                                session_id: str = Query(..., min_length=1)):
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user['uid'])

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT session_id, title, status,
                              long_task_ids, messages, create_time, update_time
                       FROM conversations
                       WHERE session_id = %s AND user_id = %s AND status != 2""",
                    (session_id, user_id))
                row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Session not found")
            row['create_time'] = row['create_time'].isoformat() if row['create_time'] else None
            row['update_time'] = row['update_time'].isoformat() if row['update_time'] else None
            msgs = row.get('messages')
            if isinstance(msgs, str):
                try:
                    row['messages'] = json.loads(msgs)
                except (json.JSONDecodeError, TypeError):
                    row['messages'] = []
            lt = row.get('long_task_ids')
            if isinstance(lt, str):
                try:
                    row['long_task_ids'] = json.loads(lt)
                except (json.JSONDecodeError, TypeError):
                    row['long_task_ids'] = []
            elif lt is None:
                row['long_task_ids'] = []
            return {"success": True, **row}
        finally:
            conn.close()
```

- [ ] **Step 4: 实现——`GET /session/{session_id}`**

替换 `get_session` 函数为：

```python
    @router.get("/session/{session_id}")
    async def get_session(session_id: str, http_request: Request):
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user['uid'])

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id, session_id, user_id, scene_id, title,
                              messages, long_task_ids, status,
                              create_time, update_time
                       FROM conversations
                       WHERE session_id = %s AND user_id = %s AND status != 2""",
                    (session_id, user_id))
                row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Session not found")
            row['messages'] = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
            row['create_time'] = row['create_time'].isoformat() if row['create_time'] else None
            row['update_time'] = row['update_time'].isoformat() if row['update_time'] else None
            return {"success": True, "session": row}
        finally:
            conn.close()
```

- [ ] **Step 5: 实现——`POST /session/{session_id}/message`**

替换 `append_message` 函数为（注意：SQL 顺带补了 `status != 2`，已归档会话不再能追加消息）：

```python
    @router.post("/session/{session_id}/message")
    async def append_message(session_id: str, req: AppendMessageRequest,
                             http_request: Request):
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user['uid'])

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT messages FROM conversations
                       WHERE session_id = %s AND user_id = %s AND status != 2""",
                    (session_id, user_id))
                row = cur.fetchone()
                if row is None:
                    raise HTTPException(status_code=404, detail="Session not found")

                messages = json.loads(row['messages']) if isinstance(row['messages'], str) else row['messages']
                new_msg = {"role": req.role, "content": req.content}
                if req.patent_data:
                    new_msg["patent_data"] = req.patent_data
                if req.timestamp:
                    new_msg["timestamp"] = req.timestamp
                messages.append(new_msg)

                cur.execute(
                    """UPDATE conversations SET messages = %s
                       WHERE session_id = %s AND user_id = %s""",
                    (json.dumps(messages, ensure_ascii=False), session_id, user_id))
                conn.commit()
            logger.info(f"Message appended to session: {session_id}")
            return {"success": True}
        finally:
            conn.close()
```

- [ ] **Step 6: 实现——`PUT /session/{session_id}/messages`**

替换 `save_messages` 函数为（这是注释与实现不符的那处）：

```python
    @router.put("/session/{session_id}/messages")
    async def save_messages(session_id: str, req: SaveMessagesRequest,
                            http_request: Request):
        """Bulk-save all messages for a session (replace entire messages array)."""
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user['uid'])

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                # Verify session exists and belongs to user
                cur.execute(
                    """SELECT id FROM conversations
                       WHERE session_id = %s AND user_id = %s AND status != 2""",
                    (session_id, user_id))
                if cur.fetchone() is None:
                    raise HTTPException(status_code=404, detail="Session not found")
                updates = ["messages = %s"]
                params = [json.dumps(req.messages, ensure_ascii=False)]
                if req.title:
                    updates.append("title = %s")
                    params.append(req.title)
                params.extend([session_id, user_id])
                cur.execute(
                    f"UPDATE conversations SET {', '.join(updates)} "
                    f"WHERE session_id = %s AND user_id = %s",
                    params)
                conn.commit()
            logger.info(f"Session messages saved: {session_id}, count={len(req.messages)}")
            return {"success": True}
        finally:
            conn.close()
```

- [ ] **Step 7: 实现——`DELETE /session/{session_id}`**

替换 `archive_session` 函数为：

```python
    @router.delete("/session/{session_id}")
    async def archive_session(session_id: str, http_request: Request):
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user['uid'])

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE conversations SET status = 2
                       WHERE session_id = %s AND user_id = %s""",
                    (session_id, user_id))
                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Session not found")
                conn.commit()
            logger.info(f"Session archived: {session_id}")
            return {"success": True}
        finally:
            conn.close()
```

- [ ] **Step 8: 跑全量测试确认通过**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  REDIS_HOST=127.0.0.1 REDIS_PORT=6379 PYTHONUTF8=1 \
  python -m pytest tests/test_session_api.py -v
```

Expected: `19 passed`（Task 1 的 7 条 + 本任务 12 条：5 require_auth + 5 filter_by_owner + 2 条 save_messages 回归）。

- [ ] **Step 9: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add api_routes/session.py tests/test_session_api.py && \
  git commit -m "fix: 会话端点补鉴权+归属校验——修 IDOR 与 PUT /messages 注释实现不符"
```

---

## Task 3: 新增 `PUT /session/{session_id}/title`

专用改名端点。不复用 `PUT /messages`：后者会重写整个 messages 数组，流式对话进行中改名会与落库竞态丢消息。

**Files:**
- Modify: `api_routes/session.py`
- Test: `tests/test_session_api.py`

**Interfaces:**
- Consumes: Task 1 的 `auth` fixture / `AUTH`；Task 2 的 `_executed_sql` helper
- Produces: `PUT /session/{session_id}/title`，body `{"title": str}`，成功返回 `{"success": True}`

- [ ] **Step 1: 写失败测试**

在 `tests/test_session_api.py` 末尾追加：

```python
def test_rename_session(client, mock_db):
    """正常路径：改标题成功，且 SQL 带 user_id 归属过滤。"""
    _, cursor = mock_db
    cursor.rowcount = 1

    response = client.put('/session/sess_001/title',
                          json={'title': '折叠桌专利检索'},
                          headers=AUTH)

    assert response.status_code == 200
    assert response.json()['success'] is True
    sql = _executed_sql(cursor)
    assert 'user_id' in sql
    assert 'title' in sql


def test_rename_session_not_owner(client, mock_db):
    """非本人会话 → 404（不暴露"存在但不属于你"）。"""
    _, cursor = mock_db
    cursor.rowcount = 0

    response = client.put('/session/sess_001/title',
                          json={'title': 'x'}, headers=AUTH)

    assert response.status_code == 404


@pytest.mark.parametrize('bad_title', ['', '   '])
def test_rename_session_rejects_blank_title(client, mock_db, bad_title):
    """空标题在 SQL 之前就被拒，不进数据库。"""
    _, cursor = mock_db
    cursor.rowcount = 1

    response = client.put('/session/sess_001/title',
                          json={'title': bad_title}, headers=AUTH)

    assert response.status_code == 400
    assert cursor.execute.call_count == 0


def test_rename_session_requires_auth(client):
    """缺 Authorization → 401。"""
    response = client.put('/session/sess_001/title', json={'title': 'x'})
    assert response.status_code == 401


def test_rename_session_does_not_touch_messages(client, mock_db):
    """专用端点只写 title 列，绝不触碰 messages —— 这是它存在的理由。"""
    _, cursor = mock_db
    cursor.rowcount = 1

    client.put('/session/sess_001/title', json={'title': 'abc'}, headers=AUTH)

    sql = _executed_sql(cursor).lower()
    assert 'update conversations set title' in sql
    assert 'messages' not in sql
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  REDIS_HOST=127.0.0.1 REDIS_PORT=6379 PYTHONUTF8=1 \
  python -m pytest tests/test_session_api.py -v -k rename
```

Expected: 6 条全 FAIL，均为 `404`（路由不存在）。

- [ ] **Step 3: 加请求模型**

在 `api_routes/session.py` 的 `SaveMessagesRequest` 之后加：

```python
class RenameSessionRequest(BaseModel):
    title: str
```

- [ ] **Step 4: 加端点**

在 `api_routes/session.py` 的 `save_messages` 与 `archive_session` 之间插入：

```python
    @router.put("/session/{session_id}/title")
    async def rename_session(session_id: str, req: RenameSessionRequest,
                             http_request: Request):
        """只改 title 列，不碰 messages。

        专用端点而非复用 PUT /messages：后者会重写整个 messages 数组，
        流式对话落库期间改名会丢消息。
        """
        auth_header = http_request.headers.get("Authorization")
        user = verify_firebase_token(auth_header)
        user_id = int(user['uid'])

        title = req.title.strip()
        if not title:
            raise HTTPException(status_code=400, detail="Title must not be empty")
        if len(title) > 256:
            raise HTTPException(status_code=400, detail="Title too long")

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE conversations SET title = %s
                       WHERE session_id = %s AND user_id = %s AND status != 2""",
                    (title, session_id, user_id))
                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Session not found")
                conn.commit()
            logger.info(f"Session renamed: {session_id}")
            return {"success": True}
        finally:
            conn.close()
```

- [ ] **Step 5: 跑全量测试确认通过**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  REDIS_HOST=127.0.0.1 REDIS_PORT=6379 PYTHONUTF8=1 \
  python -m pytest tests/test_session_api.py -v
```

Expected: `25 passed`。

- [ ] **Step 6: 确认没打断既有会话相关测试**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  REDIS_HOST=127.0.0.1 REDIS_PORT=6379 PYTHONUTF8=1 \
  python -m pytest tests/test_session_api.py tests/test_session_anchor.py -q
```

Expected: 全绿（`test_session_anchor.py` 基线 9 passed）。

- [ ] **Step 7: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add api_routes/session.py tests/test_session_api.py && \
  git commit -m "feat: 会话重命名专用端点 PUT /session/{id}/title——只写 title 列不碰 messages"
```

---

## Task 4: 前端 service 层加 rename / archive

**Files:**
- Modify: `frontend/weapp/src/services/sessions.ts`

**Interfaces:**
- Consumes: Task 3 的 `PUT /session/{id}/title`、Task 2 的 `DELETE /session/{id}`
- Produces: `renameSession(sessionId: string, title: string): Promise<void>`、`archiveSession(sessionId: string): Promise<void>`

- [ ] **Step 1: 加两个函数**

在 `frontend/weapp/src/services/sessions.ts` 末尾追加（`request` 已在文件顶部 import）：

```ts
/** 重命名会话（专用端点，只改标题，不重写 messages）。 */
export async function renameSession(
  sessionId: string,
  title: string,
): Promise<void> {
  await request(`/session/${sessionId}/title`, {
    method: 'PUT',
    data: { title: title.slice(0, 60) },
  })
}

/** 归档会话（后端置 status=2，列表与详情都不再返回）。 */
export async function archiveSession(sessionId: string): Promise<void> {
  await request(`/session/${sessionId}`, { method: 'DELETE' })
}
```

- [ ] **Step 2: 类型检查**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc
```

Expected: exit 0，无输出。

- [ ] **Step 3: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add frontend/weapp/src/services/sessions.ts && \
  git commit -m "feat(weapp): sessions service 加重命名/归档接口"
```

---

## Task 5: NavBar 自绘顶栏

`☰` 要落在导航栏左上角，必须放弃系统导航栏（详见 spec §3.2）。

**Files:**
- Create: `frontend/weapp/src/components/NavBar/index.tsx`
- Create: `frontend/weapp/src/components/NavBar/index.scss`

**Interfaces:**
- Consumes: 无
- Produces: `default export NavBar`，props `{ title: string; onMenuClick: () => void }`

- [ ] **Step 1: 写组件**

创建 `frontend/weapp/src/components/NavBar/index.tsx`：

```tsx
import { useMemo } from 'react'
import { Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import './index.scss'

type Props = {
  title: string
  onMenuClick: () => void
}

const FALLBACK_NAV_HEIGHT = 44

/**
 * 自绘顶栏（页面配 navigationStyle: 'custom' 后系统导航栏不再渲染）。
 * 高度按微信官方公式算，避免与右上角胶囊按钮错位：
 *   navBarHeight = (胶囊top - 状态栏高度) * 2 + 胶囊高度
 * 胶囊在右上角、☰ 在左上角，水平方向不冲突，只需对齐垂直。
 */
export default function NavBar({ title, onMenuClick }: Props) {
  const { statusBarHeight, navBarHeight } = useMemo(() => {
    const info = Taro.getSystemInfoSync()
    const status = info.statusBarHeight || 0
    let height = FALLBACK_NAV_HEIGHT
    try {
      const menu = Taro.getMenuButtonBoundingClientRect()
      if (menu && menu.height) {
        height = (menu.top - status) * 2 + menu.height
      }
    } catch {
      // 取不到胶囊（非微信端/调试环境）时退回默认导航高
    }
    return { statusBarHeight: status, navBarHeight: height }
  }, [])

  return (
    <View className='navbar' style={{ paddingTop: `${statusBarHeight}px` }}>
      <View className='navbar-inner' style={{ height: `${navBarHeight}px` }}>
        <Text className='navbar-title'>{title}</Text>
        <View className='navbar-menu' onClick={onMenuClick}>
          <Text className='navbar-menu-icon'>☰</Text>
        </View>
      </View>
    </View>
  )
}
```

- [ ] **Step 2: 写样式**

创建 `frontend/weapp/src/components/NavBar/index.scss`：

```scss
.navbar {
  background: var(--c-bg);
  flex-shrink: 0;
}

.navbar-inner {
  position: relative;
  display: flex;
  align-items: center;
}

/* 标题居中；左右给胶囊/按钮留出空间，避免视觉偏移 */
.navbar-title {
  position: absolute;
  left: 160px;
  right: 160px;
  text-align: center;
  font-size: 32px;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.navbar-menu {
  position: relative;
  z-index: 1;
  width: 88px;
  height: 100%;
  margin-left: 12px;
  display: flex;
  align-items: center;
  justify-content: center;

  &:active {
    opacity: 0.5;
  }
}

.navbar-menu-icon {
  font-size: 40px;
  line-height: 1;
  color: var(--c-text);
}
```

- [ ] **Step 3: 类型检查**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc
```

Expected: exit 0。

- [ ] **Step 4: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add frontend/weapp/src/components/NavBar && \
  git commit -m "feat(weapp): NavBar 自绘顶栏——状态栏高度+胶囊避让"
```

---

## Task 6: RenameModal 重命名弹窗

平台约束：微信小程序**没有带输入框的原生弹窗**（`Taro.showModal` 无 `editable`，`window.prompt` 不存在），所以必须自绘。

**Files:**
- Create: `frontend/weapp/src/components/RenameModal/index.tsx`
- Create: `frontend/weapp/src/components/RenameModal/index.scss`

**Interfaces:**
- Consumes: 无
- Produces: `default export RenameModal`，props `{ visible: boolean; initialTitle: string; busy?: boolean; error?: string; onCancel: () => void; onConfirm: (title: string) => void }`

- [ ] **Step 1: 写组件**

创建 `frontend/weapp/src/components/RenameModal/index.tsx`：

```tsx
import { useEffect, useState } from 'react'
import { Button, Input, Text, View } from '@tarojs/components'
import './index.scss'

type Props = {
  visible: boolean
  initialTitle: string
  busy?: boolean
  error?: string
  onCancel: () => void
  onConfirm: (title: string) => void
}

const MAX_TITLE_LEN = 60

/**
 * 重命名弹窗（纯展示，不碰网络）。
 * 自绘理由：微信小程序没有带输入框的原生弹窗 —— Taro.showModal 只有
 * 确定/取消，window.prompt 在小程序不存在。
 */
export default function RenameModal({
  visible,
  initialTitle,
  busy,
  error,
  onCancel,
  onConfirm,
}: Props) {
  const [value, setValue] = useState(initialTitle)

  // 每次打开都用当前标题重新预填
  useEffect(() => {
    if (visible) setValue(initialTitle)
  }, [visible, initialTitle])

  if (!visible) return null

  const trimmed = value.trim()
  const canSubmit = trimmed.length > 0 && !busy

  return (
    <View className='rename-mask' catchMove>
      <View className='rename-box'>
        <Text className='rename-heading'>重命名对话</Text>
        <Input
          className='rename-input'
          value={value}
          maxlength={MAX_TITLE_LEN}
          focus
          placeholder='输入新的标题'
          placeholderClass='rename-placeholder'
          onInput={(e) => setValue(e.detail.value)}
        />
        {error ? <Text className='rename-err'>{error}</Text> : null}
        <View className='rename-actions'>
          <Button className='rename-btn' disabled={busy} onClick={onCancel}>
            取消
          </Button>
          <Button
            className='rename-btn rename-btn-primary'
            disabled={!canSubmit}
            onClick={() => onConfirm(trimmed)}
          >
            {busy ? '保存中…' : '保存'}
          </Button>
        </View>
      </View>
    </View>
  )
}
```

- [ ] **Step 2: 写样式**

创建 `frontend/weapp/src/components/RenameModal/index.scss`：

```scss
.rename-mask {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 300;
  background: rgba(0, 0, 0, 0.45);
  display: flex;
  align-items: center;
  justify-content: center;
}

.rename-box {
  width: 600px;
  padding: 40px 36px 28px;
  box-sizing: border-box;
  background: var(--c-surface);
  border-radius: 24px;
}

.rename-heading {
  display: block;
  font-size: 32px;
  font-weight: 600;
  text-align: center;
}

.rename-input {
  margin-top: 32px;
  padding: 18px 22px;
  box-sizing: border-box;
  background: var(--c-bg);
  border: 1px solid var(--c-border);
  border-radius: 14px;
  font-size: 30px;
}

.rename-placeholder {
  color: #aeb5bd;
}

.rename-err {
  display: block;
  margin-top: 16px;
  font-size: 24px;
  color: var(--c-danger);
}

.rename-actions {
  display: flex;
  margin-top: 32px;
  gap: 20px;
}

.rename-btn {
  flex: 1;
  height: 80px;
  line-height: 80px;
  background: var(--c-bg);
  color: var(--c-text);
  border-radius: 999px;
  font-size: 28px;

  &::after {
    border: none;
  }

  &[disabled] {
    opacity: 0.45;
  }
}

.rename-btn-primary {
  background: var(--c-primary);
  color: #fff;

  &[disabled] {
    color: #fff;
    background: var(--c-primary);
  }
}
```

- [ ] **Step 3: 类型检查**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc
```

Expected: exit 0。

- [ ] **Step 4: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add frontend/weapp/src/components/RenameModal && \
  git commit -m "feat(weapp): RenameModal 自绘重命名弹窗——小程序无原生输入弹窗"
```

---

## Task 7: SessionDrawer 会话抽屉

纯展示组件：只渲染列表并发意图，不碰网络。`…` 上的 ActionSheet 属于视图细节，留在组件内；删除的**二次确认**与网络由 chat 页负责。

**Files:**
- Create: `frontend/weapp/src/components/SessionDrawer/index.tsx`
- Create: `frontend/weapp/src/components/SessionDrawer/index.scss`

**Interfaces:**
- Consumes: `SessionItem` from `../../services/sessions`
- Produces: `default export SessionDrawer`，props `{ visible: boolean; sessions: SessionItem[]; currentSessionId: string; loading?: boolean; error?: string; onClose: () => void; onSelect: (sessionId: string) => void; onNew: () => void; onRename: (session: SessionItem) => void; onDelete: (session: SessionItem) => void }`

- [ ] **Step 1: 写组件**

创建 `frontend/weapp/src/components/SessionDrawer/index.tsx`：

```tsx
import { ScrollView, Text, View } from '@tarojs/components'
import Taro from '@tarojs/taro'
import type { SessionItem } from '../../services/sessions'
import './index.scss'

type Props = {
  visible: boolean
  sessions: SessionItem[]
  currentSessionId: string
  loading?: boolean
  error?: string
  onClose: () => void
  onSelect: (sessionId: string) => void
  onNew: () => void
  onRename: (session: SessionItem) => void
  onDelete: (session: SessionItem) => void
}

const ACTIONS = ['重命名', '删除']

/** 会话抽屉（纯展示：只发意图，不碰网络）。 */
export default function SessionDrawer({
  visible,
  sessions,
  currentSessionId,
  loading,
  error,
  onClose,
  onSelect,
  onNew,
  onRename,
  onDelete,
}: Props) {
  function showActions(session: SessionItem) {
    Taro.showActionSheet({
      itemList: ACTIONS,
      success: (res) => {
        if (res.tapIndex === 0) onRename(session)
        else if (res.tapIndex === 1) onDelete(session)
      },
      fail: () => {},
    })
  }

  if (!visible) return null

  return (
    <View className='drawer'>
      <View className='drawer-mask' onClick={onClose} catchMove />
      <View className='drawer-panel'>
        <View className='drawer-new' onClick={onNew}>
          <Text className='drawer-new-plus'>＋</Text>
          <Text className='drawer-new-text'>新对话</Text>
        </View>

        <ScrollView className='drawer-scroll' scrollY>
          {sessions.map((s) => (
            <View
              key={s.session_id}
              className={`drawer-item${
                s.session_id === currentSessionId ? ' drawer-item-active' : ''
              }`}
              onClick={() => onSelect(s.session_id)}
            >
              <Text className='drawer-item-title'>
                {s.title || '未命名对话'}
              </Text>
              <View
                className='drawer-item-more'
                onClick={(e) => {
                  e.stopPropagation()
                  showActions(s)
                }}
              >
                <Text className='drawer-item-more-icon'>…</Text>
              </View>
            </View>
          ))}

          {!loading && !error && sessions.length === 0 ? (
            <View className='drawer-empty'>
              <Text className='text-muted'>还没有对话</Text>
            </View>
          ) : null}
          {loading ? (
            <View className='drawer-empty'>
              <Text className='text-muted'>加载中…</Text>
            </View>
          ) : null}
          {error ? (
            <View className='drawer-empty'>
              <Text className='drawer-error'>{error}</Text>
            </View>
          ) : null}
        </ScrollView>
      </View>
    </View>
  )
}
```

- [ ] **Step 2: 写样式**

创建 `frontend/weapp/src/components/SessionDrawer/index.scss`：

```scss
.drawer {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 200;
}

.drawer-mask {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  background: rgba(0, 0, 0, 0.35);
}

.drawer-panel {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  width: 620px;
  background: var(--c-surface);
  display: flex;
  flex-direction: column;
  padding-top: env(safe-area-inset-top);
  box-shadow: 4px 0 24px rgba(31, 35, 40, 0.12);
}

.drawer-new {
  display: flex;
  align-items: center;
  padding: 32px 32px 24px;
  border-bottom: 1px solid var(--c-border);

  &:active {
    background: var(--c-bg);
  }
}

.drawer-new-plus {
  font-size: 40px;
  line-height: 1;
  color: var(--c-primary);
}

.drawer-new-text {
  margin-left: 16px;
  font-size: 30px;
  font-weight: 500;
}

.drawer-scroll {
  flex: 1;
  min-height: 0;
}

.drawer-item {
  display: flex;
  align-items: center;
  padding: 26px 24px 26px 32px;
  border-bottom: 1px solid var(--c-border);

  &:active {
    background: var(--c-bg);
  }
}

.drawer-item-active {
  background: var(--c-primary-soft);
}

.drawer-item-title {
  flex: 1;
  font-size: 28px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.drawer-item-more {
  width: 72px;
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;

  &:active {
    background: rgba(31, 35, 40, 0.08);
  }
}

.drawer-item-more-icon {
  font-size: 32px;
  line-height: 1;
  color: var(--c-text-muted);
}

.drawer-empty {
  padding: 48px 32px;
  text-align: center;

  Text {
    font-size: 26px;
  }
}

.drawer-error {
  color: var(--c-danger);
}
```

- [ ] **Step 3: 类型检查**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc
```

Expected: exit 0。

- [ ] **Step 4: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add frontend/weapp/src/components/SessionDrawer && \
  git commit -m "feat(weapp): SessionDrawer 会话抽屉——新建/切换/重命名/删除入口"
```

---

## Task 8: chat 页整合（首页化）

把三个组件挂上，chat 页持有全部状态与网络调用。同时去掉 `useRouter().params.session_id` —— 首页化后没有任何入口再传这个参数（唯一调用方 `pages/index/index.tsx:41-43` 在 Task 9 删除）。

**Files:**
- Modify: `frontend/weapp/src/pages/chat/index.tsx`
- Modify: `frontend/weapp/src/pages/chat/index.config.ts`

**Interfaces:**
- Consumes: Task 4 的 `renameSession` / `archiveSession`；Task 5/6/7 的三个组件
- Produces: `default export ChatPage`（首页）

- [ ] **Step 1: 页面配置改为自绘顶栏**

把 `frontend/weapp/src/pages/chat/index.config.ts` 整个替换为：

```ts
const config = {
  navigationStyle: 'custom',
  navigationBarTitleText: '对话',
}

export default config
```

- [ ] **Step 2: 重写 chat 页**

把 `frontend/weapp/src/pages/chat/index.tsx` 整个替换为：

```tsx
import { useCallback, useRef, useState } from 'react'
import {
  Button,
  RichText,
  ScrollView,
  Text,
  Textarea,
  View,
} from '@tarojs/components'
import Taro, { useDidShow } from '@tarojs/taro'
import {
  ChatMsg,
  createSession,
  extractPatentIds,
  fetchSession,
  saveMessages,
} from '../../services/chat'
import { streamQuery } from '../../services/chatStream'
import { errorText } from '../../services/api'
import { isLoggedIn } from '../../services/auth'
import {
  archiveSession,
  fetchSessions,
  renameSession,
  SessionItem,
} from '../../services/sessions'
import NavBar from '../../components/NavBar'
import SessionDrawer from '../../components/SessionDrawer'
import RenameModal from '../../components/RenameModal'
import { markdownToHtml } from '../../utils/markdown'
import './index.scss'

interface MsgView {
  role: 'user' | 'assistant'
  content: string
  html?: string
  patents?: string[]
  streaming?: boolean
}

/**
 * 形态一：首页即对话页（DeepSeek App 式）。
 * 左上角 ☰ 呼出抽屉收历史，抽屉内可新建/切换/重命名/删除。
 * 富文本经 <RichText> 渲染（weapp 正确原语；内容来自自有后端，用户输入恒为纯文本）。
 */
export default function ChatPage() {
  const sessionIdRef = useRef('')
  const assistantRef = useRef('') // 最新一轮助手全文（落库用，防闭包过期）
  const [msgs, setMsgs] = useState<MsgView[]>([])
  const [sessionTitle, setSessionTitle] = useState('')
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [anchor, setAnchor] = useState('')

  // 抽屉
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [sessions, setSessions] = useState<SessionItem[]>([])
  const [listLoading, setListLoading] = useState(false)
  const [listError, setListError] = useState('')

  // 重命名弹窗
  const [renameTarget, setRenameTarget] = useState<SessionItem | null>(null)
  const [renaming, setRenaming] = useState(false)
  const [renameError, setRenameError] = useState('')

  const scrollToBottom = () => setAnchor(`msg-${Date.now()}`)

  // 首页必须自己把门（原 pages/index 的职责搬来）
  useDidShow(() => {
    if (!isLoggedIn()) {
      Taro.navigateTo({ url: '/pages/login/index' })
    }
  })

  const loadSessions = useCallback(async () => {
    setListLoading(true)
    setListError('')
    try {
      setSessions(await fetchSessions())
    } catch (err) {
      setListError(errorText(err, '会话列表加载失败'))
    } finally {
      setListLoading(false)
    }
  }, [])

  function openDrawer() {
    setDrawerOpen(true)
    loadSessions()
  }

  function resetToNewChat() {
    sessionIdRef.current = ''
    assistantRef.current = ''
    setMsgs([])
    setSessionTitle('')
    setError('')
    setStatus('')
  }

  /** ＋ 新对话：只重置本地状态，不立刻建会话（否则每点一次留一条空会话）。 */
  function newChat() {
    resetToNewChat()
    setDrawerOpen(false)
  }

  async function selectSession(sessionId: string) {
    setDrawerOpen(false)
    if (sessionId === sessionIdRef.current) return

    sessionIdRef.current = sessionId
    setError('')
    setMsgs([])
    try {
      const detail = await fetchSession(sessionId)
      const history: MsgView[] = (detail.messages || [])
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .map((m: ChatMsg) => ({
          role: m.role as 'user' | 'assistant',
          content: m.content || '',
        }))
      setMsgs(history)
      setSessionTitle(detail.title || '')
      scrollToBottom()
    } catch (err) {
      // 顺序要紧：resetToNewChat() 内部会 setError('')，
      // 必须先重置再设错误，否则错误提示会被同批 state 更新覆盖掉。
      resetToNewChat()
      setError(errorText(err, '历史会话加载失败'))
    }
  }

  async function confirmRename(title: string) {
    const target = renameTarget
    if (!target) return
    setRenaming(true)
    setRenameError('')
    try {
      await renameSession(target.session_id, title)
      setSessions((prev) =>
        prev.map((s) =>
          s.session_id === target.session_id ? { ...s, title } : s,
        ),
      )
      if (target.session_id === sessionIdRef.current) setSessionTitle(title)
      setRenameTarget(null)
    } catch (err) {
      setRenameError(errorText(err, '重命名失败，请重试'))
    } finally {
      setRenaming(false)
    }
  }

  function removeSession(session: SessionItem) {
    Taro.showModal({
      title: '删除对话',
      content: `确定删除「${session.title || '未命名对话'}」吗？`,
      confirmText: '删除',
      confirmColor: '#d32f2f',
      success: async (res) => {
        if (!res.confirm) return
        try {
          await archiveSession(session.session_id)
          setSessions((prev) =>
            prev.filter((s) => s.session_id !== session.session_id),
          )
          // 删的正好是当前会话 → 回空态，避免停在已归档会话上
          if (session.session_id === sessionIdRef.current) resetToNewChat()
        } catch (err) {
          setListError(errorText(err, '删除失败，请重试'))
        }
      },
    })
  }

  const appendToken = useCallback((chunk: string) => {
    assistantRef.current += chunk
    setMsgs((prev) => {
      const next = prev.slice()
      const last = next[next.length - 1]
      if (last && last.role === 'assistant') {
        next[next.length - 1] = { ...last, content: assistantRef.current }
      }
      return next
    })
  }, [])

  const startAssistant = useCallback(() => {
    assistantRef.current = ''
    setMsgs((prev) => [
      ...prev,
      { role: 'assistant', content: '', streaming: true },
    ])
  }, [])

  const finalizeAssistant = useCallback(() => {
    setMsgs((prev) => {
      const next = prev.slice()
      const last = next[next.length - 1]
      if (last && last.role === 'assistant') {
        next[next.length - 1] = {
          ...last,
          streaming: false,
          patents: extractPatentIds(last.content),
        }
      }
      return next
    })
  }, [])

  async function send() {
    const text = input.trim()
    if (!text || sending) return
    setInput('')
    setSending(true)
    setStatus('连接中…')
    setError('')
    try {
      const history: ChatMsg[] = msgs.map((m) => ({
        role: m.role,
        content: m.content,
      }))
      const userMsg: ChatMsg = { role: 'user', content: text }

      // 首条消息建会话（scene 1 = 专利检索默认场景）
      let sid = sessionIdRef.current
      if (!sid) {
        sid = await createSession(text, [...history, userMsg])
        sessionIdRef.current = sid
        setSessionTitle(text.slice(0, 60))
      }
      setMsgs((prev) => [...prev, { role: 'user', content: text }])
      scrollToBottom()

      startAssistant()
      let completed = false
      try {
        await streamQuery(text, history, {
          onStatus: (s) => {
            if (!completed) setStatus(s)
          },
          onToken: (chunk) => {
            if (!completed) {
              setStatus('')
              appendToken(chunk)
            }
          },
          onError: (message) => setError(message),
        })
      } finally {
        completed = true
        finalizeAssistant()
      }
      // 终稿落库（web 同款持久化）
      const assistantMsg: ChatMsg = {
        role: 'assistant',
        content: assistantRef.current,
      }
      await saveMessages(sid, [...history, userMsg, assistantMsg])
    } catch (err) {
      setError(errorText(err))
    } finally {
      setSending(false)
      setStatus('')
      scrollToBottom()
    }
  }

  function copyPatent(pid: string) {
    Taro.setClipboardData({ data: pid })
  }

  return (
    <View className='chat'>
      <NavBar title={sessionTitle || '新对话'} onMenuClick={openDrawer} />

      <ScrollView
        className='chat-scroll'
        scrollY
        scrollIntoView={anchor}
        scrollWithAnimation
      >
        <View className='chat-list'>
          {msgs.length === 0 ? (
            <View className='chat-welcome'>
              <Text className='chat-welcome-title'>专利智能对话</Text>
              <Text className='chat-welcome-sub text-muted'>
                描述您的产品、技术或专利号，例如：
              </Text>
              <Text className='chat-welcome-example text-muted'>
                “查一下可折叠桌子相关的专利”
              </Text>
            </View>
          ) : null}

          {msgs.map((m, i) => (
            <View
              key={`${m.role}-${i}`}
              className={`chat-msg chat-msg-${m.role}`}
            >
              {m.role === 'assistant' && m.content ? (
                <View className='chat-msg-body'>
                  <RichText nodes={markdownToHtml(m.content)} />
                </View>
              ) : (
                <View className='chat-msg-body'>
                  <Text className='chat-msg-text'>
                    {m.content || (m.streaming ? '…' : '')}
                  </Text>
                </View>
              )}
              {m.role === 'assistant' && m.patents && m.patents.length > 0 ? (
                <View className='chat-msg-patents'>
                  {m.patents.map((pid) => (
                    <View
                      key={pid}
                      className='chat-patent'
                      onClick={() => copyPatent(pid)}
                    >
                      <Text className='chat-patent-id'>{pid}</Text>
                      <Text className='chat-patent-copy'>复制</Text>
                    </View>
                  ))}
                </View>
              ) : null}
            </View>
          ))}

          {status ? (
            <View className='chat-status'>
              <Text className='text-muted'>{status}</Text>
            </View>
          ) : null}
          {error ? (
            <View className='chat-err'>
              <Text>{error}</Text>
            </View>
          ) : null}
        </View>
      </ScrollView>

      <View className='chat-inputbar'>
        <Textarea
          className='chat-textarea'
          value={input}
          maxlength={4000}
          autoHeight
          placeholder='输入问题…'
          placeholderClass='chat-placeholder'
          onInput={(e) => setInput(e.detail.value)}
          disabled={sending}
        />
        <Button
          className='chat-send'
          disabled={sending || !input.trim()}
          onClick={send}
        >
          {sending ? '…' : '发送'}
        </Button>
      </View>

      <SessionDrawer
        visible={drawerOpen}
        sessions={sessions}
        currentSessionId={sessionIdRef.current}
        loading={listLoading}
        error={listError}
        onClose={() => setDrawerOpen(false)}
        onSelect={selectSession}
        onNew={newChat}
        onRename={(s) => {
          setRenameError('')
          setRenameTarget(s)
        }}
        onDelete={removeSession}
      />

      <RenameModal
        visible={renameTarget !== null}
        initialTitle={renameTarget?.title || ''}
        busy={renaming}
        error={renameError}
        onCancel={() => setRenameTarget(null)}
        onConfirm={confirmRename}
      />
    </View>
  )
}
```

- [ ] **Step 3: 类型检查**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc
```

Expected: exit 0。

- [ ] **Step 4: 构建**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run build:weapp
```

Expected: `Compiled successfully`（约 7-19s）。

- [ ] **Step 5: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add frontend/weapp/src/pages/chat && \
  git commit -m "feat(weapp): chat 页首页化——挂 NavBar/SessionDrawer/RenameModal，接入会话切换与增删改"
```

---

## Task 9: 切换首页 + 删除列表页 + 登录兜底改向

**这一步有个必须一起改的连带点**：`pages/login/index.tsx:23` 登录成功后的兜底是 `Taro.reLaunch({ url: '/pages/index/index' })` —— 删掉 index 页后这条路径会指向不存在的页面。必须同步改成 `/pages/chat/index`。

**Files:**
- Modify: `frontend/weapp/src/app.config.ts`
- Modify: `frontend/weapp/src/pages/login/index.tsx:23`
- Delete: `frontend/weapp/src/pages/index/`（三文件）

**Interfaces:**
- Consumes: Task 8 首页化的 chat 页
- Produces: 无

- [ ] **Step 1: app.config 改首页**

把 `frontend/weapp/src/app.config.ts` 的 `pages` 数组改为（去掉 index，chat 置首）：

```ts
  pages: [
    'pages/chat/index',  // 对话页（形态一：首页即对话页）
    'pages/login/index', // 微信登录（M1）
  ],
```

- [ ] **Step 2: 修 login 页的兜底跳转**

`frontend/weapp/src/pages/login/index.tsx:23`：

```ts
      Taro.navigateBack({
        fail: () => Taro.reLaunch({ url: '/pages/chat/index' }),
      })
```

- [ ] **Step 3: 确认没有别的地方还引用 index 页**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  grep -rn "pages/index/index" frontend/weapp/src/ || echo "OK：无残留引用"
```

Expected: `OK：无残留引用`。若有输出，先把引用改掉再继续。

- [ ] **Step 4: 删除列表页**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git rm -r frontend/weapp/src/pages/index
```

（若微信开发者工具正开着占用 `dist/`，先关掉再执行。）

- [ ] **Step 5: 类型检查 + 构建**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && npm run tsc && npm run build:weapp
```

Expected: tsc exit 0；构建 `Compiled successfully`。

- [ ] **Step 6: 确认构建产物里没有残留的 index 页**

```bash
cd E:/online/workspace/copiioai/langsistance/frontend/weapp && \
  ls dist/pages/ && echo "---" && grep -o "pages/index/index" dist/app.json || echo "OK：app.json 已无 index 页"
```

Expected: 只列出 `chat` 与 `login` 两个目录；`app.json` 无 index 引用。

- [ ] **Step 7: 提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git add frontend/weapp/src/app.config.ts frontend/weapp/src/pages/login/index.tsx && \
  git commit -m "feat(weapp): 首页切到 chat 页并删除会话列表页，登录兜底改向 /pages/chat/index"
```

---

## Task 10: 端到端验证

**Files:** 无代码改动；产出验证记录。

- [ ] **Step 1: 后端全量回归**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  REDIS_HOST=127.0.0.1 REDIS_PORT=6379 PYTHONUTF8=1 \
  python -m pytest tests/test_session_api.py tests/test_session_anchor.py tests/test_wechat_login.py -q
```

Expected: 全绿。

- [ ] **Step 2: 确认无密码学/凭据文件被误提交**

```bash
cd E:/online/workspace/copiioai/langsistance && \
  git log --name-only -10 --format="%h %s" | grep -i "firebase_service_key" && \
    echo "!! 凭据文件被提交了，必须移除" || echo "OK：凭据文件未入库"
```

Expected: `OK：凭据文件未入库`。

- [ ] **Step 3: 微信开发者工具真机验证（人工）**

在微信开发者工具导入 `frontend/weapp/dist/`，逐条走 spec §8 的验收标准：

1. 打开即是对话页，左上角有 `☰`，顶栏不与右上角胶囊重叠
2. `☰` 呼出抽屉，列出历史会话，当前会话高亮
3. 抽屉内可新建、切换、重命名、删除
4. 删除当前会话后回到空态，可正常开始新对话
5. 登录态失效时跳登录页，登录后能回到对话页

- [ ] **Step 4: 记录验证结果**

把 Step 3 的实际结果追加到本计划文件末尾的「执行记录」小节（成功/失败、失败时的现象与日志）。

---

## 执行记录

**执行方式**：subagent-driven development（每任务独立实现 + 独立评审），2026-09-11。
**提交范围**：`8e95e74`（计划）→ `e2ad692`，共 10 个提交。

### 自动化验证结果

| 项 | 命令 | 结果 |
|---|---|---|
| Task 10 Step 1 后端回归 | `REDIS_HOST=127.0.0.1 REDIS_PORT=6379 PYTHONUTF8=1 python -m pytest tests/test_session_api.py tests/test_session_anchor.py tests/test_wechat_login.py -q` | **52 passed**（session 30 + anchor 9 + wechat 13） |
| Task 10 Step 2 凭据未入库 | `git log --name-only \| grep -i firebase_service_key` | **无命中**，`firebase_service_key.json` 未入库 |
| 前端类型检查 | `cd frontend/weapp && npm run tsc` | exit 0 |
| 前端构建 | `cd frontend/weapp && npm run build:weapp` | `Compiled successfully` |
| 首页切换 | `cat frontend/weapp/dist/app.json` | `{"pages":["pages/chat/index","pages/login/index"],...}` |

### 计划外发现并修复的问题

1. **`login/index.tsx:23` 的兜底跳转**指向被删除的 `pages/index/index` —— 已在 Task 9 同步改为 `/pages/chat/index`。
2. **既有测试基线是红的** —— `test_create_session` / `test_get_user_sessions` 早已因端点要求鉴权而失败（测试债），Task 1 修复后基线才回到 7 passed。
3. **本机测试环境缺 `firebase_admin`** —— 已装入系统 python；另需在仓库根放置 gitignored 的一次性假凭据 `firebase_service_key.json` 才能 import `passport.py`。
4. **`append_message` 的 SELECT 补了 `status != 2`** —— 计划要求的顺带修正，已归档会话不再能追加消息。
5. **删除失败的静默吞错** —— `setListError` 只在抽屉内渲染，删除失败时抽屉通常已关闭，错误不可见。终审发现，已改为 `Taro.showToast`（`e2ad692`）。

### 进程事实（供后续参考）

- Task 2 的 RED 实测为 4 条失败而非计划预期的 5 条：`PUT /messages` 本就调用了 `verify_firebase_token` 只是丢弃返回值 —— 这正是"注释声称校验归属、SQL 却没有"的根源。
- Task 3 的 `test_rename_session_not_owner` 在 RED 阶段空过（路由缺失与"非本人"同为 404）。评审确认实现后非空过，可接受。
- Taro 4 React 运行时**不产出逐组件目录**，组件内联进宿主页面的 `index.wxss`；全局 `comp.json` 为 `styleIsolation: "apply-shared"`，故 `app.scss` 中 `page` 上的 `--c-*` token 可达组件。

### Task 10 Step 3 真机验收 —— **未执行，待人工**

以下两项无法自动化验证，需在微信开发者工具 / 真机上确认：

1. **`NavBar` 标题的 `right: 160px` 硬编码 inset** 在异形状态栏/胶囊几何下是否错位（spec §7 已列为风险）。若真机出现偏移，应改为由 `Taro.getMenuButtonBoundingClientRect().right` 推导。
2. **删除失败的 toast** 实际是否弹出（本项目无前端测试框架，该路径仅通过编译验证）。

另需逐条走 spec §8 的五条验收标准（首页即对话页、☰ 抽屉、增删改、删当前会话回空态、跨账号 404、web/nextjs 回归）。
