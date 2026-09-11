#!/usr/bin/env python3
"""工件生成必须走线程，不得占用事件循环。

``build_result_artifacts`` 是同步 CPU 活（纯 Python 的 zipfile 拼 multi-MB
的 xlsx/csv）。直接在事件循环里调用会阻塞整个循环，``/query_stream`` 的 SSE
生成器随之停转、``: ping`` 心跳断供。客户端在答案正文已经出完之后长时间
收不到任何字节，于是把一个已经答完的对话判成「请求超时，请重试」。

回归点就是「跑在别的线程上」——断言线程身份，而不是断言它被调用过：
后者在直接 await 同步函数的写法下同样会通过。
"""

import asyncio
import os
import sys
import threading
from unittest.mock import MagicMock, patch

os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
sys.modules.setdefault("firebase_admin", MagicMock())


def test_build_artifacts_runs_off_the_event_loop():
    """必须在工作线程里执行，否则会阻塞整条 SSE 流。"""
    from sources.agents import general_agent

    seen = {}

    def fake_build(*args, **kwargs):
        seen["thread"] = threading.current_thread()
        seen["args"] = args
        seen["kwargs"] = kwargs
        return ["artifact"]

    caller_thread = threading.current_thread()

    with patch.object(general_agent, "build_result_artifacts", fake_build):
        result = asyncio.run(
            general_agent._build_artifacts_off_loop("items", source="uspto")
        )

    assert result == ["artifact"]
    assert seen["thread"] is not caller_thread, (
        "工件生成跑在主线程上——会阻塞事件循环、掐断 SSE 心跳"
    )
    # 参数原样透传，不能被包装函数吃掉
    assert seen["args"] == ("items",)
    assert seen["kwargs"] == {"source": "uspto"}


def test_general_agent_has_no_direct_artifact_calls():
    """三个调用点都必须经由包装函数——漏掉一个，阻塞就回来了。"""
    import inspect
    from sources.agents import general_agent

    src = inspect.getsource(general_agent)
    assert "= build_result_artifacts(" not in src, (
        "存在绕过 _build_artifacts_off_loop 的直接调用，会阻塞事件循环"
    )
    assert src.count("await _build_artifacts_off_loop(") == 3
