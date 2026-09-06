"""design_vision 薄视觉适配器 (design P1 T5) — 契约测试。

镜像 patent_analyzer vision wire (见 brief/代码注释)。恒注入 post mock 禁真实网络;
invariant 级断言, 不拷死 wire 细节 (URL/头/payload 只断"存在且成形", 不断具体值)。
- call_vision 发送 images_base64(即 _pdf_to_base64_images 输出 data-uri 页图) + prompt,
  timeout 透传; 200 → 返回文本; 非 200 / 异常 → DesignVisionError。
- enabled=false → DesignVisionError("vision disabled")。post 缺省走 httpx 生产包装。
- 生产代码零产品词。函数 <50 行, 无 print。
"""
import asyncio

import sources.design.design_vision as dv
from sources.design.design_vision import (
    DesignVisionError,
    call_vision,
    load_vision_config,
)


# ---------- 无 config.ini 时的默认值 (provider/model/enabled 型态) ----------

def test_load_config_defaults_when_no_file(tmp_path):
    cfg_path = tmp_path / "missing.ini"
    out = load_vision_config(str(cfg_path))
    assert out["enabled"] is True
    assert isinstance(out["provider"], str) and out["provider"]
    assert isinstance(out["model"], str) and out["model"]


def test_load_config_reads_long_task_section(tmp_path):
    ini = tmp_path / "cfg.ini"
    ini.write_text(
        "[LONG_TASK]\n"
        "vision_enabled = false\n"
        "vision_provider = minimax\n"
        "vision_model = MiniMax-M3\n",
        encoding="utf-8",
    )
    out = load_vision_config(str(ini))
    assert out == {"provider": "minimax", "model": "MiniMax-M3", "enabled": False}


def test_load_config_absent_enabled_true(tmp_path):
    # 无 [LONG_TASK] 段 → enabled 默认 True。
    ini = tmp_path / "cfg.ini"
    ini.write_text("[OTHER]\nk=1\n", encoding="utf-8")
    out = load_vision_config(str(ini))
    assert out["enabled"] is True


# ---------- call_vision: 请求组装 (injected post, 禁真实网络) ----------

def test_call_vision_sends_images_and_prompt():
    calls = {}

    async def fake_post(url, headers, json, timeout):  # noqa: ARG001
        calls["url"] = url
        calls["headers"] = headers
        calls["json"] = json
        calls["timeout"] = timeout

        class R:
            status_code = 200
            text = '{"choices":[{"message":{"content":"looks distinctive"}}]}'

        return R()

    out = asyncio.run(call_vision(["aGVsbG8="], "describe", post=fake_post))
    assert calls["url"].startswith("http")
    assert "aGVsbG8=" in str(calls["json"])       # 页图 b64 进 payload
    assert "describe" in str(calls["json"])       # prompt 进 payload
    assert calls["timeout"] == 90                 # 默认超时透传
    # 鉴权头存在 (镜像 Authorization shell), 不断具体值。
    assert isinstance(calls["headers"], dict)
    assert out  # 200 → 模型文本返回 (非错误抛)


def test_call_vision_error_raises():
    async def bad(url, headers, json, timeout):  # noqa: ARG001
        class R:
            status_code = 500
            text = "boom"

        return R()

    try:
        asyncio.run(call_vision(["aGk="], "x", post=bad))
        assert False, "should raise"
    except DesignVisionError:
        pass


def test_call_vision_transport_exception_raises():
    async def boom(url, headers, json, timeout):  # noqa: ARG001
        raise RuntimeError("net down")

    try:
        asyncio.run(call_vision(["aGk="], "x", post=boom))
        assert False, "should raise"
    except DesignVisionError as exc:
        assert "net down" in str(exc)  # 异常链带根因, 便于排障


def test_call_vision_nonzero_timeout_forwarded():
    calls = {}

    async def spy(url, headers, json, timeout):  # noqa: ARG001
        calls["timeout"] = timeout

        class R:
            status_code = 200
            text = '{"choices":[{"message":{"content":"ok"}}]}'

        return R()

    asyncio.run(call_vision(["aA=="], "p", timeout=35, post=spy))
    assert calls["timeout"] == 35


def test_call_vision_missing_api_key_raises(monkeypatch):
    # 未注入 post → 走生产 httpx; 但无鉴权密钥时应明确报错, 不真实外呼。
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.delenv("MINIMAX_KEY", raising=False)
    try:
        asyncio.run(call_vision(["aGk="], "x"))
        assert False, "should raise"
    except DesignVisionError as exc:
        assert "key" in str(exc).lower()


# ---------- call_vision: config 显式参数 (PostProvider 覆盖) ----------

def test_call_vision_disabled_raises():
    # enabled=false (显式 config) → DesignVisionError("vision disabled"), 不做外呼。
    async def unexpected(*a, **k):  # noqa: ARG001
        raise AssertionError("post 不应被调用 (disabled 提前短路)")

    try:
        asyncio.run(call_vision(["aGk="], "x", post=unexpected, config={"enabled": False}))
        assert False, "should raise"
    except DesignVisionError as exc:
        assert "vision disabled" in str(exc)


def test_call_vision_returns_exact_extracted_text():
    # mock post 200 + choices[0].message.content → call_vision 恰返回该文本 (非仅非空)。
    async def fake_post(url, headers, json, timeout):  # noqa: ARG001
        class R:
            status_code = 200
            text = '{"choices":[{"message":{"content":"exact text"}}]}'

        return R()

    out = asyncio.run(call_vision(["aGVsbG8="], "describe", post=fake_post))
    assert out == "exact text"
