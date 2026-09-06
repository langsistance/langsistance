"""design_vision 薄视觉适配器 (design P1 T5) — 镜像 patent_analyzer vision wire。

规格: docs/superpowers/specs/2026-09-06-us-design-clearance-design.md §5;
task-5-brief.md。
本模块只做"URL/鉴权头/payload/超时"的组装与 200 判定, 不参与文案/维度编排
(编排在 pipeline 层, Task 6/7 复用); 生产代码零产品词, 持久对象无 IO。

wire 对齐参考 (写代码时已读源, 不另起炉灶):
- sources/long_task/patent_analyzer.py:435-478 `_call_vision_api` —— OpenAI 兼容
  ChatCompletions: system+user 消息, user 内容含 text 块 + image_url data-uri 页图块;
  temperature=0.3, max_tokens=4096; 取 choices[0].message.content。
- sources/long_task/config.py:19-60 —— [LONG_TASK] vision_* 默认 minimax/MiniMax-M3。
- 本项目 minimax 端与密钥源与 llm_provider minimax 分支同源 (MINIMAX_API_KEY / MINIMAX_API_BASE)。

接口形态按 brief: call_vision 的 post(url, headers, json, timeout) 由调用方注入 (测试),
生产缺省走 httpx.AsyncClient.post; non-200/网络异常 → DesignVisionError(str 含 根因)。
enabled=false → DesignVisionError("vision disabled") (Task 7 编排拿该异常走文本降级)。
"""
import configparser
import os
import json

_DEFAULT_PROVIDER = "minimax"
_DEFAULT_MODEL = "MiniMax-M3"
_DEFAULT_CHAT_URL = "https://api.minimaxi.com/v1/chat/completions"
_DEFAULT_TIMEOUT = 90
_MAX_TOKENS = 4096
_MAX_PREVIEW = 300   # 错误信息里截断响应体预览长度
_MD_JSON_HEADERS = {"Content-Type": "application/json"}


class DesignVisionError(RuntimeError):
    """视觉调用失败 (disabled / 密钥缺失 / 非 200 / 网络异常)。带响应预览与状态码。"""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def load_vision_config(config_path: str = "config.ini") -> dict:
    """读 [LONG_TASK] vision_provider/vision_model/vision_enabled; 缺省 minimax/MiniMax-M3/True。

    env 覆盖留待 Task 7 (见 brief)。返回 {"provider","model","enabled"}。
    """
    cfg = configparser.ConfigParser()
    try:
        cfg.read(config_path, encoding="utf-8")
    except Exception:
        cfg.read(config_path)
    provider = _DEFAULT_PROVIDER
    model = _DEFAULT_MODEL
    enabled = True
    if cfg.has_section("LONG_TASK"):
        provider = cfg.get("LONG_TASK", "vision_provider", fallback=_DEFAULT_PROVIDER)
        model = cfg.get("LONG_TASK", "vision_model", fallback=_DEFAULT_MODEL)
        enabled = cfg.getboolean("LONG_TASK", "vision_enabled", fallback=True)
    return {"provider": provider.strip(), "model": model.strip(), "enabled": enabled}


def _apply_config(config: dict | None) -> dict:
    """合并显式 config 与文件默认。显式 provider/model/enabled 优先, 缺则文件随源读取。"""
    resolved = load_vision_config()
    if config:
        for key in ("provider", "model", "enabled"):
            if key in config and config[key] is not None:
                resolved[key] = config[key]
    return resolved


def _prompt_kwargs(model: str, images_base64: list[str], prompt: str) -> dict:
    """组装为镜像 patent_analyzer 的 ChatCompletions messages (零 md5 固化)。"""
    user_content: list[dict] = [{"type": "text", "text": prompt}]
    for raw in images_base64:
        # images_base64 为 _pdf_to_base64_images 输出 data-uri (da开头自带头), 直接使用。
        user_content.append({"type": "image_url", "image_url": {"url": raw}})
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.3,
        "max_tokens": _MAX_TOKENS,
    }


async def call_vision(
    images_base64,
    prompt,
    post=None,
    timeout: int = 90,
    *,
    config=None,
) -> str:
    """把页图 + prompt 发给视觉模型, 返回 choices[0].message.content。

    签名契约 (与 brief 一致): images_base64, prompt, post, timeout 按位次; 优先级 config 为
    keyword-only 后置参数。config: {"provider","model","enabled"} (Provider 覆盖), 缺省走
    config.ini。post: async (url, headers, json, timeout) -> {status_code,text}; 测试注入禁
    真实网络, 缺省走 httpx 生产封装。非 200 / 网络异常 → DesignVisionError; enabled=false →
    同样抛 DesignVisionError("vision disabled")。

    注: 本编排把 provider 名解为鉴权来源, 简化单一视觉域; 键缺失 / 未知 provider → 报错。
    """
    cfg = _apply_config(config)
    if not cfg.get("enabled"):
        raise DesignVisionError("vision disabled")
    provider = (cfg.get("provider") or "").strip().lower()
    model = (cfg.get("model") or "").strip()
    uses_default_post = post is None  # 生产路径才强制鉴权密钥 (测试注入双用途禁真实外呼)
    key = _resolve_api_key(provider, require=uses_default_post)
    url = _resolve_chat_url(provider)
    payload = _prompt_kwargs(model, images_base64, prompt)
    headers = {**_MD_JSON_HEADERS, "Authorization": f"Bearer {key}"}
    poster = post or _http_post
    status = None
    try:
        resp = await poster(url, headers, payload, timeout)
        st, text = resp.status_code, getattr(resp, "text", "")
        status = int(st)
    except Exception as exc:  # 传输层异常 → DesignVisionError 带根因
        raise DesignVisionError(f"vision call failed — {exc}") from exc
    if status != 200:
        raise DesignVisionError(
            f"vision non-200 ({status}): {_preview_text(text)}", status_code=status
        )
    return _extract_content(text, model)


def _resolve_api_key(provider: str, require: bool = True) -> str:
    """OpenAI 兼容鉴权来源 (本项目 minimax 用 MINIMAX_API_KEY)。

    require=False (post 注入的测试场景) 查到即用, 查不到返回空壳不阻断 —— 测试禁真实外呼,
    不会用到真实密钥。require=True (生产 default-httpx) 缺失密钥 → 报错, 不做外呼。
    """
    env_names = {"minimax": ("MINIMAX_API_KEY",)}
    names = env_names.get(provider)
    if not names:
        raise DesignVisionError(f"unsupported vision provider '{provider}'")
    for name in names:
        candidate = os.getenv(name, "").strip()
        if candidate:
            return candidate
    if require:
        raise DesignVisionError(f"missing vision API key env for provider '{provider}'")
    return ""


def _resolve_chat_url(provider: str) -> str:
    """OpenAI 兼容端 (本项目 minimax 用 MINIMAX_API_BASE)。"""
    env_names = {"minimax": ("MINIMAX_API_BASE",)}
    names = env_names.get(provider)
    if not names:
        raise DesignVisionError(f"unsupported vision provider '{provider}'")
    override = os.getenv(names[0], "").strip()
    if override:
        return override if override.endswith("/chat/completions") else override.rstrip("/") + "/chat/completions"
    return _DEFAULT_CHAT_URL


async def _http_post(url, headers, json, timeout):
    """生产默认 post: httpx POST → {status_code, text}; 网络异常向上抛由调用方转 DesignVisionError。"""
    import httpx
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=json)
        return resp


def _extract_content(text: str, model: str) -> str:
    """从 ChatCompletions 响应提 choices[0].message.content; 非该形态回退原文。"""
    try:
        data = json.loads(text or "")
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, str) and content:
            return content
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        pass
    return text or ""


def _preview_text(text: str) -> str:
    """错误信息里的响应体截断预览 (避免日志超限)。"""
    return (text or "")[:_MAX_PREVIEW]
