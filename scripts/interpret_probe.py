#!/usr/bin/env python3
"""Probe: can a strong LLM produce architecture-level (方案级) patent
interpretation from a bare user query?

Runs a generic interpretation prompt against several provider/model
candidates and prints the JSON plus a check for architecture-level
terms (error amplifier / VCR / reference signal / constant current /
per-channel loop ...).  The prompt is generic — the query is a runtime
argument, and the CPC hints come from the real cpc match + local CPC
titles.  No production code is touched.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_env(path: str) -> None:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                value = value.strip()
                if (len(value) >= 2 and value[0] == value[-1]
                        and value[0] in ("'", '"')):
                    value = value[1:-1]
                os.environ.setdefault(key.strip(), value)
    except OSError:
        pass


_load_env(os.path.join(os.path.dirname(__file__), "..", ".env"))

QUERY = (sys.argv[1] if len(sys.argv) > 1
         else "控制放大器，独立控制 RGB 颜色输出")
# Real cpc match for this query (production log 2026-08-16):
CPC_MATCH = ["H05B45/20", "Y10S388/91", "B60Y2400/92",
             "H03F2200/264", "G09G3/16"]

_TITLES = {}
try:
    with open(os.path.join(os.path.dirname(__file__), "..",
                           "data/cpc/cpc_titles_subgroups.json"),
              encoding="utf-8") as fh:
        for entry in json.load(fh) or []:
            if isinstance(entry, dict) and entry.get("code"):
                _TITLES[entry["code"]] = entry.get("title", "")
except (OSError, ValueError):
    pass

cpc_lines = []
for code in CPC_MATCH:
    title = _TITLES.get(code) or "(无标题)"
    cpc_lines.append(f"{code}: {title}")

SYSTEM = (
    "你是资深专利检索专家，熟悉 US 授权专利的撰写风格。"
    "用户会给出一句技术需求（可能来自非专利领域的中文表述）。"
    "你的任务是做专利检索级的技术解读：把需求映射到专利文献中实际的"
    "电路/系统架构模式，并产出可直接执行的检索词。"
    "只输出 JSON，不要其他文字。"
)

USER = """技术需求：{query}

该需求经 CPC 语义匹配命中以下分类（含分类标题线索）：
{cpc}

请输出 JSON，字段如下：
1. "scheme": 该需求在专利文献中通常对应的电路/系统架构模式（1-2 句，如"每通道独立的恒流控制环路"）
2. "structure_terms": 该方案在专利中可能出现的核心结构/元件英文词，10-15 个（如 error amplifier、voltage controlled resistor、reference signal、constant current、multi-channel 这类方案级词汇，不是直译词）
3. "independence_terms": "独立控制"在专利文献中的常见英文表述（如 per-channel、individual、separate loop、independently）
4. "scenarios": 该需求可能出现的专利应用场景（3-5 个）
5. "queries": 用于 US 授权专利全文检索的布尔检索式 3-5 条，方案词优先、可直接执行（引号包裹精确短语）"""


def check_architecture(text: str) -> dict:
    arch_terms = [
        "error amplifier", "voltage controlled resistor", "vcr",
        "reference signal", "constant current", "current control circuit",
        "per-channel", "individual channel", "channel controller",
        "sense", "feedback loop", "closed loop", "linear regulator",
        "multi-channel", "color channel", "black body", "dimming curve",
    ]
    low = text.lower()
    return {t: (t in low) for t in arch_terms}


def run_one(model: str, provider_name: str) -> str:
    """Interpret via the project Provider (production path), falling back
    to a direct OpenAI-compatible SDK call against OPENAI_BASE_URL.
    Meant to run on the deployment server where the gateway is
    reachable; locally it fails with connection errors."""
    user = USER.format(query=QUERY, cpc="\n".join(cpc_lines))
    try:
        from sources.llm_provider import Provider
        provider = Provider(provider_name=provider_name, model=model,
                            server_address="", is_local=False)
        return str(provider.complete_json(SYSTEM, user, max_retries=1))
    except Exception as exc:
        print(f"   (Provider 失败，回退 SDK: {type(exc).__name__}: "
              f"{str(exc)[:100]})")
        from openai import OpenAI
        base = os.environ.get("OPENAI_BASE_URL", "")
        key = (os.environ.get("OPENAI_API_KEY", "") or
               os.environ.get("DEEPSEEK_API_KEY", ""))
        if not base or not key:
            raise RuntimeError("OPENAI_BASE_URL/API key 缺失")
        client = OpenAI(api_key=key, base_url=base)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=2500,
        )
        return resp.choices[0].message.content or ""


def main() -> int:
    attempts = [
        ("deepseek-v4-pro", "deepseek"),
        ("deepseek-v4-flash", "deepseek"),
    ]
    for model, provider_name in attempts:
        print(f"\n{'='*70}\n== {provider_name} / {model}")
        try:
            content = run_one(model, provider_name, provider_name)
        except Exception as exc:
            print(f"   FAILED: {type(exc).__name__}: {str(exc)[:160]}")
            continue
        print(content[:2500])
        hits = [t for t, ok in check_architecture(content).items() if ok]
        print(f"\n-- 方案级词汇命中({len(hits)}/17): {', '.join(hits)}")
        if len(hits) >= 6:
            print(">> 判定：达到方案级解读门槛")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
