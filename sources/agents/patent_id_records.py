# -*- coding: utf-8 -*-
"""需求#29 自产 patent_ids 的可回读性 —— 来源感知的号码记录。

**问题**：会话里下发给前端面板的号码，此前只以**裸字符串列表**持久化
（Redis `lt:conv:{user}:patent_ids`）。号码一旦是 CN **公开号**
（`CN…A`），后续轮次就只能把它当自由文本检索词重新查——而佰腾真正
需要的是**申请号**（`app_num`），于是系统读不回自己刚产出的号码。

**解法**：持久化时连同「来源」与「来源原生键」一起记录，并据此判定
``retrievable``（能否免自由文本检索直接取回）。

**为什么是叶子模块**：``celery_worker.py`` 是独立进程入口，与
``general_agent`` 共用本模块的映射逻辑，因此**本模块自身只允许 stdlib
导入**（当前仅 ``json``/``typing``），不引入任何项目内依赖。

注意：该保证只覆盖模块体 —— 经由 ``sources.agents.patent_id_records``
导入时仍会执行 ``sources/agents/__init__.py``，从而拉起整个 agent 包。
worker 侧实测导入开销 <1s，可以接受；若将来 agent 包变重，把本模块移到
``sources/`` 下的叶子位置（``sources/__init__.py`` 是空的）即可恢复隔离。

``native_key`` 的定义是**该来源的查询 API 真正接受的键**；未知时不臆造
（宁可标不可回读，也不要给出一个查不到的键）。
"""
from __future__ import annotations

import json
from typing import Any

RECORD_VERSION = 2

# 与 general_agent._extract_patent_ids_from_items 的回退分支**逐项一致**——
# 平铺契约（返回哪些 id、什么顺序）必须保持不变，本模块只在其上追加
# 来源/原生键的标注。
_ID_KEYS: tuple = (
    "patent_id", "patent_number", "app_num",
    "applicationNumber", "patentApplicationNumber",
    "apc", "patentNumber", "专利申请号",
)

_MIN_ID_LEN = 8


def _is_uspto_application_number(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.isdigit() and 7 <= len(text) <= 12


def _normalize_record(entry: dict) -> dict:
    return {
        "id": str(entry.get("id") or "").strip(),
        "source": str(entry.get("source") or ""),
        "native_key": str(entry.get("native_key") or ""),
        "native_key_kind": str(entry.get("native_key_kind") or ""),
        "retrievable": bool(entry.get("retrievable")),
    }


def _legacy_record(pid: str) -> dict:
    """v1 存量值（裸号码）→ 记录。没有来源信息，因此不可回读。"""
    return {"id": pid, "source": "", "native_key": "",
            "native_key_kind": "", "retrievable": False}


def _record_for_key(item: dict, key: str, value: str) -> dict:
    """按命中字段与条目形状判定来源与原生键。"""
    if item.get("source") == "baiten":
        app_num = str(item.get("app_num") or "").strip()
        # 没有键就不给 kind 标签 —— 否则会诱导下游去解引用一个空键。
        return {"id": value, "source": "baiten",
                "native_key": app_num,
                "native_key_kind": "app_num" if app_num else "",
                "retrievable": bool(app_num)}
    if isinstance(item.get("applicationMetaData"), dict) or "patentNumber" in item:
        return {"id": value, "source": "uspto", "native_key": value,
                "native_key_kind": key, "retrievable": True}
    return {"id": value, "source": "", "native_key": "",
            "native_key_kind": "", "retrievable": False}


def _record_for_item(item: dict) -> dict | None:
    raw = str(item.get("applicationNumberText", "") or "").strip()
    if _is_uspto_application_number(raw):
        return {"id": raw, "source": "uspto", "native_key": raw,
                "native_key_kind": "applicationNumberText",
                "retrievable": True}
    for key in _ID_KEYS:
        value = str(item.get(key, "") or "").strip()
        if len(value) >= _MIN_ID_LEN:
            return _record_for_key(item, key, value)
    return None


def extract_patent_id_records(items: Any) -> list[dict]:
    """条目 → 号码记录列表。纯函数，永不抛。

    去重按 id 保持出现顺序（与旧行为一致）；同一个 id 重复出现时，
    **保留带原生键的那一条** —— 否则先到的无键形态会让后到的可回读
    信息白白丢掉。
    """
    out: list[dict] = []
    index: dict = {}
    for item in (items or []):
        if not isinstance(item, dict):
            continue
        record = _record_for_item(item)
        if record is None:
            continue
        pid = record["id"]
        if pid in index:
            previous = out[index[pid]]
            if record["retrievable"] and not previous["retrievable"]:
                out[index[pid]] = record
            continue
        index[pid] = len(out)
        out.append(record)
    return out


def records_from_long_task_rows(table_rows: Any, columns: Any) -> list[dict]:
    """celery 长任务完成后的号码行 → 记录。

    长任务表只有号码列，没有来源原生键，因此一律标为不可回读——宁少
    不滥，不给出查不到的键。
    """
    if not table_rows or not columns:
        return []
    id_column = columns[0]
    out: list[dict] = []
    seen: set = set()
    for row in table_rows:
        if not isinstance(row, dict) or row.get("_failed"):
            continue
        pid = str(row.get(id_column, "") or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append({"id": pid, "source": "long_task", "native_key": "",
                    "native_key_kind": "", "retrievable": False})
    return out


def encode_records(records: Any) -> str:
    """版本化落盘。``{"v": 2, "ids": [...]}`` —— 带版本号才能让读侧
    区分新旧形状。"""
    return json.dumps({"v": RECORD_VERSION, "ids": list(records or [])},
                      ensure_ascii=False)


def decode_records(raw: Any) -> list[dict]:
    """宽容读取：v2 版本化记录、v1 裸字符串列表、坏数据。

    部署顺序安全：滚动重启期间新旧代码可能同时读写同一个键，读侧必须
    两种形状都能吃，坏数据降级为空而不是抛。
    """
    if not raw:
        return []
    try:
        parsed = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except Exception:
        return []
    is_v2 = isinstance(parsed, dict)
    if is_v2:
        ids = parsed.get("ids")
        if not isinstance(ids, list):
            return []
    elif isinstance(parsed, list):
        ids = parsed
    else:
        return []

    out: list[dict] = []
    for entry in ids:
        if is_v2:
            if isinstance(entry, dict) and str(entry.get("id") or "").strip():
                out.append(_normalize_record(entry))
        elif isinstance(entry, str):
            pid = entry.strip()
            if pid:
                out.append(_legacy_record(pid))
    return out
