#!/usr/bin/env python3
"""
用户活动日志分析脚本

从 .logs/ 目录解析日志文件，按天统计：
- 每日活跃用户及每个用户的操作明细
- 各类操作次数：查询、创建知识、删除知识、反馈、分享等
- 使用率趋势

用法:
    python scripts/analyze_usage.py                  # 默认分析昨天
    python scripts/analyze_usage.py 2026-06-13       # 分析指定日期
    python scripts/analyze_usage.py 2026-06-10 2026-06-13  # 分析日期范围
    python scripts/analyze_usage.py --today          # 分析今天
    python scripts/analyze_usage.py --last 7         # 分析最近N天
"""

from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple


# ── 配置 ──────────────────────────────────────────────────────────────────────

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".logs")

# 日志行格式: YYYY-MM-DD HH:MM:SS,mmm - logfile - LEVEL - message
LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"  # timestamp
    r" - (\S+)"                                          # log file name
    r" - (INFO|WARNING|ERROR)"                            # level
    r" - (.*)"                                           # message
)

# 从日志消息中提取 user_id 的模式
USER_ID_PATTERNS = [
    re.compile(r"\[user=(\S+?)\]"),                              # [user=xxx]
    re.compile(r"for user:\s*(\S+)"),                            # for user: xxx
    re.compile(r"user_id[=:]\s*(\S+)"),                          # user_id=xxx
    re.compile(r"by user\s+(\S+)"),                              # by user xxx
    re.compile(r"from user_id:\s*(\S+)"),                        # from user_id: xxx
    re.compile(r"user\s+(\S+?)\s+(?:rejected|canceled|not)"),   # user xxx rejected/canceled
]

# 操作分类规则: (pattern, action_name)
ACTION_RULES = [
    (re.compile(r"Processing query", re.I), "query"),
    (re.compile(r"Processing query_stream", re.I), "query_stream"),
    (re.compile(r"Creating (?:tool and )?knowledge record", re.I), "create_knowledge"),
    (re.compile(r"Deleting (?:tool|knowledge)", re.I), "delete"),
    (re.compile(r"Updating (?:tool|knowledge)", re.I), "update"),
    (re.compile(r"Feedback submitted", re.I), "feedback"),
    (re.compile(r"Copy knowledge", re.I), "copy_knowledge"),
    (re.compile(r"Granted access to", re.I), "share_knowledge"),
    (re.compile(r"canceled knowledge share", re.I), "cancel_share"),
    (re.compile(r"accepted knowledge share", re.I), "accept_share"),
    (re.compile(r"rejected knowledge share", re.I), "reject_share"),
    (re.compile(r"Creating tool", re.I), "create_tool"),
    (re.compile(r"Admin sent message", re.I), "admin_message"),
    (re.compile(r"USPTO", re.I), "uspto_download"),
    (re.compile(r"Knowledge record created successfully", re.I), "create_knowledge"),
    (re.compile(r"find_knowledge_tool|Finding knowledge tool", re.I), "find_knowledge"),
    (re.compile(r"tool_result_filter", re.I), "tool_filter"),
]

# 无 user_id 但值得统计的行
ANONYMOUS_PATTERNS = [
    (re.compile(r"Provider initialized"), "system_start"),
    (re.compile(r"Browser initialized"), "system_start"),
    (re.compile(r"Agents initialized"), "system_start"),
    (re.compile(r"USPTO"), "uspto_download"),
]


def extract_user_id(message: str) -> Optional[str]:
    """从日志消息中提取 user_id"""
    for pattern in USER_ID_PATTERNS:
        m = pattern.search(message)
        if m:
            uid = m.group(1)
            # 过滤掉明显不是 user_id 的值
            if len(uid) > 2 and uid not in ("", "user:", "user_id:"):
                return uid.strip("',\"")
    return None


def classify_action(message: str) -> str:
    """根据日志消息分类操作类型"""
    for pattern, action in ACTION_RULES:
        if pattern.search(message):
            return action
    return "other"


def parse_log_line(line: str) -> Optional[dict]:
    """解析一行日志，返回结构化 dict 或 None"""
    m = LOG_LINE_RE.match(line.strip())
    if not m:
        return None

    timestamp_str, log_file, level, message = m.groups()
    try:
        ts = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None

    return {
        "timestamp": ts,
        "date": ts.strftime("%Y-%m-%d"),
        "time": ts.strftime("%H:%M:%S"),
        "log_file": log_file,
        "level": level,
        "message": message,
        "user_id": extract_user_id(message),
        "action": classify_action(message),
    }


def iter_log_files(log_dir: str = LOGS_DIR) -> List[str]:
    """列出所有日志文件"""
    files = []
    if not os.path.isdir(log_dir):
        print(f"WARNING: Log directory not found: {log_dir}")
        return files

    for fname in sorted(os.listdir(log_dir)):
        if fname.endswith(".log"):
            full = os.path.join(log_dir, fname)
            if os.path.isfile(full) and os.path.getsize(full) > 0:
                files.append(full)
    return files


def parse_logs(
    log_files: List[str],
    target_dates: Optional[set] = None,
) -> List[dict]:
    """解析所有日志文件，返回匹配日期的日志条目列表"""
    entries = []
    for filepath in log_files:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    entry = parse_log_line(line)
                    if entry is None:
                        continue
                    if target_dates and entry["date"] not in target_dates:
                        continue
                    entries.append(entry)
        except Exception as exc:
            print(f"  [skip] {os.path.basename(filepath)}: {exc}")

    entries.sort(key=lambda e: e["timestamp"])
    return entries


# ── 报表生成 ──────────────────────────────────────────────────────────────────


def report_daily_summary(entries: List[dict]):
    """按天汇总报表"""
    by_date: Dict[str, Dict[str, List[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for e in entries:
        uid = e["user_id"] or "(anonymous)"
        by_date[e["date"]][uid].append(e)

    for d in sorted(by_date):
        users = by_date[d]
        total_actions = sum(len(acts) for acts in users.values())
        active_users = sum(1 for uid in users if uid != "(anonymous)")

        print(f"\n{'='*70}")
        print(f"  {d}  |  {active_users} active users  |  {total_actions} actions")
        print(f"{'='*70}")

        # 按操作数排名
        user_ranking = sorted(
            users.items(),
            key=lambda kv: len(kv[1]),
            reverse=True,
        )

        for uid, acts in user_ranking:
            # 统计该用户的操作类型分布
            action_counts = defaultdict(int)
            for a in acts:
                action_counts[a["action"]] += 1

            action_summary = " ".join(
                f"{action}={count}" for action, count in sorted(action_counts.items())
            )

            first_ts = acts[0]["timestamp"].strftime("%H:%M")
            last_ts = acts[-1]["timestamp"].strftime("%H:%M")
            label = uid if uid != "(anonymous)" else "(no user_id)"
            print(f"\n  {label}")
            print(f"    {len(acts)} actions ({first_ts} ~ {last_ts}): {action_summary}")

            # 详情（每条操作）
            for a in acts:
                msg_preview = a["message"][:120]
                if len(a["message"]) > 120:
                    msg_preview += "..."
                print(f"    {a['time']}  [{a['action']}]  {msg_preview}")

    # 汇总趋势
    print(f"\n{'='*70}")
    print(f"  TREND")
    print(f"{'='*70}")
    dates_sorted = sorted(by_date)
    for d in dates_sorted:
        users = by_date[d]
        n_users = sum(1 for uid in users if uid != "(anonymous)")
        n_actions = sum(len(acts) for acts in users.values())
        bar = "█" * min(n_users, 40)
        print(f"  {d}  users={n_users:>3}  actions={n_actions:>4}  {bar}")


def report_user_summary(entries: List[dict]):
    """按用户汇总，显示每个用户在分析周期内的总活动"""
    by_user: Dict[str, List[dict]] = defaultdict(list)
    for e in entries:
        uid = e["user_id"] or "(anonymous)"
        by_user[uid].append(e)

    print(f"\n{'='*70}")
    print(f"  USER TOTALS (period aggregate)")
    print(f"{'='*70}")

    ranked = sorted(by_user.items(), key=lambda kv: len(kv[1]), reverse=True)
    for uid, acts in ranked:
        days = len(set(a["date"] for a in acts))
        action_counts = defaultdict(int)
        for a in acts:
            action_counts[a["action"]] += 1
        summary = " ".join(f"{k}={v}" for k, v in sorted(action_counts.items()))
        print(f"  {uid}")
        print(f"    {len(acts)} actions over {days} days: {summary}")


def resolve_dates(args: List[str]) -> set:
    """解析命令行参数为日期集合"""
    if not args:
        # 默认昨天
        yesterday = date.today() - timedelta(days=1)
        return {yesterday.strftime("%Y-%m-%d")}

    if args[0] == "--today":
        return {date.today().strftime("%Y-%m-%d")}

    if args[0] == "--last":
        n = int(args[1]) if len(args) > 1 else 7
        return {
            (date.today() - timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(n)
        }

    # 具体日期
    dates = set()
    for arg in args:
        try:
            datetime.strptime(arg, "%Y-%m-%d")
            dates.add(arg)
        except ValueError:
            print(f"Invalid date format: {arg} (expected YYYY-MM-DD)")
            sys.exit(1)

    if len(dates) == 1:
        return dates

    # 日期范围: 取最小到最大
    min_d = min(dates)
    max_d = max(dates)
    d = datetime.strptime(min_d, "%Y-%m-%d")
    end = datetime.strptime(max_d, "%Y-%m-%d")
    result = set()
    while d <= end:
        result.add(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return result


# ── 入口 ──────────────────────────────────────────────────────────────────────


def main():
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    target_dates = resolve_dates(args)
    print(f"Analyzing logs for: {', '.join(sorted(target_dates))}")

    log_files = iter_log_files()
    if not log_files:
        print("No .log files found in .logs/")
        return

    print(f"Found {len(log_files)} log files")
    entries = parse_logs(log_files, target_dates)

    if not entries:
        print("No matching log entries found for the specified date(s).")
        return

    print(f"Parsed {len(entries)} log entries")

    report_daily_summary(entries)
    report_user_summary(entries)

    print()


if __name__ == "__main__":
    main()
