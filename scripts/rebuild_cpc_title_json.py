#!/usr/bin/env python3
"""Regenerate the tracked CPC title JSON files from the scheme zip.

CPC indexing codes (classification-item additional-only="true") are
excluded from the runtime corpora: their titles restate sensor/control
vocabulary across unrelated domains, so they dominate the title-cosine
ranking for carrier-word queries — observed 2026-09-07, a wafer
process-control question matched B60G2800/F01N2240/B65H2553/F25J2280
index codes whose titles literally say the carrier terms (thermocouple,
"a plasma reactor", adaptive control...), crowding out the real domain
codes (H01L/C23C/...).  Every kept entry carries additional_only=False.

Usage:
    python scripts/rebuild_cpc_title_json.py [--zip path/to/CPCSchemeXML*.zip]

Run after updating the scheme zip.  The titles JSON is git-tracked;
the server-side vector cache (scripts/build_cpc_vectors.py
--groups main|sub) must then be rebuilt so json and .npy stay aligned.
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sources.long_task.cpc_semantic import (
    CPC_DATA_DIR,
    CPC_TITLES_JSON,
    CPC_TITLES_SUB_JSON,
    parse_cpc_zip,
)


def _latest_zip(data_dir: str) -> str:
    matches = sorted(glob.glob(os.path.join(data_dir, "CPCSchemeXML*.zip")))
    if not matches:
        raise SystemExit(
            f"No CPCSchemeXML*.zip under {data_dir} — pass --zip explicitly")
    return matches[-1]


def _write(path: str, entries: list, label: str, compact: bool) -> None:
    # Strip the parser's additional_only diagnostic key — the tracked
    # JSON schema carries code/title only.
    payload = [
        {"code": e["code"], "title": e["title"]} for e in entries]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            payload, f, ensure_ascii=False, indent=None if compact else 1)
    print(f"{label}: {len(entries)} entries -> {path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip", default=None,
        help="CPC scheme zip (default: newest CPCSchemeXML*.zip under "
             "data/cpc)")
    parser.add_argument(
        "--keep-indexing", action="store_true",
        help="keep CPC indexing codes (diagnostics; not the runtime "
             "corpus default)")
    args = parser.parse_args()

    zip_path = args.zip or _latest_zip(CPC_DATA_DIR)
    exclude = not args.keep_indexing
    _write(CPC_TITLES_JSON,
           parse_cpc_zip(zip_path, main_groups_only=True,
                         exclude_indexing=exclude),
           "main", compact=False)
    _write(CPC_TITLES_SUB_JSON,
           parse_cpc_zip(zip_path, main_groups_only=False,
                         exclude_indexing=exclude),
           "sub", compact=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
