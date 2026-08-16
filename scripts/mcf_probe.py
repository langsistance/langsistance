#!/usr/bin/env python3
"""Probe which URL serves the CPC Master Classification File (MCF).

The MCF maps patent document numbers to CPC symbols — the recall
expansion's CPC route builds a local CPC->patent index from it.  The
USPTO bulkdata paths moved onto the Open Data Portal in 2026; this
script tests the candidate routes on the server (which holds the ODP
API key).  Run once and share the output.

Usage (server, venv):
    python scripts/mcf_probe.py
"""
import json
import os
import sys
import urllib.request

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

KEY = os.getenv("USPTO_API_KEY", "")


def _head(url, headers=None, timeout=30, with_body=False):
    req = urllib.request.Request(url, headers=headers or {})
    req.add_header("User-Agent", "copiioai-mcf-probe/1.0")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(400) if with_body else b""
            return (f"HTTP {r.status} type={r.headers.get('Content-Type')} "
                    f"len={r.headers.get('Content-Length', '?')}"
                    + (f" body={body[:160]!r}" if body else ""))
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} ({e.reason})"
    except Exception as e:
        return f"FAIL {type(e).__name__}: {e}"


def main() -> int:
    print(f"USPTO_API_KEY set: {bool(KEY)}")
    print()
    print("=== C: ODP datasets products/search (correct endpoint) ===")
    print(_head(
        "https://api.uspto.gov/api/v1/datasets/products/search"
        "?q=CPCMCPT",
        headers={"X-API-KEY": KEY}, with_body=True))
    print()
    print("=== D: ODP datasets products/search (cpc keyword) ===")
    print(_head(
        "https://api.uspto.gov/api/v1/datasets/products/search?q=CPC",
        headers={"X-API-KEY": KEY}, with_body=True))
    print()
    print("=== E: ODP product detail (CPCMCPT) ===")
    print(_head(
        "https://api.uspto.gov/api/v1/datasets/products/CPCMCPT",
        headers={"X-API-KEY": KEY}, with_body=True))
    print()
    print("=== F: ODP product files (CPCMCPT) ===")
    print(_head(
        "https://api.uspto.gov/api/v1/datasets/products/files/CPCMCPT",
        headers={"X-API-KEY": KEY}, with_body=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
