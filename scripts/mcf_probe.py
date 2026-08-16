#!/usr/bin/env python3
"""Probe the ODP CPC Master Classification File (MCF) download chain.

The MCF maps patent document numbers to CPC symbols — the recall
expansion's CPC route builds a local CPC->patent index from it.  This
script verifies the ODP product detail response, extracts the file
download URIs, and inspects the newest file's XML structure so the
index builder can be written against the real format.

Usage (server, venv):
    python scripts/mcf_probe.py
"""
import io
import json
import os
import sys
import urllib.request
import zipfile

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
DETAIL_URL = "https://api.uspto.gov/api/v1/datasets/products/CPCMCPT"


def _get(url, headers=None, timeout=60):
    req = urllib.request.Request(url, headers=headers or {})
    req.add_header("User-Agent", "copiioai-mcf-probe/1.0")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:
        return None, f"{type(e).__name__}: {e}".encode()


def main() -> int:
    print(f"USPTO_API_KEY set: {bool(KEY)}")
    print()
    print("=== product detail CPCMCPT ===")
    status, body = _get(DETAIL_URL, {"X-API-KEY": KEY})
    print(f"HTTP {status}, {len(body)} bytes")
    if status != 200:
        print("ABORT — cannot fetch product detail")
        return 1
    detail = json.loads(body)
    product = (detail.get("bulkDataProductBag") or [{}])[0]
    print("product keys:", sorted(product.keys()))
    files = (product.get("productFileBag")
             or product.get("bulkDataFileBag")
             or product.get("fileBag") or [])
    print(f"file entries: {len(files)}")
    if files:
        print("first file entry keys:", sorted(files[0].keys()))
        print("newest 3 entries:")
        for f in files[-3:]:
            print(json.dumps(f, ensure_ascii=False)[:500])
    print()
    print("=== download newest file + inspect XML ===")
    # find the URI key whatever it is named, preferring the newest file
    uri = ""
    for f in reversed(files):
        for key in ("fileDownloadURI", "downloadURI", "fileDownloadUrl",
                    "fileDownloadUri"):
            v = f.get(key) or ""
            if v:
                uri = v if v.startswith("http") \
                    else "https://api.uspto.gov" + v
                break
        if uri:
            break
    if not uri:
        print("ABORT — no download URI found in file entries")
        return 1
    print(f"downloading: {uri}")
    status, body = _get(uri, {"X-API-KEY": KEY})
    print(f"HTTP {status}, {len(body)} bytes")
    if status != 200 or len(body) < 100:
        print("ABORT — download failed")
        return 1
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            names = z.namelist()
            print("zip entries:", names[:5])
            first = names[0]
            xml = z.read(first).decode("utf-8", errors="replace")
            print(f"--- first XML ({first}), {len(xml)} chars, head 2500:")
            print(xml[:2500])
    except Exception as e:
        print(f"zip parse failed: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
