#!/usr/bin/env python3
"""Build the local CPC->patent index from the USPTO MCF bulk data.

One-time + monthly refresh step: downloads the newest
US_Grant_CPC_MCF_Text_*.zip via the ODP datasets API (existing
USPTO_API_KEY), parses the B records, and writes a compact sqlite index
at data/cpc/cpc_index.db mapping canonical CPC codes to patent numbers.
fetch_by_cpc() then resolves CPC recall locally and fetches the
matching patents' metadata through the number search.

MCF text layout (fixed-width, one record per line):
  A ... — scheme-version records (skipped)
  B <9-char record index><patent in an 8-char field><CPC symbol>
    <version> 0 0

Usage (server, venv):
    python scripts/build_cpc_index.py            # download + build
    python scripts/build_cpc_index.py --zip <file>  # build from a
                                                  # downloaded zip
"""
import argparse
import json
import os
import re
import sqlite3
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

PRODUCT_ID = "CPCMCPT"
DETAIL_URL = f"https://api.uspto.gov/api/v1/datasets/products/{PRODUCT_ID}"
CPC_DATA_DIR = os.getenv("CPC_DATA_DIR", "data/cpc")
INDEX_DB = os.path.join(CPC_DATA_DIR, "cpc_index.db")

_SYMBOL_RE = re.compile(r"^([A-HY]\d{2}[A-Z]\d{0,4})(\d{0,4})/(\d{0,4})$")
# B records: a 9-char record index (ignored) followed by the patent
# number in an 8-char field — e.g. "21849611012650000" is patent
# 12,650,000.  Short patents may be space- or zero-padded.
_B_LINE_RE = re.compile(
    r"^B(?P<idx>[\d ]{9})(?P<patent>[\d ]{8})"
    r"(?P<symbol>[A-HY]\d{2}[A-Z][\w\s]*?/\d{1,4})")


def _normalize_symbol(raw):
    """Canonicalize a raw MCF symbol ('E02F   3/844' -> 'E02F3/844').

    None when the text does not look like a CPC code.
    """
    if not isinstance(raw, str):
        return None
    collapsed = raw.replace(" ", "").strip()
    m = _SYMBOL_RE.match(collapsed)
    if not m:
        return None
    return f"{m.group(1) + m.group(2)}/{m.group(3)}"


def _parse_mcf_line(line: str):
    """Parse one MCF record line -> (patent, canonical_cpc).

    None for A records and anything unparseable.  The patent number is
    an 8-char field; leading zeros/spaces are stripped.  Never raises.
    """
    if not isinstance(line, str) or not line.startswith("B"):
        return None
    m = _B_LINE_RE.match(line)
    if not m:
        return None
    symbol = _normalize_symbol(m.group("symbol"))
    if not symbol:
        return None
    return str(int(m.group("patent"))), symbol


def build_index(zip_path: str, db_path: str, batch: int = 20000) -> dict:
    """Parse a downloaded MCF zip into the sqlite index.

    Returns stats {patents, pairs, chunks}.  Raises on unreadable
    input; the index is written incrementally so a mid-build failure
    leaves no partial table behind (the table is recreated on each
    run).
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS cpc_patents")
        conn.execute(
            "CREATE TABLE cpc_patents "
            "(cpc TEXT NOT NULL, patent TEXT NOT NULL, "
            "PRIMARY KEY (cpc, patent))")
        buffer: list = []
        patents = set()
        pairs = 0
        chunks = 0

        def flush():
            nonlocal pairs
            if buffer:
                conn.executemany(
                    "INSERT OR IGNORE INTO cpc_patents VALUES (?, ?)",
                    buffer)
                pairs += len(buffer)
                buffer.clear()
                conn.commit()

        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if not name.endswith(".txt"):
                    continue
                chunks += 1
                with z.open(name) as fh:
                    for raw in fh:
                        try:
                            line = raw.decode("utf-8", errors="replace")
                        except Exception:
                            continue
                        parsed = _parse_mcf_line(line)
                        if not parsed:
                            continue
                        patent, cpc = parsed
                        patents.add(patent)
                        buffer.append((cpc, patent))
                        if len(buffer) >= batch:
                            flush()
        flush()
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cpc "
                     "ON cpc_patents(cpc)")
        conn.commit()
    finally:
        conn.close()
    return {"patents": len(patents), "pairs": pairs, "chunks": chunks}


def _fetch_newest_text_entry(api_key: str) -> dict:
    """Return the newest *_Text_* file entry from the ODP product detail."""
    req = urllib.request.Request(DETAIL_URL,
                                 headers={"X-API-KEY": api_key})
    req.add_header("User-Agent", "copiioai-cpc-index/1.0")
    with urllib.request.urlopen(req, timeout=60) as r:
        detail = json.loads(r.read())
    product = (detail.get("bulkDataProductBag") or [{}])[0]
    bag = (product.get("productFileBag") or {}).get("fileDataBag") or []
    for entry in bag:
        if "_Text_" in (entry.get("fileName") or ""):
            return entry
    return {}


def _download(url: str, api_key: str, dest: str) -> None:
    """Stream a file to disk in 1MB blocks (the zip is ~373MB)."""
    req = urllib.request.Request(url, headers={"X-API-KEY": api_key})
    req.add_header("User-Agent", "copiioai-cpc-index/1.0")
    total = 0
    with urllib.request.urlopen(req, timeout=120) as r, \
            open(dest, "wb") as out:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            out.write(block)
            total += len(block)
            if total % (64 << 20) < (1 << 20):
                print(f"  downloaded {total >> 20}MB…", flush=True)
    print(f"  downloaded {total} bytes -> {dest}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", help="build from an already-downloaded zip")
    args = parser.parse_args()

    if args.zip:
        zip_path = args.zip
    else:
        api_key = os.getenv("USPTO_API_KEY", "")
        if not api_key:
            print("USPTO_API_KEY not set — cannot download the MCF")
            return 1
        entry = _fetch_newest_text_entry(api_key)
        uri = entry.get("fileDownloadURI") or ""
        if not uri:
            print("no _Text_ file entry found in product detail")
            return 1
        zip_path = os.path.join(CPC_DATA_DIR, entry.get("fileName", "mcf.zip"))
        if os.path.exists(zip_path):
            print(f"using cached zip: {zip_path}")
        else:
            print(f"downloading {entry.get('fileSize', '?')} bytes: {uri}")
            _download(uri, api_key, zip_path)

    print(f"building index from {zip_path} -> {INDEX_DB}")
    stats = build_index(zip_path, INDEX_DB)
    print(f"done — patents={stats['patents']} pairs={stats['pairs']} "
          f"chunks={stats['chunks']} -> {INDEX_DB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
