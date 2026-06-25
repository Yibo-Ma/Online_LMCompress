"""Shared helpers for the download / normalize scripts.

Centralises three things every script needs:

* **hf-mirror by default** — many environments cannot reach ``huggingface.co``.
  We point ``HF_ENDPOINT`` at https://hf-mirror.com unless the caller overrides
  it (``--no-mirror`` / ``HF_ENDPOINT`` already set / ``--hf-endpoint``).
* **resumable, length-controlled HTTP** — :func:`http_download` uses HTTP Range
  so an interrupted download resumes from the current file size, and asking for
  *more* bytes later simply continues appending (never re-downloads).
* **progress manifests** — small JSON sidecars that record how much of a dataset
  has already been materialised, so a second run with a larger ``--limit``
  extends instead of restarting.

All paths are resolved relative to the repo root, so every script is meant to be
run from the repo root (``python scripts/<name>.py``).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
CKPT_ROOT = REPO_ROOT / "checkpoints"

DEFAULT_HF_MIRROR = "https://hf-mirror.com"


# --------------------------------------------------------------------------
# Hugging Face endpoint (mirror)
# --------------------------------------------------------------------------

def setup_hf_endpoint(use_mirror: bool = True, endpoint: Optional[str] = None) -> str:
    """Configure the HF endpoint for this process (and child libraries).

    Precedence: explicit ``endpoint`` > existing ``$HF_ENDPOINT`` > mirror/default.
    Returns the endpoint actually in effect.
    """
    if endpoint:
        chosen = endpoint
    elif os.environ.get("HF_ENDPOINT"):
        chosen = os.environ["HF_ENDPOINT"]
    elif use_mirror:
        chosen = DEFAULT_HF_MIRROR
    else:
        chosen = "https://huggingface.co"
    os.environ["HF_ENDPOINT"] = chosen
    return chosen


# --------------------------------------------------------------------------
# Sizes / pretty printing
# --------------------------------------------------------------------------

def human_bytes(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f}{unit}" if unit != "B" else f"{int(f)}B"
        f /= 1024
    return f"{f:.1f}TB"


def parse_size(s) -> Optional[int]:
    """Parse '10MB' / '500k' / '1_000_000' / int -> bytes. None passes through."""
    if s is None:
        return None
    if isinstance(s, int):
        return s
    s = str(s).strip().replace("_", "").upper()
    mult = 1
    for suf, m in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10),
                   ("T", 1 << 40), ("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10), ("B", 1)):
        if s.endswith(suf):
            mult = m
            s = s[: -len(suf)]
            break
    return int(float(s) * mult)


# --------------------------------------------------------------------------
# Resumable HTTP download with optional byte cap
# --------------------------------------------------------------------------

def http_download(
    url: str,
    dest: Path,
    limit_bytes: Optional[int] = None,
    chunk: int = 1 << 20,
    desc: str = "",
) -> int:
    """Download ``url`` -> ``dest`` resumably; stop after ``limit_bytes`` total.

    * If ``dest`` already holds ``cur`` bytes, request ``Range: bytes=cur-`` so an
      interrupted transfer (or a previous smaller ``--limit``) continues appending.
    * If the server ignores Range (returns 200) we restart from scratch.
    * Returns the final size of ``dest`` in bytes.
    """
    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cur = dest.stat().st_size if dest.exists() else 0

    if limit_bytes is not None and cur >= limit_bytes:
        return cur  # already have enough

    headers = {"Range": f"bytes={cur}-"} if cur else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        # 416 => our file is already complete; 200 w/ cur>0 => server ignored Range.
        if r.status_code == 416:
            return cur
        if cur and r.status_code == 200:
            dest.unlink(missing_ok=True)
            cur = 0
        r.raise_for_status()

        total = None
        clen = r.headers.get("Content-Length")
        if clen is not None:
            total = cur + int(clen)
        if limit_bytes is not None:
            total = min(total, limit_bytes) if total else limit_bytes

        mode = "ab" if cur else "wb"
        written = cur
        with open(dest, mode) as f:
            for block in r.iter_content(chunk_size=chunk):
                if not block:
                    continue
                if limit_bytes is not None and written + len(block) > limit_bytes:
                    block = block[: limit_bytes - written]
                f.write(block)
                written += len(block)
                _progress(desc or dest.name, written, total)
                if limit_bytes is not None and written >= limit_bytes:
                    break
    sys.stdout.write("\n")
    return written


def _progress(name: str, done: int, total: Optional[int]) -> None:
    if total:
        pct = 100.0 * done / total
        sys.stdout.write(f"\r  {name}: {human_bytes(done)}/{human_bytes(total)} ({pct:4.1f}%)")
    else:
        sys.stdout.write(f"\r  {name}: {human_bytes(done)}")
    sys.stdout.flush()


# --------------------------------------------------------------------------
# Manifests (progress sidecars)
# --------------------------------------------------------------------------

def load_progress(d: Path) -> Dict:
    p = Path(d) / "_progress.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_progress(d: Path, **kv) -> None:
    p = Path(d) / "_progress.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = load_progress(d)
    cur.update(kv)
    p.write_text(json.dumps(cur, indent=2, ensure_ascii=False), encoding="utf-8")
