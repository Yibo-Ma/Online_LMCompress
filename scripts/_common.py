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
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def write_download_status(modality_root: Path, ok, failed, skipped=None) -> None:
    """Record per-modality download outcome so setup.py can report failures."""
    p = Path(modality_root) / "_download_status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"ok": list(ok), "failed": list(failed),
                             "skipped": list(skipped or [])}, indent=2, ensure_ascii=False),
                 encoding="utf-8")


def read_download_status(modality_root: Path) -> Dict:
    p = Path(modality_root) / "_download_status.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


# --------------------------------------------------------------------------
# Safety + archive helpers (shared by the three downloaders)
# --------------------------------------------------------------------------

def external_guard(out: Path, force: bool) -> bool:
    """Refuse to write into a dir holding data this tool did not create.

    Tool-made dirs carry a ``_progress.json`` marker; externally-provided data
    does not, so appending/overwriting would corrupt it.  ``--force`` overrides.
    """
    out = Path(out)
    if force or not out.exists():
        return True
    has_files = any(p.is_file() and p.name != "_progress.json" for p in out.rglob("*"))
    if has_files and not (out / "_progress.json").exists():
        print(f"  [{out.name}] REFUSING: {out} holds data not created by this tool "
              f"(no _progress.json). Use --force, or download into a clean dir.")
        return False
    return True


def read_zip_member(src: Path, member: Optional[str], limit_bytes: Optional[int] = None) -> bytes:
    """Read up to ``limit_bytes`` from one member of a zip file."""
    with zipfile.ZipFile(src) as zf:
        name = member or zf.namelist()[0]
        with zf.open(name) as f:
            return f.read(limit_bytes) if limit_bytes else f.read()


def is_streamable_tar(url: str) -> bool:
    """True for tar(.gz/.bz2) — can be streamed and stopped early.

    zip cannot: its central directory is at the end, so extracting any member
    requires the whole file.
    """
    u = url.lower()
    return u.endswith((".tar", ".tgz", ".tar.gz", ".gz", ".tar.bz2", ".bz2"))


def stream_extract_tar(url: str, out_dir: Path, limit: int,
                       exts: Tuple[str, ...], prefix: str, desc: str = "") -> int:
    """Stream a tar(.gz/.bz2) over HTTP, extract up to ``limit`` files matching
    ``exts``, then stop — so only the bytes up to the limit-th file are downloaded.

    Files are renamed ``prefix_NNNNN.ext``.  Resumable: already-extracted files are
    skipped (a re-run re-streams from the start but only down to the new limit).
    Returns the number of files now present.
    """
    import requests

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    have = len(list(out_dir.glob(f"{prefix}_*")))
    if have >= limit:
        return have

    u = url.lower()
    mode = "r|bz2" if u.endswith(".bz2") else ("r|gz" if u.endswith((".gz", ".tgz")) else "r|")
    n, seen = have, 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        r.raw.decode_content = True                       # undo any HTTP content-encoding
        with tarfile.open(fileobj=r.raw, mode=mode) as tf:
            for m in tf:
                if n >= limit:
                    break                                 # closes the connection -> stops downloading
                if not (m.isfile() and m.name.lower().endswith(exts)):
                    continue
                if seen < have:                           # resume: skip already-extracted
                    seen += 1
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                (out_dir / f"{prefix}_{n:05d}{Path(m.name).suffix.lower()}").write_bytes(f.read())
                n += 1
                sys.stdout.write(f"\r  {desc or prefix}: {n}/{limit} files")
                sys.stdout.flush()
    sys.stdout.write("\n")
    return n


def extract_media(archive: Path, out_dir: Path, limit: int,
                  exts: Tuple[str, ...], prefix: str) -> int:
    """Extract up to ``limit`` files matching ``exts`` from a zip/tar(.gz/.bz2)
    archive into ``out_dir``, renamed ``prefix_NNNNN.ext``.  Returns the count.

    Resumable: files already present (by index) are skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    have = len(list(out_dir.glob(f"{prefix}_*")))
    n = have
    al = str(archive).lower()
    if al.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            names = [m for m in zf.namelist()
                     if m.lower().endswith(exts) and not m.endswith("/")]
            for m in names[have:limit]:
                (out_dir / f"{prefix}_{n:05d}{Path(m).suffix.lower()}").write_bytes(zf.read(m))
                n += 1
    else:
        mode = "r:bz2" if al.endswith(".bz2") else ("r:gz" if al.endswith((".gz", ".tgz")) else "r:*")
        seen = 0
        with tarfile.open(archive, mode) as tf:
            for m in tf:
                if n >= limit:
                    break
                if m.isfile() and m.name.lower().endswith(exts):
                    if seen < have:                  # skip already-extracted
                        seen += 1
                        continue
                    f = tf.extractfile(m)
                    if f is not None:
                        (out_dir / f"{prefix}_{n:05d}{Path(m.name).suffix.lower()}").write_bytes(f.read())
                        n += 1
    return n
