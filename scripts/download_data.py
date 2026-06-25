#!/usr/bin/env python3
"""Download text / image / audio datasets into ``data/`` (catalog-driven).

Run from the repo root.

Examples
--------
    # one dataset, cap the amount fetched
    python scripts/download_data.py --dataset pile_of_law_eurlex --limit 20MB
    python scripts/download_data.py --dataset kodak                 # all 24 imgs
    python scripts/download_data.py --dataset ljspeech --limit 50   # 50 clips

    # whole modality / everything
    python scripts/download_data.py --modality text --limit 20MB
    python scripts/download_data.py --all

    # list the catalog (key | gain | domain)
    python scripts/download_data.py --list

Length control + resume
-----------------------
``--limit`` means **bytes** for text and **item count** for image/audio.  Every
downloader is resumable and *extends* rather than restarts: re-running with a
larger ``--limit`` continues from where the last run stopped (HF streams skip the
rows already written; URL files resume via HTTP Range).  Defaults to mirror
(hf-mirror.com); pass ``--no-mirror`` or ``--hf-endpoint URL`` to change.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (  # noqa: E402
    DATA_ROOT, http_download, human_bytes, parse_size,
    load_progress, save_progress, setup_hf_endpoint,
)
import catalog  # noqa: E402

SHARD_BYTES = 50 << 20          # 50 MB text shards
DEFAULT_TEXT_LIMIT = 20 << 20   # 20 MB
DEFAULT_ITEM_LIMIT = 50         # images / audio clips


# ---------------------------------------------------------------------------
# TEXT
# ---------------------------------------------------------------------------

def _external_data_guard(out: Path, force: bool) -> bool:
    """Refuse to write into a dir holding data this tool did not create.

    Externally-provided datasets have part-*.txt but no _progress.json; appending
    to / overwriting them would corrupt the user's data.  ``--force`` overrides.
    """
    if force or not out.exists():
        return True
    if list(out.glob("part-*.txt")) and not (out / "_progress.json").exists():
        print(f"  [{out.name}] REFUSING: {out} already holds data not created by this "
              f"downloader (no _progress.json). Use --force, or download into a clean dir.")
        return False
    return True


def _row_text(row: dict, fields) -> str:
    if fields:
        parts = [str(row[f]) for f in fields if row.get(f) not in (None, "")]
    else:  # concatenate every string-valued field
        parts = [v for v in row.values() if isinstance(v, str) and v]
    return "\n".join(parts)


def download_text_hf(key: str, spec: dict, limit_bytes: int, force: bool = False) -> None:
    from datasets import load_dataset
    out = DATA_ROOT / "text" / "raw" / key
    if not _external_data_guard(out, force):
        return
    out.mkdir(parents=True, exist_ok=True)
    prog = load_progress(out)
    rows, written, shard = prog.get("rows", 0), prog.get("bytes", 0), prog.get("shard", 0)
    if written >= limit_bytes:
        print(f"  [{key}] already at {human_bytes(written)} (>= limit); nothing to do")
        return

    streaming = spec.get("streaming", True)
    lk = dict(spec.get("load_kwargs", {}))
    mode = "streaming" if streaming else "downloading"
    print(f"  [{key}] {mode} {spec['hf_id']} ({spec.get('config') or 'default'})"
          f" — have {human_bytes(written)}, target {human_bytes(limit_bytes)}")
    ds = load_dataset(spec["hf_id"], spec.get("config"), split=spec["split"],
                      streaming=streaming, **lk)

    man = open(out / "manifest.jsonl", "a", encoding="utf-8")
    part = open(out / f"part-{shard:05d}.txt", "a", encoding="utf-8")
    shard_sz = (out / f"part-{shard:05d}.txt").stat().st_size
    state = {"rows": rows, "written": written, "shard": shard, "sz": shard_sz, "part": part}

    def emit(row) -> bool:
        """Write one row; return True once the byte target is reached."""
        text = _row_text(row, spec.get("fields"))
        if not text:
            state["rows"] += 1
            return False
        if state["sz"] >= SHARD_BYTES:                       # roll to a new shard
            state["part"].close(); state["shard"] += 1; state["sz"] = 0
            state["part"] = open(out / f"part-{state['shard']:05d}.txt", "a", encoding="utf-8")
        blob = text + "\n"
        nbytes = len(blob.encode("utf-8"))
        state["part"].write(blob)
        man.write(json.dumps({"source": spec["hf_id"], "config": spec.get("config"),
                              "split": spec["split"], "row": state["rows"], "dataset": key,
                              "shard": state["shard"], "bytes": nbytes}, ensure_ascii=False) + "\n")
        state["rows"] += 1; state["written"] += nbytes; state["sz"] += nbytes
        if state["rows"] % 200 == 0:
            sys.stdout.write(f"\r    {human_bytes(state['written'])} / {human_bytes(limit_bytes)}")
            sys.stdout.flush()
        return state["written"] >= limit_bytes

    start = rows                                            # rows already written -> skip them
    if streaming:
        seen = 0
        for row in ds:
            if seen < start:
                seen += 1
                continue
            seen += 1
            if emit(row):
                break
    else:                                                   # random-access Dataset
        for i in range(start, len(ds)):
            if emit(ds[i]):
                break

    state["part"].close(); man.close()
    save_progress(out, rows=state["rows"], bytes=state["written"], shard=state["shard"],
                  source=spec["hf_id"], gain=spec.get("gain"))
    print(f"\n  [{key}] -> {out}  ({human_bytes(state['written'])}, {state['rows']} docs)")


def download_text_url(key: str, spec: dict, limit_bytes: int, force: bool = False) -> None:
    out = DATA_ROOT / "text" / "raw" / key
    if not _external_data_guard(out, force):
        return
    out.mkdir(parents=True, exist_ok=True)
    src = out / ("_source" + Path(spec["url"]).suffix)
    print(f"  [{key}] fetching {spec['url']}")
    http_download(spec["url"], src, desc=f"{key} source")     # full file, resumable

    member = spec.get("member")
    raw = _read_member_bytes(src, member, limit_bytes)
    target = out / "part-00000.txt"
    target.write_bytes(raw)
    save_progress(out, bytes=len(raw), source=spec["url"], gain=spec.get("gain"))
    print(f"  [{key}] -> {target}  ({human_bytes(len(raw))})")


def _read_member_bytes(src: Path, member, limit_bytes: int) -> bytes:
    """Read up to limit_bytes from a (possibly zipped) source file."""
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            name = member or zf.namelist()[0]
            with zf.open(name) as f:
                return f.read(limit_bytes)
    with open(src, "rb") as f:
        return f.read(limit_bytes)


# ---------------------------------------------------------------------------
# IMAGE
# ---------------------------------------------------------------------------

def download_image_hf(key: str, spec: dict, limit_n: int) -> None:
    from datasets import load_dataset
    out = DATA_ROOT / "image" / key
    out.mkdir(parents=True, exist_ok=True)
    have = len(list(out.glob("img_*.png")))
    if have >= limit_n:
        print(f"  [{key}] already have {have} images (>= limit)"); return
    print(f"  [{key}] streaming {spec['hf_id']} — have {have}, target {limit_n}")
    ds = load_dataset(spec["hf_id"], spec.get("config"), split=spec["split"], streaming=True)
    idx = 0
    for row in ds:
        if idx < have:
            idx += 1; continue
        img = row[spec.get("image_key", "image")]
        if isinstance(img, dict) and img.get("bytes"):
            from PIL import Image
            img = Image.open(io.BytesIO(img["bytes"]))
        img.convert("RGB").save(out / f"img_{idx:06d}.png")
        idx += 1
        sys.stdout.write(f"\r    {idx}/{limit_n}"); sys.stdout.flush()
        if idx >= limit_n:
            break
    print(f"\n  [{key}] -> {out}  ({idx} images)")


def download_image_url_files(key: str, spec: dict, limit_n: int) -> None:
    out = DATA_ROOT / "image" / key
    out.mkdir(parents=True, exist_ok=True)
    urls = spec["urls"][:limit_n]
    for u in urls:
        dest = out / Path(u).name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        http_download(u, dest, desc=dest.name)
    print(f"  [{key}] -> {out}  ({len(list(out.glob('*')))} files)")


def download_image_url_zip(key: str, spec: dict, limit_n: int) -> None:
    out = DATA_ROOT / "image" / key
    out.mkdir(parents=True, exist_ok=True)
    src = out / ("_source.zip")
    print(f"  [{key}] fetching {spec['url']} (large)")
    http_download(spec["url"], src, desc=f"{key} source")
    with zipfile.ZipFile(src) as zf:
        members = [m for m in zf.namelist()
                   if m.lower().endswith((".png", ".bmp", ".jpg", ".jpeg"))][:limit_n]
        for m in members:
            dest = out / Path(m).name
            if not dest.exists():
                dest.write_bytes(zf.read(m))
    print(f"  [{key}] -> {out}  ({len(list(out.glob('*.*')))} images)")


# ---------------------------------------------------------------------------
# AUDIO
# ---------------------------------------------------------------------------

def download_audio_hf(key: str, spec: dict, limit_n: int) -> None:
    from datasets import load_dataset
    out = DATA_ROOT / "audio" / key
    out.mkdir(parents=True, exist_ok=True)

    if spec.get("as_parquet"):                   # peoples_speech: keep raw bytes
        from datasets import Audio
        import pandas as pd  # noqa: F401
        dst = out / f"{spec.get('config','data')}-{spec['split']}.parquet"
        if dst.exists():
            print(f"  [{key}] {dst.name} exists; skip"); return
        ds = load_dataset(spec["hf_id"], spec.get("config"), split=spec["split"])
        ds = ds.cast_column(spec.get("audio_key", "audio"), Audio(decode=False))
        if limit_n:
            ds = ds.select(range(min(limit_n, len(ds))))
        ds.to_parquet(str(dst))
        print(f"  [{key}] -> {dst}  ({len(ds)} clips)")
        return

    import soundfile as sf
    import numpy as np
    have = len(list(out.glob("clip_*.wav")))
    if have >= limit_n:
        print(f"  [{key}] already have {have} clips (>= limit)"); return
    print(f"  [{key}] streaming {spec['hf_id']} — have {have}, target {limit_n}")
    ds = load_dataset(spec["hf_id"], spec.get("config"), split=spec["split"], streaming=True)
    idx = 0
    for row in ds:
        if idx < have:
            idx += 1; continue
        a = row[spec.get("audio_key", "audio")]
        sf.write(out / f"clip_{idx:06d}.wav",
                 np.asarray(a["array"]), int(a["sampling_rate"]))
        idx += 1
        sys.stdout.write(f"\r    {idx}/{limit_n}"); sys.stdout.flush()
        if idx >= limit_n:
            break
    print(f"\n  [{key}] -> {out}  ({idx} clips)")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def fetch(key: str, modality: str, spec: dict, limit, force: bool = False) -> None:
    src = spec["source"]
    if src == "manual":
        print(f"  [{key}] MANUAL — {spec.get('note', '')}")
        return
    if modality == "text":
        lim = parse_size(limit) if limit is not None else DEFAULT_TEXT_LIMIT
        (download_text_hf if src == "hf" else download_text_url)(key, spec, lim, force)
    elif modality == "image":
        n = int(limit) if limit is not None else DEFAULT_ITEM_LIMIT
        {"hf": download_image_hf, "url_files": download_image_url_files,
         "url": download_image_url_zip}[src](key, spec, n)
    elif modality == "audio":
        n = int(limit) if limit is not None else DEFAULT_ITEM_LIMIT
        download_audio_hf(key, spec, n)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Download datasets into data/ (catalog-driven)")
    p.add_argument("--dataset", nargs="*", help="dataset key(s) from the catalog")
    p.add_argument("--modality", choices=["text", "image", "audio"], help="download a whole modality")
    p.add_argument("--all", action="store_true", help="download everything (auto sources)")
    p.add_argument("--limit", default=None,
                   help="text: bytes (e.g. 20MB); image/audio: item count")
    p.add_argument("--list", action="store_true", help="print the catalog and exit")
    p.add_argument("--force", action="store_true",
                   help="write even into a dir holding externally-provided data (DANGER)")
    p.add_argument("--no-mirror", action="store_true", help="use huggingface.co, not hf-mirror")
    p.add_argument("--hf-endpoint", default=None, help="explicit HF endpoint URL")
    args = p.parse_args()

    if args.list:
        for modality, group in catalog.ALL_DATA.items():
            print(f"\n[{modality}]")
            for k, s in group.items():
                tag = "" if s["source"] != "manual" else "  (manual)"
                print(f"  {k:<22} gain={s.get('gain','?'):<4} {s.get('domain','')}{tag}")
        return 0

    ep = setup_hf_endpoint(use_mirror=not args.no_mirror, endpoint=args.hf_endpoint)
    print(f"HF endpoint: {ep}")

    # Build the work list
    work = []
    if args.all:
        for modality, group in catalog.ALL_DATA.items():
            work += [(k, modality, s) for k, s in group.items()]
    elif args.modality:
        group = catalog.ALL_DATA[args.modality]
        work += [(k, args.modality, s) for k, s in group.items()]
    elif args.dataset:
        for k in args.dataset:
            modality, s = catalog.find_dataset(k)
            if s is None:
                print(f"  [{k}] unknown dataset key — see --list"); continue
            work.append((k, modality, s))
    else:
        p.error("specify --dataset KEY..., --modality MOD, or --all (or --list)")

    for k, modality, s in work:
        try:
            fetch(k, modality, s, args.limit, args.force)
        except Exception as e:
            print(f"  [{k}] FAILED: {type(e).__name__}: {e}")
            if s.get("status") == "verify":
                print(f"        (catalog hint: {s.get('note','check hf id/config/fields')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
