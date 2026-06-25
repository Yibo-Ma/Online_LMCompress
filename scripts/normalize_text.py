#!/usr/bin/env python3
"""Normalize raw text datasets so they survive a tokenizer round-trip losslessly.

The compressor tokenizes text then reconstructs it via ``tokenizer.decode``; a few
characters a tokenizer cannot represent would otherwise break the byte-exact
round-trip.  This script applies the same ``encode -> decode`` normalization the
compressor would see, once, ahead of time.

It batch-processes every dataset under ``data/text/raw/`` and writes the result to
``data/text/normalized/<dataset>/`` (part-*.txt + manifest.jsonl + a summary),
matching the layout ``download_data.py`` produces and ``eval_online.py`` reads.

Run from the repo root:

    python scripts/normalize_text.py                       # all datasets, Qwen2.5-0.5B
    python scripts/normalize_text.py --dataset pile_of_law_eurlex medal
    python scripts/normalize_text.py --model checkpoints/Qwen3-1.7B-Base --force
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import DATA_ROOT, human_bytes  # noqa: E402

RAW_ROOT = DATA_ROOT / "text" / "raw"
NORM_ROOT = DATA_ROOT / "text" / "normalized"
DEFAULT_MODEL = "checkpoints/Qwen2.5-0.5B"
SHARD_BYTES = 50 << 20
CHUNK_BYTES = 2048              # reporting granularity (matches compressor segment size)


def roundtrip(text: str, tok) -> str:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def read_raw(ds_dir: Path, max_bytes) -> str:
    """Concatenate part-*.txt (in order), capped at max_bytes."""
    buf, total = [], 0
    for part in sorted(ds_dir.glob("part-*.txt")):
        data = part.read_bytes()
        if max_bytes is not None and total + len(data) > max_bytes:
            data = data[: max_bytes - total]
        buf.append(data)
        total += len(data)
        if max_bytes is not None and total >= max_bytes:
            break
    return b"".join(buf).decode("utf-8", errors="ignore")


def write_shards(text: str, out_dir: Path) -> int:
    """Write text into part-*.txt shards of <= SHARD_BYTES; return #shards."""
    for old in out_dir.glob("part-*.txt"):
        old.unlink()
    data = text.encode("utf-8")
    n = max(1, (len(data) + SHARD_BYTES - 1) // SHARD_BYTES)
    man = open(out_dir / "manifest.jsonl", "w", encoding="utf-8")
    for s in range(n):
        chunk = data[s * SHARD_BYTES:(s + 1) * SHARD_BYTES]
        (out_dir / f"part-{s:05d}.txt").write_bytes(chunk)
        man.write(json.dumps({"shard": s, "bytes": len(chunk)}) + "\n")
    man.close()
    return n


def normalize_dataset(key: str, tok, max_bytes, force: bool) -> dict:
    src = RAW_ROOT / key
    out = NORM_ROOT / key
    summary_path = out / "normalize_summary.json"
    if summary_path.exists() and not force:
        prev = json.loads(summary_path.read_text(encoding="utf-8"))
        print(f"  [{key}] already normalized ({human_bytes(prev.get('output_bytes', 0))})"
              f" — use --force to redo")
        return prev
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    text = read_raw(src, max_bytes)
    in_bytes = len(text.encode("utf-8"))
    print(f"  [{key}] {human_bytes(in_bytes)} in -> normalizing ...")

    norm = roundtrip(text, tok)
    # idempotency guard: a second pass must be a no-op
    if roundtrip(norm, tok) != norm:
        norm = roundtrip(norm, tok)
    out_bytes = len(norm.encode("utf-8"))
    changed = sum(
        1 for i in range(0, len(text), CHUNK_BYTES)
        if text[i:i + CHUNK_BYTES] != norm[i:i + CHUNK_BYTES]
    )
    shards = write_shards(norm, out)

    summary = {
        "dataset": key, "model": getattr(tok, "name_or_path", "?"),
        "input_dir": str(src), "output_dir": str(out),
        "shard_bytes": SHARD_BYTES, "roundtrip_chunk_bytes": CHUNK_BYTES,
        "max_input_bytes": max_bytes, "input_bytes": in_bytes, "output_bytes": out_bytes,
        "approx_changed_chunks": changed, "shards": shards,
        "seconds": round(time.time() - t0, 2),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    flag = "byte-identical" if changed == 0 else f"~{changed} chunks adjusted"
    print(f"  [{key}] -> {out}  ({human_bytes(out_bytes)}, {flag}, {summary['seconds']}s)")
    return summary


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Normalize raw text datasets (tokenizer round-trip)")
    p.add_argument("--dataset", nargs="*", help="dataset key(s); default: all under data/text/raw")
    p.add_argument("--model", default=DEFAULT_MODEL, help="tokenizer path/name")
    p.add_argument("--max-bytes", type=int, default=None, help="cap input bytes per dataset")
    p.add_argument("--force", action="store_true", help="re-normalize even if a summary exists")
    args = p.parse_args()

    if not RAW_ROOT.exists():
        print(f"No raw text at {RAW_ROOT}. Run scripts/download_data.py first."); return 1

    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    keys = args.dataset or sorted(d.name for d in RAW_ROOT.iterdir() if d.is_dir())
    if not keys:
        print(f"No datasets under {RAW_ROOT}."); return 1

    NORM_ROOT.mkdir(parents=True, exist_ok=True)
    index = []
    for k in keys:
        if not (RAW_ROOT / k).is_dir():
            print(f"  [{k}] no raw dir; skip"); continue
        index.append(normalize_dataset(k, tok, args.max_bytes, args.force))
    (NORM_ROOT / "normalize_summary.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in index), encoding="utf-8")
    print(f"\nNormalized {len(index)} dataset(s) -> {NORM_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
