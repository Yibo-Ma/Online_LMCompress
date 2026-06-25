#!/usr/bin/env python3
"""Download model weights into ``checkpoints/`` (catalog-driven).

Run from the repo root.

Examples
--------
    python scripts/download_models.py --model qwen2.5-0.5b
    python scripts/download_models.py --model qwen2.5-0.5b qwen3-1.7b bgpt
    python scripts/download_models.py --all
    python scripts/download_models.py --list

Uses hf-mirror.com by default (``--no-mirror`` / ``--hf-endpoint`` to change) and
``huggingface_hub`` resumable downloads, so an interrupted pull continues.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import CKPT_ROOT, setup_hf_endpoint  # noqa: E402
import catalog  # noqa: E402


def download_repo(name: str, spec: dict) -> None:
    from huggingface_hub import snapshot_download
    dest = CKPT_ROOT / spec["dir"]
    print(f"  [{name}] {spec['repo']} -> {dest}")
    snapshot_download(
        repo_id=spec["repo"],
        local_dir=str(dest),
        # skip the duplicate consolidated file when sharded weights exist
        ignore_patterns=["*.pth", "*.msgpack", "*.h5", "original/*"],
    )
    print(f"  [{name}] done")


def download_files(name: str, spec: dict) -> None:
    from huggingface_hub import hf_hub_download
    dest = CKPT_ROOT / spec["dir"]
    dest.mkdir(parents=True, exist_ok=True)
    for fname in spec["files"]:
        print(f"  [{name}] {spec['repo']}:{fname} -> {dest}")
        try:
            hf_hub_download(repo_id=spec["repo"], filename=fname,
                            local_dir=str(dest))
        except Exception as e:
            print(f"  [{name}] {fname} FAILED: {type(e).__name__}: {e}")
            if spec.get("note"):
                print(f"        {spec['note']}")
    print(f"  [{name}] done")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Download model weights into checkpoints/")
    p.add_argument("--model", nargs="*", help="model key(s) from the catalog")
    p.add_argument("--all", action="store_true", help="download every model")
    p.add_argument("--list", action="store_true")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--hf-endpoint", default=None)
    args = p.parse_args()

    if args.list:
        for k, s in catalog.MODELS.items():
            print(f"  {k:<14} {s['repo']:<26} -> checkpoints/{s['dir']}")
        return 0

    ep = setup_hf_endpoint(use_mirror=not args.no_mirror, endpoint=args.hf_endpoint)
    print(f"HF endpoint: {ep}")

    keys = list(catalog.MODELS) if args.all else (args.model or [])
    if not keys:
        p.error("specify --model KEY..., --all, or --list")

    for k in keys:
        s = catalog.MODELS.get(k)
        if s is None:
            print(f"  [{k}] unknown model key — see --list"); continue
        (download_repo if s["source"] == "hf_repo" else download_files)(k, s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
