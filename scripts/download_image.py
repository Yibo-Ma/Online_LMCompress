#!/usr/bin/env python3
"""Download IMAGE datasets into ``data/image/<key>/``.

Only license-clean, verified-downloadable datasets.  Direct-URL sources need no
Hugging Face; EuroSAT streams through hf-mirror.  Run from the repo root.

    python scripts/download_image.py --list
    python scripts/download_image.py --dataset kodak
    python scripts/download_image.py --dataset eurosat --limit 200
    python scripts/download_image.py --all

``--limit`` = number of images per dataset.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    DATA_ROOT, external_guard, extract_media, http_download, setup_hf_endpoint,
    save_progress, write_download_status,
)

IMG = DATA_ROOT / "image"
DEFAULT_LIMIT = 24
IMG_EXTS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")

DATASETS = {
    "kodak": dict(kind="hf", hf_id="msdkhairi/kodak", split="train", image_key="image",
                  license="free research (benchmark)", gain="low",
                  note="24 lossless 768x512 PNGs; via HF mirror because r0k.us IP-blocks some clusters"),
    "div2k": dict(kind="url_zip", url="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
                  license="academic research only (non-commercial)", gain="low"),
    "clic2024": dict(kind="url_zip",
                     url="https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip",
                     license="CLIC (research use)", gain="low",
                     note="CLIC professional validation — lossless PNG (the standard uncompressed benchmark)"),
    "dtd":   dict(kind="url_zip", url="https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
                  license="research (Oxford VGG)", gain="high"),
    "eurosat": dict(kind="hf", hf_id="blanchon/EuroSAT_RGB", split="train", image_key="image",
                    license="MIT / Sentinel-2 open", gain="high"),
    # homogeneous single-category sets -> strong cross-image online gain (incl. cats)
    "cifar10": dict(kind="hf", hf_id="uoft-cs/cifar10", split="train", image_key="img",
                    license="research (Krizhevsky)", gain="high",
                    note="32x32; pick a single class for max homogeneity (e.g. cats)"),
    "oxford_pet": dict(kind="url_zip",
                       url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
                       license="CC BY-SA 4.0", gain="high",
                       note="cats & dogs (~37 breeds); homogeneous pet imagery"),
}


def dl_url_files(key, spec, out, limit):
    out.mkdir(parents=True, exist_ok=True)
    for u in spec["urls"][:limit]:
        dest = out / Path(u).name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        http_download(u, dest, desc=dest.name)
    n = sum(1 for p in out.iterdir() if p.suffix.lower() in IMG_EXTS)
    save_progress(out, files=n, gain=spec.get("gain"))
    print(f"  [{key}] -> {out}  ({n} images)")


def dl_url_zip(key, spec, out, limit):
    out.mkdir(parents=True, exist_ok=True)
    src = out / ("_source" + "".join(Path(spec['url']).suffixes[-2:]))
    print(f"  [{key}] {spec['url']} (downloads archive, extracts {limit} images)")
    http_download(spec["url"], src, desc=f"{key} src")
    n = extract_media(src, out, limit, IMG_EXTS, "img")
    save_progress(out, files=n, source=spec["url"], gain=spec.get("gain"))
    print(f"  [{key}] -> {out}  ({n} images)")


def dl_hf(key, spec, out, limit):
    from datasets import load_dataset
    out.mkdir(parents=True, exist_ok=True)
    have = len(list(out.glob("img_*.png")))
    if have >= limit:
        print(f"  [{key}] already {have} images (>= limit)"); return
    print(f"  [{key}] stream {spec['hf_id']} -> {limit} images")
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
        sys.stdout.write(f"\r    {idx}/{limit}"); sys.stdout.flush()
        if idx >= limit:
            break
    save_progress(out, files=idx, source=spec["hf_id"], gain=spec.get("gain"))
    print(f"\n  [{key}] -> {out}  ({idx} images)")


HANDLERS = {"url_files": dl_url_files, "url_zip": dl_url_zip, "hf": dl_hf}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Download IMAGE datasets -> data/image/")
    p.add_argument("--dataset", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="# images per dataset")
    p.add_argument("--list", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--hf-endpoint", default=None)
    args = p.parse_args()

    if args.list:
        for k, s in DATASETS.items():
            print(f"  {k:<10} gain={s.get('gain','?'):<4} {s['kind']:<10} {s.get('license','')}")
        return 0

    ep = setup_hf_endpoint(use_mirror=not args.no_mirror, endpoint=args.hf_endpoint)
    print(f"HF endpoint: {ep}")
    limit = args.limit if args.limit is not None else DEFAULT_LIMIT

    keys = list(DATASETS) if args.all else (args.dataset or [])
    if not keys:
        p.error("specify --dataset KEY..., --all, or --list")
    ok, failed, skipped = [], [], []
    for k in keys:
        spec = DATASETS.get(k)
        if spec is None:
            print(f"  [{k}] unknown — see --list"); failed.append(k); continue
        out = IMG / k
        if not external_guard(out, args.force):
            skipped.append(k); continue
        try:
            HANDLERS[spec["kind"]](k, spec, out, limit)
            ok.append(k)
        except Exception as e:
            print(f"  [{k}] FAILED: {type(e).__name__}: {str(e)[:140]}")
            failed.append(k)
    write_download_status(IMG, ok, failed, skipped)
    print(f"\nIMAGE: {len(ok)} ok"
          + (f", {len(failed)} FAILED: {', '.join(failed)}" if failed else "")
          + (f", {len(skipped)} skipped: {', '.join(skipped)}" if skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
