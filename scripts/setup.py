#!/usr/bin/env python3
"""One-command setup: fetch the default models + data and normalize, so a fresh
clone is ready to run.

It just orchestrates the other scripts in the right order (models before
normalize, which needs the tokenizer), passing the HF-endpoint flags through.

Run from the repo root:

    python scripts/setup.py                      # defaults (see below)
    python scripts/setup.py --dry-run            # print the plan, do nothing
    python scripts/setup.py --pip                # also: pip install -r requirements.txt
    python scripts/setup.py --no-mirror          # use huggingface.co instead of hf-mirror
    python scripts/setup.py --models all --text-datasets pile_of_law_eurlex cuad \\
        --image-datasets kodak eurosat --audio-datasets ljspeech --text-limit 50MB

Defaults give a runnable all-3-modality setup:
  models   qwen2.5-0.5b + bgpt
  text     pile_of_law_eurlex (20MB) -> normalized   (eval text default)
  image    kodak                                       (eval image default)
  audio    none (eval audio uses --data synthetic, no download needed)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import REPO_ROOT  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="One-command setup (models + data + normalize)")
    p.add_argument("--models", nargs="*", default=["qwen2.5-0.5b", "bgpt"])
    p.add_argument("--text-datasets", nargs="*", default=["pile_of_law_eurlex"])
    p.add_argument("--image-datasets", nargs="*", default=["kodak"])
    p.add_argument("--audio-datasets", nargs="*", default=[])
    p.add_argument("--text-limit", default="20MB", help="bytes per text dataset")
    p.add_argument("--image-limit", default="24", help="images per image dataset")
    p.add_argument("--audio-limit", default="50", help="clips per audio dataset")
    p.add_argument("--normalize-model", default="checkpoints/Qwen2.5-0.5B",
                   help="tokenizer used for normalization")
    p.add_argument("--pip", action="store_true", help="also pip install -r requirements.txt")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--hf-endpoint", default=None)
    p.add_argument("--keep-going", action="store_true", help="continue past a failing step")
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = p.parse_args()

    mirror = (["--no-mirror"] if args.no_mirror else []) + \
             (["--hf-endpoint", args.hf_endpoint] if args.hf_endpoint else [])

    # (title, argv) — built in dependency order
    steps: list[tuple[str, list[str]]] = []
    if args.pip:
        steps.append(("pip install", [sys.executable, "-m", "pip", "install", "-r",
                                       os.path.join(REPO_ROOT, "requirements.txt")]))
    if args.models:
        steps.append(("download models",
                      [sys.executable, f"{SCRIPTS}/download_models.py", "--model", *args.models, *mirror]))
    if args.text_datasets:
        steps.append(("download text",
                      [sys.executable, f"{SCRIPTS}/download_data.py", "--dataset", *args.text_datasets,
                       "--limit", args.text_limit, *mirror]))
    if args.image_datasets:
        steps.append(("download image",
                      [sys.executable, f"{SCRIPTS}/download_data.py", "--dataset", *args.image_datasets,
                       "--limit", args.image_limit, *mirror]))
    if args.audio_datasets:
        steps.append(("download audio",
                      [sys.executable, f"{SCRIPTS}/download_data.py", "--dataset", *args.audio_datasets,
                       "--limit", args.audio_limit, *mirror]))
    if args.text_datasets:                       # normalize AFTER models (needs tokenizer)
        steps.append(("normalize text",
                      [sys.executable, f"{SCRIPTS}/normalize_text.py", "--dataset", *args.text_datasets,
                       "--model", args.normalize_model]))

    print("=" * 64)
    print("  OnlineLMCompress setup plan")
    print("=" * 64)
    for i, (title, argv) in enumerate(steps, 1):
        shown = " ".join(a if " " not in a else f'"{a}"' for a in argv[1:])
        print(f"  {i}. {title}\n       python {shown}")
    print("=" * 64)
    if args.dry_run:
        print("  (dry-run: nothing executed)")
        return 0

    for i, (title, argv) in enumerate(steps, 1):
        print(f"\n>>> step {i}/{len(steps)}: {title}")
        rc = subprocess.run(argv, cwd=REPO_ROOT).returncode
        if rc != 0:
            print(f"  step '{title}' exited {rc}.")
            if not args.keep_going:
                print("  aborting (use --keep-going to continue past failures).")
                return rc

    print("\n" + "=" * 64)
    print("  Setup complete. Try:")
    print("    python evaluation/eval_online.py --modality text  --mode both")
    print("    python evaluation/eval_online.py --modality image --mode both")
    print("    python evaluation/eval_online.py --modality audio --mode both --data synthetic")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
