#!/usr/bin/env python3
"""Helpers for the datasets that don't fit the unified ``download_data.py``.

Why these are separate (see catalog notes):
  * silesia  — official host is flaky and the corpus is *mixed* (text + binaries);
               we pull only the text members from a reliable GitHub mirror.
  * maestro  — distributed as one ~120 GB zip with no per-clip streaming, so a
               byte-capped streaming download is impossible; this guides you and,
               with ``--download``, fetches + extracts a subset.
  * musdb18  — ships in a multitrack ``.stem.mp4`` container that needs the
               ``musdb`` package (+ ffmpeg) to decode into wav.

Run from the repo root:

    python scripts/download_manual.py silesia
    python scripts/download_manual.py maestro --download --limit 30
    python scripts/download_manual.py musdb18 --limit 20
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

from _common import DATA_ROOT, http_download, human_bytes  # noqa: E402


# ---------------------------------------------------------------------------
# Silesia  (auto: text members from the GitHub mirror)
# ---------------------------------------------------------------------------

SILESIA_BASE = "https://raw.githubusercontent.com/MiloszKrajewski/SilesiaCorpus/master"
SILESIA_TEXT_MEMBERS = ["dickens", "webster", "reymont", "samba", "xml"]  # skip binaries


def do_silesia(args) -> None:
    out = DATA_ROOT / "text" / "raw" / "silesia"
    out.mkdir(parents=True, exist_ok=True)
    man = open(out / "manifest.jsonl", "w", encoding="utf-8")
    total = 0
    for shard, member in enumerate(SILESIA_TEXT_MEMBERS):
        zpath = out / f"_{member}.zip"
        print(f"  [silesia] {member} ...")
        http_download(f"{SILESIA_BASE}/{member}.zip", zpath, desc=f"{member}.zip")
        with zipfile.ZipFile(zpath) as zf:
            data = zf.read(zf.namelist()[0])
        (out / f"part-{shard:05d}.txt").write_bytes(data)
        man.write(json.dumps({"source": "silesia", "member": member,
                              "shard": shard, "bytes": len(data)}) + "\n")
        zpath.unlink()
        total += len(data)
    man.close()
    (out / "_progress.json").write_text(
        json.dumps({"bytes": total, "members": SILESIA_TEXT_MEMBERS}, indent=2),
        encoding="utf-8")
    print(f"  [silesia] -> {out}  ({human_bytes(total)}, {len(SILESIA_TEXT_MEMBERS)} text members)")
    print("  Note: run scripts/normalize_text.py --dataset silesia before compressing.")


# ---------------------------------------------------------------------------
# MAESTRO  (guided; --download fetches the big zip and extracts a subset)
# ---------------------------------------------------------------------------

MAESTRO_ZIP = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"


def do_maestro(args) -> None:
    out = DATA_ROOT / "audio" / "maestro"
    if not args.download:
        print(
            "  [maestro] MANUAL — MAESTRO v3 is a single ~120 GB zip (piano audio+MIDI).\n"
            f"    URL : {MAESTRO_ZIP}\n"
            "    It has no per-clip streaming, so there is no byte-capped download.\n"
            "    Options:\n"
            "      • re-run with --download (fetches the full zip, resumable, then\n"
            "        extracts the first --limit wav files into data/audio/maestro/), or\n"
            "      • download manually from magenta.tensorflow.org/datasets/maestro and\n"
            "        drop wavs into data/audio/maestro/.\n"
            "    (Solo piano is very homogeneous -> a strong *high* online-gain set.)")
        return
    out.mkdir(parents=True, exist_ok=True)
    src = out / "_maestro-v3.0.0.zip"
    print(f"  [maestro] downloading {MAESTRO_ZIP} (~120 GB, resumable) ...")
    http_download(MAESTRO_ZIP, src, desc="maestro.zip")
    with zipfile.ZipFile(src) as zf:
        wavs = [m for m in zf.namelist() if m.lower().endswith(".wav")][: args.limit]
        for m in wavs:
            dest = out / Path(m).name
            if not dest.exists():
                dest.write_bytes(zf.read(m))
    print(f"  [maestro] -> {out}  ({len(list(out.glob('*.wav')))} wav files)")


# ---------------------------------------------------------------------------
# MUSDB18  (needs the `musdb` package; downloads the free 7s STEMS sample)
# ---------------------------------------------------------------------------

def do_musdb18(args) -> None:
    out = DATA_ROOT / "audio" / "musdb18"
    try:
        import musdb
        import soundfile as sf
    except ImportError:
        print(
            "  [musdb18] MANUAL — needs the decoder package:\n"
            "      pip install musdb soundfile        # (+ ffmpeg on PATH)\n"
            "    Then re-run this command to fetch the free 7-second STEMS sample.\n"
            "    For the full HQ set: download musdb18-hq from\n"
            "      https://sigsep.github.io/datasets/musdb.html (Zenodo) into data/audio/musdb18/.")
        return
    out.mkdir(parents=True, exist_ok=True)
    print("  [musdb18] fetching the free 7s STEMS sample via `musdb` ...")
    db = musdb.DB(root=str(out / "_stems"), download=True)
    n = 0
    for track in db[: args.limit]:
        sf.write(out / f"{track.name}.wav", track.audio, track.rate)
        n += 1
    print(f"  [musdb18] -> {out}  ({n} mixture wavs)  "
          "[full HQ set: musdb18-hq from Zenodo]")


# ---------------------------------------------------------------------------

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Manual download helpers (silesia / maestro / musdb18)")
    p.add_argument("dataset", choices=["silesia", "maestro", "musdb18"])
    p.add_argument("--download", action="store_true", help="maestro: actually fetch the big zip")
    p.add_argument("--limit", type=int, default=30, help="maestro/musdb18: # clips to extract")
    args = p.parse_args()
    {"silesia": do_silesia, "maestro": do_maestro, "musdb18": do_musdb18}[args.dataset](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
