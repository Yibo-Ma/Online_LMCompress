#!/usr/bin/env python3
"""Download AUDIO datasets into ``data/audio/<key>/``.

Only license-clean, verified-downloadable datasets.  LibriSpeech / LJSpeech /
MAESTRO come from **direct URLs** (no Hugging Face, and avoid the HF audio-decode
deps); People's Speech streams through hf-mirror as raw-byte parquet.
Run from the repo root.

    python scripts/download_audio.py --list
    python scripts/download_audio.py --dataset librispeech --limit 50
    python scripts/download_audio.py --dataset ljspeech maestro --limit 30
    python scripts/download_audio.py --all

``--limit`` = number of clips per dataset.  bGPT eval reads these as WAV/FLAC
blobs (see eval_online --modality audio).

tar(.gz/.bz2) sources (librispeech/ljspeech/nsynth/esc50/speech_commands) are
**streamed**: the connection is closed once ``--limit`` clips are extracted, so
``--limit`` also caps how much is downloaded.  zip sources (MAESTRO) keep their
index at the end, so they must download fully before extracting ``--limit`` wavs.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    DATA_ROOT, external_guard, extract_media, http_download, is_streamable_tar,
    save_progress, setup_hf_endpoint, stream_extract_tar, write_download_status,
)

AUD = DATA_ROOT / "audio"
DEFAULT_LIMIT = 50
AUDIO_EXTS = (".flac", ".wav")

DATASETS = {
    "librispeech": dict(kind="url_tar", url="https://www.openslr.org/resources/12/test-clean.tar.gz",
                        license="CC-BY 4.0", gain="med",
                        note="OpenSLR direct (avoids HF librosa dep); single reader per chapter"),
    "ljspeech":    dict(kind="url_tar", url="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
                        license="public domain", gain="high", note="single female speaker (2.6GB)"),
    "maestro":     dict(kind="url_tar", url="https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip",
                        license="CC-BY-NC-SA (non-commercial)", gain="high",
                        note="solo piano; ~120GB zip — downloads fully before extracting --limit wavs"),
    "peoples_speech": dict(kind="hf_parquet", hf_id="MLCommons/peoples_speech", config="test",
                           split="test", audio_key="audio", license="CC-BY 4.0 / CC0", gain="med"),
    # homogeneous / single-source -> strong online gain
    "nsynth": dict(kind="url_tar",
                   url="http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
                   license="CC-BY 4.0", gain="high",
                   note="single instrument notes — very homogeneous (great for online)"),
    "speech_commands": dict(kind="url_tar",
                            url="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                            license="CC-BY 4.0", gain="med",
                            note="short spoken words; many near-identical utterances"),
    "esc50": dict(kind="url_tar",
                  url="https://github.com/karoldvl/ESC-50/archive/master.tar.gz",
                  license="CC BY-NC 3.0 (non-commercial)", gain="high",
                  note="environmental sounds; stationary texture (bioacoustic-like)"),
}


def dl_url_tar(key, spec, out, limit):
    """tar(.gz/.bz2): stream and close the connection after ``limit`` clips, so
    ``--limit`` caps the download size.  zip (MAESTRO) keeps its index at the end,
    so it must download fully before we can extract ``limit`` clips."""
    out.mkdir(parents=True, exist_ok=True)
    url = spec["url"]
    if is_streamable_tar(url):
        print(f"  [{key}] {url}  (streaming; stops after {limit} clips)")
        n = stream_extract_tar(url, out, limit, AUDIO_EXTS, "clip", desc=key)
    else:
        big = " (MAESTRO is ~120GB)" if key == "maestro" else ""
        print(f"  [{key}] {url}{big}  (zip: downloads fully, then extracts {limit} clips)")
        src = out / ("_source" + "".join(Path(url).suffixes[-2:]))
        http_download(url, src, desc=f"{key} src")
        n = extract_media(src, out, limit, AUDIO_EXTS, "clip")
    save_progress(out, clips=n, source=url, gain=spec.get("gain"))
    print(f"  [{key}] -> {out}  ({n} clips)")


def dl_hf_parquet(key, spec, out, limit):
    """Stream the HF parquet and stop after ``limit`` clips so --limit caps the
    download (the old non-streaming path pulled the whole split first).  Audio stays
    raw bytes (decode=False -> no librosa).  The written parquet keeps the same
    {audio:{bytes,path}, ...} schema the loader expects (utils/audio_utils)."""
    from datasets import load_dataset, Audio, Dataset
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{spec.get('config', 'data')}-{spec['split']}.parquet"
    if dst.exists():
        print(f"  [{key}] {dst.name} exists; skip"); return
    print(f"  [{key}] {spec['hf_id']} ({spec.get('config')}) -> parquet "
          f"(streaming raw bytes; stops after {limit} clips)")
    ds = load_dataset(spec["hf_id"], spec.get("config"), split=spec["split"], streaming=True)
    ds = ds.cast_column(spec.get("audio_key", "audio"), Audio(decode=False))
    rows = []
    for row in ds:
        rows.append(row)
        if limit and len(rows) >= limit:
            break
    Dataset.from_list(rows).to_parquet(str(dst))
    save_progress(out, clips=len(rows), source=spec["hf_id"], gain=spec.get("gain"))
    print(f"  [{key}] -> {dst}  ({len(rows)} clips)")


HANDLERS = {"url_tar": dl_url_tar, "hf_parquet": dl_hf_parquet}


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Download AUDIO datasets -> data/audio/")
    p.add_argument("--dataset", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="# clips per dataset")
    p.add_argument("--list", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--hf-endpoint", default=None)
    args = p.parse_args()

    if args.list:
        for k, s in DATASETS.items():
            print(f"  {k:<14} gain={s.get('gain','?'):<4} {s['kind']:<11} {s.get('license','')}")
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
        out = AUD / k
        if not external_guard(out, args.force):
            skipped.append(k); continue
        try:
            HANDLERS[spec["kind"]](k, spec, out, limit)
            ok.append(k)
        except Exception as e:
            print(f"  [{k}] FAILED: {type(e).__name__}: {str(e)[:140]}")
            failed.append(k)
    write_download_status(AUD, ok, failed, skipped)
    print(f"\nAUDIO: {len(ok)} ok"
          + (f", {len(failed)} FAILED: {', '.join(failed)}" if failed else "")
          + (f", {len(skipped)} skipped: {', '.join(skipped)}" if skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
