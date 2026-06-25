"""Online / static LoRA-adaptive compression evaluation (text / image / audio).

Runs the same data through StaticCompressor and/or OnlineCompressor, verifies a
lossless round-trip (fresh model reload for the decoder, proving no hidden state
is reused), and reports compression ratio / bpb / wall time.

  text   — causal LLM over token chunks            (model: checkpoints/Qwen2.5-0.5B)
  image  — bGPT over 32x32 BMP patches             (model: checkpoints/bgpt/weights-image.pth)
  audio  — bGPT over fixed-duration WAV chunks      (model: checkpoints/bgpt/weights-audio.pth)

Examples
--------
    # text, both modes
    python evaluation/eval_online.py --modality text --mode both --max-bytes 80000

    # image lossless round-trip (small crop) then benefit (compress-only, larger)
    python evaluation/eval_online.py --modality image --mode both --image-crop 64
    python evaluation/eval_online.py --modality image --mode both --image-crop 128 --no-decompress

    # audio synthetic
    python evaluation/eval_online.py --modality audio --mode both --max-bytes 8192 --chunk-ms 250
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.online.config import OnlineLearningConfig
from compression.online.static_compressor import StaticCompressor
from compression.online.online_compressor import OnlineCompressor
from utils.determinism import ensure_deterministic, sync


DEFAULT_TEXT = "data/text/normalized/pile_of_law_eurlex"
DEFAULT_MODEL = {
    "text":  "checkpoints/Qwen2.5-0.5B",
    "image": "checkpoints/bgpt/weights-image.pth",
    "audio": "checkpoints/bgpt/weights-audio.pth",
}
DEFAULT_TARGETS = {                       # LoRA target modules per backbone family
    "text":  ["q_proj", "k_proj", "v_proj", "o_proj"],
    "image": ["c_attn"],
    "audio": ["c_attn"],
}


# ---------------------------------------------------------------------------
# Data loading per modality  (raw = str for text, list[bytes] for image/audio)
# ---------------------------------------------------------------------------

def _synthetic_pcm_seg(n_bytes: int):
    import numpy as np
    from utils.audio_utils import _float_pcm_to_pydub_seg
    t = np.arange(n_bytes, dtype=np.float64) / 8000
    sig = (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t)
           + 0.25 * np.sin(2 * np.pi * 110 * t)) / 1.75
    return _float_pcm_to_pydub_seg(sig, 8000)


def _load_raw(modality: str, path: str, args):
    if modality == "text":
        # A file is read directly; a directory is treated as a dataset folder and
        # its part-*.txt shards are concatenated in order (matches download_data.py).
        if os.path.isdir(path):
            raw = b""
            for part in sorted(glob.glob(os.path.join(path, "part-*.txt"))):
                with open(part, "rb") as f:
                    raw += f.read()
        else:
            with open(path, "rb") as f:
                raw = f.read()
        if args.max_bytes is not None:
            raw = raw[:args.max_bytes]
        return raw.decode("utf-8", errors="ignore")

    if modality == "image":
        from PIL import Image
        from utils.img_utils import patchify_image, _load_image_dir

        # A directory yields several images (first --image-count, sorted) that are
        # concatenated into one online stream; a single file keeps prior behavior.
        files = _load_image_dir(path, args.image_count) if os.path.isdir(path) else [path]

        blobs = []
        for fp in files:
            img = Image.open(fp).convert("RGB")
            if args.image_crop:
                c = args.image_crop
                img = img.crop((0, 0, min(c, img.width), min(c, img.height)))
            patches, _meta = patchify_image(img, patch_size=args.image_px)
            blobs.extend(p.data for p in patches)
        return blobs

    if modality == "audio":
        from utils.audio_utils import audio_to_pydub_seg, chunk_pydub_audio

        if path == "synthetic":
            return chunk_pydub_audio(_synthetic_pcm_seg(args.max_bytes or 8192),
                                     chunk_ms=args.chunk_ms)
        if os.path.isdir(path) or path.endswith(".parquet"):
            # A registered audio dataset (e.g. People's Speech parquet): take the
            # first --audio-clips clips and chunk EACH clip independently, then
            # concatenate the chunk lists.  Symmetric to image (patchify per file,
            # then concat): every clip boundary lands on a whole WAV chunk, and the
            # per-clip chunking matches base's chunk_audio_for_compression.
            from utils.audio_utils import load_audio_samples
            samples = load_audio_samples(path, n=args.audio_clips)
            blobs = []
            for s in samples[:args.audio_clips]:
                blobs.extend(chunk_pydub_audio(audio_to_pydub_seg(s), chunk_ms=args.chunk_ms))
            return blobs
        # single audio file
        return chunk_pydub_audio(audio_to_pydub_seg(path), chunk_ms=args.chunk_ms)

    raise NotImplementedError(modality)


def _raw_size(modality: str, raw) -> int:
    return len(raw.encode("utf-8")) if modality == "text" else sum(len(b) for b in raw)


def _build_backend(modality: str, model_path: str, device, args):
    if modality == "text":
        from compression.online.backends.text_backend import TextBackend
        return TextBackend(model_path, device, chunk_size_tokens=args.chunk_size)
    if modality == "image":
        from compression.online.backends.image_backend import ImageBackend
        return ImageBackend(model_path, device)
    if modality == "audio":
        from compression.online.backends.audio_backend import AudioBackend
        return AudioBackend(model_path, device)
    raise NotImplementedError(modality)


# ---------------------------------------------------------------------------
# One mode (static | online), with fresh-reload decode verification
# ---------------------------------------------------------------------------

def _run_mode(mode: str, args, device, cfg: OnlineLearningConfig):
    raw = _load_raw(args.modality, args.data, args)
    orig_bytes = _raw_size(args.modality, raw)
    n_units = 1 if args.modality == "text" else len(raw)
    print(f"\n{'=' * 64}\n  {mode.upper()}  ({args.modality})\n{'=' * 64}")
    print(f"  original: {orig_bytes} bytes" +
          ("" if args.modality == "text" else f"  ({n_units} blobs)"))

    verify = not args.no_decompress

    def make_compressor():
        backend = _build_backend(args.modality, args.model, device, args)
        if mode == "static":
            return StaticCompressor(backend, device, batch_chunks=args.batch_chunks)
        return OnlineCompressor(backend, device, cfg)

    comp = make_compressor()
    comp.setup()
    sync(device); t0 = time.time()
    archive = comp.compress(raw)
    sync(device); comp_s = time.time() - t0

    comp_bytes = len(archive)
    ratio = orig_bytes / max(comp_bytes, 1)
    bpb = comp_bytes * 8 / max(orig_bytes, 1)
    print(f"  compressed: {comp_bytes} bytes | ratio={ratio:.3f}x | "
          f"bpb={bpb:.4f} | {comp_s:.2f}s")

    del comp
    if device.type == "cuda":
        torch.cuda.empty_cache()

    decomp_s = -1.0
    if verify:
        print("  verifying with fresh model reload ...")
        comp2 = make_compressor()
        comp2.setup()
        sync(device); t0 = time.time()
        recovered = comp2.decompress(archive)
        sync(device); decomp_s = time.time() - t0

        ok = (recovered == raw)
        print(f"  round-trip: {'OK' if ok else 'FAILED'} | {decomp_s:.2f}s")
        if not ok:
            raise AssertionError(f"{mode} round-trip FAILED — not lossless!")

        del comp2
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        print("  (decompress skipped — compress-only)")

    return dict(mode=mode, orig=orig_bytes, comp=comp_bytes,
                ratio=ratio, bpb=bpb, comp_s=comp_s, decomp_s=decomp_s)


def _print_comparison(results):
    print(f"\n{'=' * 72}\n  COMPARISON\n{'=' * 72}")
    print(f"  {'mode':<10}{'size':>10}{'ratio':>10}{'bpb':>10}{'comp':>10}{'decomp':>10}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['mode']:<10}{r['comp']:>9}B{r['ratio']:>9.3f}x"
              f"{r['bpb']:>10.4f}{r['comp_s']:>9.2f}s{r['decomp_s']:>9.2f}s")
    print("  " + "-" * 60)
    if len(results) == 2:
        s, o = results[0], results[1]
        d = s["bpb"] - o["bpb"]
        pct = d / s["bpb"] * 100 if s["bpb"] else 0.0
        print(f"  delta bpb (static-online) = {d:+.4f}  ({pct:+.1f}%)")
    print("=" * 72)


def _build_parser():
    p = argparse.ArgumentParser(description="Online/static LoRA compression eval")
    p.add_argument("--modality", default="text", choices=["text", "image", "audio"])
    p.add_argument("--mode", default="both", choices=["static", "online", "both"])
    p.add_argument("--model", default=None, help="default: per-modality (see DEFAULT_MODEL)")
    p.add_argument("--data", default=None, help="default: per-modality sample")
    p.add_argument("--max-bytes", type=int, default=10000, help="text/audio length cap")
    p.add_argument("--chunk-size", type=int, default=512, help="text: tokens per chunk")
    p.add_argument("--batch-chunks", type=int, default=4, help="static batch size")
    p.add_argument("--train-interval", type=int, default=4)
    p.add_argument("--epochs-per-train", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    # --- LoRA / training knobs (OnlineLearningConfig) ---
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--train-on-all", action="store_true",
                   help="train on all chunks seen so far (default: recent interval only)")
    p.add_argument("--image-px", type=int, default=32, help="image: spatial patch px")
    p.add_argument("--image-crop", type=int, default=0, help="image: crop to NxN (0=full)")
    p.add_argument("--image-count", type=int, default=1,
                   help="image: # images from a directory to concatenate into one online stream")
    p.add_argument("--chunk-ms", type=int, default=250, help="audio: chunk duration ms")
    p.add_argument("--audio-clips", type=int, default=1,
                   help="audio: # clips from a dataset to concatenate into one stream")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-decompress", action="store_true",
                   help="compress only (skip slow lossless verification)")
    return p


def _resolve_default_data(modality: str) -> str:
    if modality == "text":
        return DEFAULT_TEXT
    if modality == "audio":
        return "synthetic"
    # image: prefer kodak (auto-downloadable benchmark); fall back to clic2024 if present.
    for img_dir in ("data/image/kodak", "data/image/clic2024/raw", "data/image/clic2024/bmp"):
        if os.path.isdir(img_dir) and glob.glob(os.path.join(img_dir, "*.*")):
            return img_dir
    raise FileNotFoundError(
        "No image dataset found. Run: python scripts/download_data.py --dataset kodak "
        "(or pass --data <dir>).")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = _build_parser().parse_args()
    if args.model is None:
        args.model = DEFAULT_MODEL[args.modality]
    if args.data is None:
        args.data = _resolve_default_data(args.modality)
    args.model = os.path.normpath(args.model)

    ensure_deterministic()
    device = torch.device(args.device)

    cfg = OnlineLearningConfig(
        target_modules=DEFAULT_TARGETS[args.modality],
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_seq_len=args.chunk_size,
        epochs_per_train=args.epochs_per_train,
        train_interval=args.train_interval,
        train_on_recent_only=not args.train_on_all,
        base_seed=args.base_seed,
    )

    modes = ["static", "online"] if args.mode == "both" else [args.mode]
    results = [_run_mode(m, args, device, cfg) for m in modes]
    _print_comparison(results)


if __name__ == "__main__":
    main()
