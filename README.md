# OnlineLMCompress

Lossless compression with large models, extended with an **online** twist: the model
keeps learning from the data it has already seen while it compresses.

Two modes share one pipeline:

- **Static** — a fixed pretrained model compresses chunks in batches (the team's base
  method, expressed through a modality backend).
- **Online** — every `train_interval` chunks, a LoRA adapter is fine-tuned on the
  just-coded chunks. The decoder replays the *exact* same init + training schedule on the
  chunks it decodes, so it reaches a bit-identical model state at every interval boundary
  and **no adapter is ever transmitted**. Losslessness relies on deterministic, fp32,
  bit-exact forward/backward passes (see `utils/determinism.py`) plus the padding trick in
  the decoder.

The better the model understands the data, the better it compresses — and online
adaptation lets it understand the current stream better as it goes.

## How it works

```
raw data ──to_chunks──▶ chunks ──interval──▶ encode_interval ──▶ arithmetic code
                                                  │
                                          (online) after each full
                                           interval: LoRA fine-tune
                                                  │
decoder replays init + training deterministically ▶ decode_interval (padding trick)
```

One modality-agnostic scheduler drives all three modalities through a single
`OnlineBackend` seam; only the model / tokenization / loss differ per modality:

| Modality | Backbone | Compressor | Unit fed to the coder |
|----------|----------|------------|------------------------|
| text  | causal LLM (e.g. Qwen2.5-0.5B) | `LLMCompressor` (token-level) | text → token chunks |
| image | bGPT (byte-level)             | `BGPTCompressor` (byte-level) | image → 32×32 BMP patches |
| audio | bGPT (byte-level)             | `BGPTCompressor` (byte-level) | audio → fixed-duration 8 kHz/8-bit WAV chunks |

The entropy backend (`arithmetic_coder/`, constriction range coder) and the base
compressors (`compression/{base,llm,bgpt}_compressor.py`) are shared, unchanged, with the
team's base repo.

## Layout

```
arithmetic_coder/        range coder (constriction) — shared with base
compression/
  base_compressor.py     shared encode kernel + padded-decode loop
  llm_compressor.py      token-level (text)
  bgpt_compressor.py     byte-level (image/audio)
  online/                the online layer
    static_compressor.py / online_compressor.py   the two schedulers
    base.py config.py trainer.py
    backends/            text_backend, image_backend, audio_backend (+ shared parents)
utils/                   determinism, archive, tokenization & I/O helpers
bgpt/                    vendored bGPT model (utils.py, config.py)
evaluation/
  eval_online.py         online/static round-trip eval (this project)
  eval_bgpt.py eval_llm.py image_codec_baselines.py   base benchmarks (kept for reference)
scripts/
  catalog.py             single source of truth: datasets + model weights
  download_models.py     weights  -> checkpoints/   (hf-mirror aware, resumable)
  download_data.py       datasets -> data/          (length-capped, resumable, append-extend)
  download_manual.py     helpers for silesia / maestro / musdb18 (special sources)
  normalize_text.py      tokenizer round-trip normalize: data/text/raw -> data/text/normalized
data/                    text/{raw,normalized}/  image/  audio/   (git-ignored; filled by scripts)
checkpoints/             Qwen2.5-0.5B, Qwen3-*, bgpt/weights-*.pth (git-ignored; filled by scripts)
```

## Setup (fresh clone → ready to run)

`data/` and `checkpoints/` ship empty. Everything runs from the repo root and defaults to
**hf-mirror.com** (pass `--no-mirror` or `--hf-endpoint URL` if your network prefers
`huggingface.co`).

**One command** (default models + data + normalize, all-3-modality runnable):

```bash
pip install -r requirements.txt
python scripts/setup.py            # add --dry-run to preview, --no-mirror to skip the mirror
```

Or step by step:

```bash
# 1. model weights -> checkpoints/
python scripts/download_models.py --model qwen2.5-0.5b bgpt      # or --all / --list

# 2. datasets -> data/   (--limit = bytes for text, item count for image/audio)
python scripts/download_data.py --dataset pile_of_law_eurlex --limit 20MB
python scripts/download_data.py --dataset kodak                  # image default (24 imgs)
python scripts/download_data.py --list                           # full catalog (key|gain|domain)
python scripts/download_manual.py silesia                        # special-source datasets

# 3. normalize text so it survives the tokenizer round-trip (AFTER models — needs the tokenizer)
python scripts/normalize_text.py --dataset pile_of_law_eurlex    # or all datasets
```

Re-running a download with a larger `--limit` **extends** the existing data (HF streams
skip rows already written; URL files resume via HTTP Range) — it never re-downloads. The
downloader refuses to write into a dir holding externally-provided data (no
`_progress.json`) unless `--force`.

## Usage

Run from the repo root. `--mode both` runs static then online and verifies a lossless
round-trip with a fresh model reload.

```bash
# text — defaults to data/text/normalized/pile_of_law_eurlex; --data takes any dataset dir
python evaluation/eval_online.py --modality text  --mode both --max-bytes 80000
python evaluation/eval_online.py --modality text  --mode both --data data/text/normalized/medal

# image — concatenate several images into one online stream (mirrors audio)
python evaluation/eval_online.py --modality image --mode both --data data/image/clic2024/raw --image-count 8

# audio — concatenate several clips; each clip is chunked then the lists are concatenated
python evaluation/eval_online.py --modality audio --mode both --data data/audio/peoples_speech --audio-clips 8 --chunk-ms 1000
```

Key flags: `--train-interval`, `--epochs-per-train`, `--lora-r/--lora-alpha`, `--lr`,
`--chunk-size` (text tokens/chunk), `--image-count`, `--audio-clips`, `--chunk-ms`,
`--no-decompress` (compress-only).

> **Determinism.** Online losslessness is bit-fragile: the decoder must reproduce the
> encoder's logits *and* its LoRA updates exactly. Compress and decompress must use the
> same settings on the same software/hardware stack; the archive stores a settings +
> environment fingerprint and refuses to decode on a mismatch.
