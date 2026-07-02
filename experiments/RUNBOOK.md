# Sweep runbook

Five stages, all driven by `experiments/sweep.py` (see `README.md` for the tool).
Everything below is copy-paste. Objective per run: **`online_rate`** (bpc text /
bpsp image / bps audio), lower is better; `delta_pct` is the static→online gain.

```
Stage 0  losslessness verification   (not a sweep — must pass before anything)
Stage 1  hyperparameter search       grids/{text,image,audio}.json
Stage 1b refine (optional)           narrowed copy of the winning region
Stage 2  scaling / amortization curve grids/scaling.json
Stage 3  headline numbers            grids/headline.json
Stage 4  ablations                   grids/ablation_targets.json
```

---

## 0. One-time setup

1. Edit `experiments/run_array.sbatch`: `PROJECT_DIR`, `CONDA_SH`, `CONDA_ENV`,
   and the `#SBATCH` resource lines for your partition.
2. Edit the `model` / `data` paths in the grid files to your cluster's layout
   (this repo uses `checkpoints/…` + `data/…`; your box uses `pretrained/…` +
   `txt_data_normalized/…`).
3. Confirm the datasets/checkpoints exist on a compute node.

---

## Stage 0 — losslessness verification (bit-exactness on THIS stack)

Online losslessness is bit-fragile; confirm a round-trip on the H200 stack once,
per modality, **before** spending a sweep. These decompress (no `--no-decompress`).

```bash
python evaluation/eval_online.py --modality text  --mode both --max-bytes 200000 \
    --data data/text/normalized/qwen2.5/pile_of_law_eurlex --chunk-size 1024 --train-interval 4 --epochs-per-train 1
python evaluation/eval_online.py --modality image --mode both --data data/image/clic2024 \
    --image-count 2 --image-crop 128 --train-interval 4 --epochs-per-train 1
python evaluation/eval_online.py --modality audio --mode both --data data/audio/ljspeech \
    --audio-clips 3 --chunk-ms 1000 --train-interval 4 --epochs-per-train 1
```

Each must print `round-trip: OK` for **both** STATIC and ONLINE. A `FAILED` here
means the deterministic stack differs (GPU/driver/lib) — fix before sweeping.

---

## Stage 1 — hyperparameter search (per modality)

`text.json` = 162 runs, `image.json` / `audio.json` = 81 each. Tune on ONE dev
dataset; the winner is confirmed across datasets in Stage 3.

```bash
# text (repeat for image / audio by swapping the grid)
python experiments/sweep.py gen --grid experiments/grids/text.json --max-parallel 32
# -> prints:  sbatch --array=0-161%32 experiments/run_array.sbatch results/sweeps/text/<group>
sbatch --array=0-161%32 experiments/run_array.sbatch results/sweeps/text/<group>

# monitor (safe any time)
python experiments/sweep.py ls
python experiments/sweep.py agg --group results/sweeps/text/<group>   # partial ok

# when done: read the table + winner
python experiments/sweep.py agg --group results/sweeps/text/<group>
#   -> summary.csv (sorted), best_config.json (ready for lossless confirm)
```

Read `summary.csv` and the printed top-K. Sanity checks: `lr` should have a clear
optimum (too high hurts); more `epochs_per_train` helps until it plateaus/overfits;
larger `lora_r` helps until it saturates. If the winner sits on a grid edge (e.g.
best `lr` = 3e-4, the max), run Stage 1b.

---

## Stage 1b — refine (optional, only if the winner is on an edge)

Copy the grid, narrow every axis around the winner (finer `lr`, neighbouring
`train_interval`/`epochs`/`r`), re-gen, re-submit. Same commands.

---

## Stage 2 — scaling / amortization curve (the key figure)

Fix the Stage-1 winner, sweep only stream size → find the plateau and the
online-vs-static crossover. Build the grid straight from `best_config.json`:

```bash
BEST=results/sweeps/text/<group>/best_config.json
python - "$BEST" > experiments/grids/scaling_filled.json <<'PY'
import json, sys
w = json.load(open(sys.argv[1]))
for k in ("mode", "max_bytes", "no_decompress"): w.pop(k, None)
w["no_decompress"] = True                      # compress-only; mode defaults to both
spec = {"group_name": "text_scaling", "modality": w.pop("modality"),
        "fixed": w,
        "grid": {"max_bytes": [1_000_000, 4_000_000, 16_000_000, 64_000_000, 256_000_000]}}
print(json.dumps(spec, indent=2))
PY

python experiments/sweep.py gen --grid experiments/grids/scaling_filled.json --max-parallel 8
sbatch --array=0-4%8 experiments/run_array.sbatch results/sweeps/text/<group>
python experiments/sweep.py agg --group results/sweeps/text/<group>
```

Plot `online_rate` (and `static_rate`) vs `max_bytes` from `summary.csv`. The
**headline size = smallest size where the curve has flattened** (marginal gain per
doubling < ~1–2%). Add `1_000_000_000` for the final plateau point if wall-time
allows (one long sequential run).

Image/audio: same, but sweep `image_count` (e.g. `[8,16,32,64,128]`) or
`audio_clips` (e.g. `[16,32,64,128,256]`) instead of `max_bytes`.

---

## Stage 3 — headline numbers (frozen winner, all datasets)

Set `headline.json`'s `fixed` to the winner + the plateau `max_bytes`, list every
test dataset in `grid.data`, then:

```bash
python experiments/sweep.py gen --grid experiments/grids/headline.json --max-parallel 16
sbatch --array=0-N%16 experiments/run_array.sbatch results/sweeps/text/<group>
python experiments/sweep.py agg --group results/sweeps/text/<group>     # the paper table

# lossless confirmation of one run (mode both, decompress on) — agg already wrote it:
python evaluation/eval_online.py --config results/sweeps/text/<group>/best_config.json
```

`summary.csv` is the per-dataset online/static/delta table for the paper. Compare
against traditional baselines (bz2/brotli, FLAC, JPEG-XL) at the same data sizes.

---

## Stage 4 — ablations (one axis each, all else = winner)

```bash
python experiments/sweep.py gen --grid experiments/grids/ablation_targets.json --max-parallel 4
sbatch --array=0-2%4 experiments/run_array.sbatch results/sweeps/text/<group>
python experiments/sweep.py agg --group results/sweeps/text/<group>
```

Clone `ablation_targets.json` for other single-axis studies: LoRA rank/capacity,
`train_interval` (adaptation frequency), `epochs_per_train` (compute–ratio
Pareto), `train_on_all: [false, true]` (recent vs all-seen), cross-domain
(specialised vs general `data`).

---

## Ops cheatsheet

| Need | Command |
|---|---|
| List every sweep + best rate | `python experiments/sweep.py ls` |
| Aggregate (safe mid-flight) | `python experiments/sweep.py agg --group <g>` |
| Resume failed/killed rows | re-run the **same** `sbatch --array=…` line (`--skip-done` skips finished) |
| Re-run only row 37 | `sbatch --array=37 experiments/run_array.sbatch <group>` |
| Reproduce one run locally | `python evaluation/eval_online.py --config <group>/runs/<id>/config.json` |
| Inspect a run | `<group>/runs/<id>/{config.json,status.json,result.json,run.log}` |
