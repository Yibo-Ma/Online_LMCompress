"""Central registry of datasets and model weights.

Single source of truth shared by ``download_data.py``, ``download_models.py`` and
the dataset loaders.  Each entry is plain data so it doubles as documentation.

Per-entry fields
----------------
``source``   how to fetch it:
             - ``hf``        : stream a Hugging Face dataset (text/image/audio)
             - ``url``       : a single (possibly zipped) file over HTTP(S)
             - ``url_files`` : a list of individual file URLs
             - ``manual``    : needs a special tool / login; we print instructions
``gain``     expected *online* compression gain on our method: high / med / low
``domain``   short human label
``note``     anything the user should know

Text → ``data/text/raw/<key>/`` (part-*.txt + manifest.jsonl)
Image → ``data/image/<key>/``
Audio → ``data/audio/<key>/``
Models → ``checkpoints/<dir>/``
"""
from __future__ import annotations

# ===========================================================================
# TEXT  (Qwen, token-level)   →  data/text/raw/<key>/
# ===========================================================================
# hf:  fields=None means "concatenate every string-valued field in the row".

TEXT = {
    # --- high online gain: templated / repetitive / niche domains ---
    "pile_of_law_eurlex": dict(
        source="hf", hf_id="pile-of-law/pile-of-law", config="eurlex",
        split="train", fields=["text"], gain="high", domain="legal (EU law)"),
    "atticus_contracts": dict(
        source="hf", hf_id="pile-of-law/pile-of-law", config="atticus_contracts",
        split="train", fields=["text"], gain="high", domain="legal (contracts / CUAD)"),
    "codesearchnet": dict(
        source="hf", hf_id="code_search_net", config="python",
        split="train", fields=["whole_func_string"], gain="high", domain="source code"),
    "edgar_corpus": dict(
        source="hf", hf_id="eloukas/edgar-corpus", config="year_2020",
        split="train", fields=None, gain="high", domain="financial filings (10-K)",
        note="fields=None -> concatenates all section_* text fields"),
    "medal": dict(
        source="hf", hf_id="McGill-NLP/medal", config=None, split="train",
        fields=["text"], gain="med", domain="medical abstracts"),
    "hupd": dict(
        source="hf", hf_id="HUPD/hupd", config="sample", split="train",
        fields=["title", "abstract", "claims", "background", "description"],
        streaming=False, load_kwargs=dict(trust_remote_code=True, uniform_split=True),
        gain="high", domain="patents (USPTO, Jan-2016 sample)",
        note="HUPD loader is non-streaming; 'sample' = ~22k patents (Jan 2016)"),

    # --- standard benchmarks (must-report; lower online gain) ---
    "enwik8": dict(
        source="url", url="http://mattmahoney.net/dc/enwik8.zip", member="enwik8",
        gain="low", domain="benchmark: Wikipedia XML (100MB)"),
    "enwik9": dict(
        source="url", url="http://mattmahoney.net/dc/enwik9.zip", member="enwik9",
        gain="low", domain="benchmark: Wikipedia XML (1GB)"),
    "text8": dict(
        source="url", url="http://mattmahoney.net/dc/text8.zip", member="text8",
        gain="low", domain="benchmark: cleaned Wikipedia"),
    "silesia": dict(
        source="manual", gain="med", domain="benchmark: Silesia corpus (text members)",
        note="auto: scripts/download_manual.py silesia (text members from GitHub mirror)"),
}

# ===========================================================================
# IMAGE  (bGPT, byte-level)   →  data/image/<key>/
# ===========================================================================

IMAGE = {
    "kodak": dict(
        source="url_files",
        urls=[f"https://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png" for i in range(1, 25)],
        gain="low", domain="benchmark: Kodak (24 imgs)"),
    "eurosat": dict(
        source="hf", hf_id="blanchon/EuroSAT_RGB", split="train", image_key="image",
        gain="high", domain="satellite / land-cover (homogeneous)"),
    "div2k": dict(
        source="url", url="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        member=None, gain="low", domain="benchmark: DIV2K 2K (zip)",
        note="large zip; --limit caps how many images are extracted"),
    "clic2024": dict(
        source="manual", gain="low", domain="benchmark: CLIC professional",
        note="download from clic2024.compression.cc into data/image/clic2024/raw"),
}

# ===========================================================================
# AUDIO  (bGPT, byte-level 8kHz/8bit)   →  data/audio/<key>/
# ===========================================================================

AUDIO = {
    "ljspeech": dict(
        source="hf", hf_id="lj_speech", split="train", audio_key="audio",
        gain="high", domain="single-speaker speech (great for online)"),
    "librispeech": dict(
        source="hf", hf_id="librispeech_asr", config="clean", split="test",
        audio_key="audio", gain="med", domain="benchmark: speech (multi-speaker)"),
    "peoples_speech": dict(
        source="hf", hf_id="MLCommons/peoples_speech", config="test", split="test",
        audio_key="audio", as_parquet=True, gain="med", domain="speech (large)"),
    "maestro": dict(
        source="manual", gain="high", domain="solo piano (very homogeneous)",
        note="scripts/download_manual.py maestro [--download] (~120GB single zip)"),
    "musdb18": dict(
        source="manual", gain="med", domain="music",
        note="scripts/download_manual.py musdb18 (needs `pip install musdb` + ffmpeg)"),
}

# ===========================================================================
# MODELS  →  checkpoints/<dir>/
# ===========================================================================
# hf_repo entries are snapshot-downloaded; bgpt pulls individual .pth files.

MODELS = {
    "qwen2.5-0.5b": dict(source="hf_repo", repo="Qwen/Qwen2.5-0.5B", dir="Qwen2.5-0.5B"),
    "qwen2.5-7b":   dict(source="hf_repo", repo="Qwen/Qwen2.5-7B",   dir="Qwen2.5-7B"),
    "qwen3-1.7b":   dict(source="hf_repo", repo="Qwen/Qwen3-1.7B-Base", dir="Qwen3-1.7B-Base"),
    "qwen3-0.6b":   dict(source="hf_repo", repo="Qwen/Qwen3-0.6B-Base", dir="Qwen3-0.6B-Base"),
    "bgpt": dict(
        source="hf_files", repo="sander-wood/bgpt", dir="bgpt",
        files=["weights-image.pth", "weights-audio.pth", "weights-text.pth"],
        note="if your team uses custom bGPT checkpoints, drop them in checkpoints/bgpt/"),
}


ALL_DATA = {"text": TEXT, "image": IMAGE, "audio": AUDIO}


def find_dataset(key: str):
    """Return (modality, spec) for a dataset key, or (None, None)."""
    for modality, group in ALL_DATA.items():
        if key in group:
            return modality, group[key]
    return None, None
