"""Dataset card (README.md) for generated datasets, plus Hub upload."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .config import BenchConfig
from .dataset import FORMAT_VERSION, SCHEMA


def _split_stats(root: Path, split: str) -> dict[str, Counter]:
    import pyarrow.parquet as pq

    cols = ["category", "noise_bucket", "noise_category", "provider", "accent", "reverb"]
    out: dict[str, Counter] = {c: Counter() for c in cols}
    for f in sorted((root / "data").glob(f"{split}-*.parquet")):
        t = pq.read_table(f, columns=cols).to_pydict()
        for c in cols:
            out[c].update(str(v) for v in t[c])
    return out


def write_card(root: Path, cfg: BenchConfig, stats: dict[str, Any], gen: Any = None) -> Path:
    splits = sorted({p.name.split("-")[0] for p in (root / "data").glob("*.parquet")})
    front = {
        # mixed: some noise / RIR sources are non-commercial; see "Audio sources" below
        "license": "other",
        "task_categories": ["audio-classification"],
        "language": [cfg.language],
        "tags": ["wake-word", "keyword-spotting", "benchmark", "livekit-wakeword-bench"],
        "configs": [
            {
                "config_name": "default",
                "data_files": [{"split": s, "path": f"data/{s}-*.parquet"} for s in splits],
            }
        ],
    }
    lines = [
        "---",
        yaml.safe_dump(front, sort_keys=False).strip(),
        "---",
        "",
        f'# Wake word benchmark: "{cfg.wake_word}"',
        "",
        "Generated with livekit-wakeword-bench "
        f"(format v{FORMAT_VERSION}). Speech is synthesized through LiveKit Inference with many "
        "providers and voices (key phrases steered with phonemes), then mixed with real background noise at controlled SNR buckets, "
        "optional room reverb and band-limiting. Every clip carries the metadata needed to "
        "slice results (noise level/type, reverb, voice, accent, provider, phrase, timing).",
        "",
        "Test voices and test noise/RIR sources are **disjoint** from train/validation ones, so the "
        "test split measures generalisation to unseen speakers and environments.",
        "",
        "## Usage",
        "",
        "```python",
        "from livekit_wakeword_bench import WakeWordData, benchmark",
        "",
        'data = WakeWordData("<this repo>", split="test")',
        'noisy_misses = data.filter(category="positive", noise_bucket=["very_noisy", "extreme"])',
        "",
        'result = benchmark("hey_livekit.onnx", "<this repo>")',
        'print(result.summary()["at_threshold"])',
        "```",
        "",
        'Plain `datasets` works too: `load_dataset("<this repo>", split="test")`.',
        "",
        "## Splits",
        "",
        "| split | rows | hours |",
        "|---|---|---|",
    ]
    import pyarrow.parquet as pq

    for s in splits:
        rows, secs = 0, 0.0
        for f in sorted((root / "data").glob(f"{s}-*.parquet")):
            t = pq.read_table(f, columns=["duration_s"])
            rows += t.num_rows
            secs += float(sum(t.column("duration_s").to_pylist()))
        lines.append(f"| {s} | {rows} | {secs / 3600:.2f} |")
    for s in splits:
        st = _split_stats(root, s)
        lines += ["", f"### {s}", ""]
        for col, cnt in st.items():
            top = ", ".join(f"{k}: {v}" for k, v in cnt.most_common(12))
            lines.append(f"- **{col}**: {top}")
    from .sources import SOURCES

    lines += [
        "",
        "## Audio sources",
        "",
        "| pool | kind | name | description |",
        "|---|---|---|---|",
    ]
    for name, spec in SOURCES.items():
        if cfg.sources and name not in cfg.sources:
            continue
        lines.append(f"| {spec.pool} | {spec.kind} | `{name}` | {spec.description} |")
    lines += [
        "",
        "Speech: LiveKit Inference TTS (Cartesia Sonic 3, Inworld TTS-2, xAI, Rime Mist v3); "
        "wake words and confusables are sent as phonemes for exact pronunciation. "
        "Some noise/RIR sources are non-commercial; check them before commercial use.",
    ]
    lines += [
        "",
        "## Columns",
        "",
        "`audio` (16 kHz FLAC) plus: " + ", ".join(f"`{k}`" for k in SCHEMA),
        "",
        "Key ones: `label` (1 = contains the wake word), `category` (positive / adversarial / "
        "speech / real_speech / noise), `noise_bucket` (clean / quiet / moderate / noisy / "
        "very_noisy / extreme / noise_only), `snr_db`, `phrase_onset_s`/`phrase_offset_s` "
        "(exact wake word timing, used for detection latency), `held_out_voice`.",
        "",
        "## Generation config",
        "",
        "```yaml",
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False).strip(),
        "```",
        "",
    ]
    path = root / "README.md"
    path.write_text("\n".join(lines))
    return path


def push_to_hub(root: str | Path, repo_id: str, private: bool = True) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    card = Path(root) / "README.md"
    if card.exists() and "<this repo>" in card.read_text():
        card.write_text(card.read_text().replace("<this repo>", repo_id))
    api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["README.md", "bench_config.yaml", "data/*.parquet"],
    )
    return f"https://huggingface.co/datasets/{repo_id}"
