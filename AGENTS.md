# AGENTS.md

Guide for coding agents that use this repo to build better wake word models.

## What this repo gives you

- A **fixed benchmark** (`binhpham/wakeword-bench-hey-livekit` on the Hugging Face Hub, or any dataset you generate). The test voices, noise and rooms are never used for training.
- **One number to maximize:** `score`, the recall at the lowest threshold whose false positives per hour (FPPH) are ≤ 0.5. Use `--score-fpph` to change the target.
- **Per-slice diagnostics** that tell you what to fix: noise level, noise type, competing talker, reverb, codec, accent, provider, sentence form and adversarial phrase, each with 95% confidence intervals.
- **A generator** that makes more data for exactly those slices.

## The loop

```bash
# 0. setup (once)
uv sync --extra torch                                 # plus livekit-wakeword v2 for .onnx models
lk cloud auth && lk app env -w                        # only needed for generation

# 1. baseline: quick iteration on a subset, full run before you claim a result
uv run lkwb bench baseline.onnx --limit 1000 --json -o results/base

# 2. read results/base/*/summary.json and breakdown.json (or the --json stdout):
#    find the worst slices with tight confidence intervals

# 3. targeted training data, e.g. more competing-talker and extreme-noise examples
uv run lkwb generate "hey livekit" -o out/train_v2 --plan ship --split train \
    --set splits.train.augment=true --set augment.p_interferer=0.6 \
    --set augment.clean_weight=0.2

# 4. train (your code; WakeWordData(...).with_transform(Augment()).torch() for PyTorch)

# 5. compare the candidate against the baseline, slice by slice
uv run lkwb bench candidate.onnx --limit 1000 --json -o results/cand
uv run lkwb compare results/base/baseline results/cand/candidate --json
```

## Rules

- **Never train on the `test` split** or on test sources. `train`/`validation` and the `Augment()` transform only use train pools, so stay inside them.
- Quick runs (`--limit`) are a random but fixed subset (seed 0), so they're comparable to each other. Don't compare a quick run against a full run.
- Treat slices whose CI width is above ~0.2 as anecdotes. `compare` ignores slices with `n < 20` by default.
- RTF (compute cost) is only accurate with `--workers 1`.
- Generation spends LiveKit Inference credits and is capped by the account plan's TTS concurrency (build 5, ship 20, scale 50). Never raise `tts.concurrency` above the plan. TTS results are cached in `.cache/lkwb/tts`, so regenerating with different augmentation is free.

## Outputs

`lkwb bench --json` prints one object per detector:

```json
{"detector": "hey_livekit", "score": 0.8983, "score_definition": "recall at FPPH <= 0.5",
 "at_threshold": {"threshold": 0.5, "recall": 0.911, "fpph": 1.31, "false_accept_rate": 0.004,
                  "adversarial_false_accept_rate": 0.299, "latency_ms_p50": -18.5, "latency_ms_p90": 83.7},
 "at_target_fpph": {"fpph<=1.0": {...}, "fpph<=0.5": {...}, "fpph<=0.1": {...}},
 "compute": {"real_time_factor": 0.0066, ...},
 "worst_slices": {"positive/interferer": [{"value": "peoples_speech", "n": 215, "recall": 0.78, "ci95": [0.72, 0.83]}], ...},
 "run_dir": "results/base/hey_livekit"}
```

`run_dir` contains:
- `summary.json` and `breakdown.json` (every slice)
- `samples.parquet`: every clip's metadata and per-threshold outcomes
- `times.npz` and `scores.npz`: raw score traces, so you can re-threshold without re-running the model
- `report.html`, one level up, for humans

`lkwb compare A B --json` returns `headline` deltas plus `slices` sorted from worst regression to best improvement. Each entry has `slice`, `metric`, `base`, `new`, `delta`, `improvement` and `n`, and `improvement > 0` means better.

## Plugging in a model

A detector is any object with `name` and `detect(audio: np.ndarray, sample_rate: int) -> Prediction`:

- `audio` is the whole clip as float32 mono. The model streams it however it wants.
- `Prediction(times, scores)` is a score timeline: `scores[i]` was emitted after hearing `times[i]` seconds of audio.
- A model that only emits triggers can return `Prediction.from_events([t1, t2])`.

From the CLI, pass `module.path:factory?key=value` as the model. A `.onnx` path is a livekit-wakeword v2 streaming model, and `openwakeword:<name>` is openWakeWord. From Python: `benchmark(model, dataset=..., limit=...)`. See `examples/custom_detector.py`.

## Knobs

`uv run lkwb knobs` lists every config key with its default. You can set any of them in three ways:
- the CLI: `--set a.b=value`
- Python: `generate(..., a__b=value)`
- a YAML file passed with `-c`

The ones you'll use most:

| knob | meaning |
|---|---|
| `splits.<split>.n_positive / n_adversarial / n_speech / n_noise / real_speech_minutes` | counts per category |
| `splits.<split>.augment` | mix noise/reverb/etc into the stored audio (`train` defaults to clean) |
| `augment.snr_buckets`, `augment.snr_weights`, `augment.clean_weight` | noise level distribution |
| `augment.p_reverb`, `p_interferer`, `p_second_noise`, `p_eq`, `p_bandlimit`, `p_clip`, `p_codec` | how often each augmentation fires |
| `hard_negatives`, `pronunciations`, `positive_texts`, `p_carrier` | phrases and pronunciation |
| `tts.plan` / `tts.concurrency`, `tts.models`, `tts.test_voice_fraction`, `tts.speed` | TTS |

## Code map

`src/livekit_wakeword_bench/`:

| file | role |
|---|---|
| `generate.py` | plan → TTS → mix → parquet |
| `augment.py` | augmentation chain |
| `phonemes.py` | phoneme rendering per provider |
| `text.py` | phrases and adversarial generation |
| `sources.py` | Hugging Face noise / RIR / speech corpora |
| `loader.py` | `WakeWordData` |
| `detectors.py` | `Detector` protocol + adapters |
| `bench.py` | metrics, breakdowns, `compare` |
| `report.py` | HTML report |
| `verify.py` | dataset QA |
| `cli.py` | `lkwb` |

Tests: `uv run pytest`. Lint: `uv run ruff check src tests`. Releasing: [docs/development.md](docs/development.md#releasing).
