# Benchmarking

You've changed your model to handle background conversation. It catches the examples you tried, but does it also trigger more often on ordinary speech?

Run the baseline and candidate on the same clips to find out. The benchmark records each detector's scores over time, measures missed wake words and false alarms, and breaks the results down by condition.

Benchmarking existing data needs no LiveKit credentials. The default dataset is the public [`binhpham/wakeword-bench-hey-livekit`](https://huggingface.co/datasets/binhpham/wakeword-bench-hey-livekit) test split: 7,873 clips, 12.3 hours.

## Quick start

```bash
uv sync --extra livekit
uv run lkwb bench -d livekit:path/to/model.onnx --limit 1000   # quick run
uv run lkwb bench -d livekit:path/to/model.onnx                # full test split
```

```python
from livekit_wakeword_bench import benchmark

res = benchmark("livekit:path/to/model.onnx", limit=1000)
print(res.summary()["score"])
```

After a CLI run, open `results/report.html` to inspect the score and listen to failures. The headline `score` is recall at no more than 0.5 false alarms per hour; the condition breakdowns use the operating threshold, which defaults to 0.5. See [Metrics](metrics.md) before comparing those numbers.

## Plugging in your model

A detector is any object with a `name` and a `detect(audio, sample_rate)` method. It receives the **whole clip** (float32 mono, values in −1…1) and returns a score timeline. How it consumes the audio (chunk size, ring buffers, VAD gating, streaming state) is entirely up to the detector.

```python
import numpy as np
from livekit_wakeword_bench import Prediction, benchmark

class MyModel:
    name = "my-model"

    def __init__(self):
        self.model = load_my_model()  # replace with your model loader

    def detect(self, audio: np.ndarray, sample_rate: int) -> Prediction:
        times, scores = [], []
        hop = int(0.08 * sample_rate)
        state = self.model.reset()
        for end in range(hop, len(audio) + 1, hop):
            score, state = self.model.step(audio[end - hop:end], state)
            times.append(end / sample_rate)   # how much audio the model has heard
            scores.append(score)              # confidence in 0…1
        return Prediction(np.array(times), np.array(scores))

res = benchmark(MyModel(), limit=1000)
```

`Prediction(times, scores)` means "after hearing `times[i]` seconds of the clip, the model's confidence was `scores[i]`". The timeline is what thresholds are swept over and where detection latency comes from, so `times` must reflect real stream positions.

A detector that only fires events (no continuous score) returns their times:

```python
return Prediction.from_events([1.92, 7.40])   # confidence 1.0 at each event
```

### Stateless window models

If your model scores a fixed window of audio (for example 2 seconds), wrap it in `SlidingWindowDetector`, which simulates a microphone stream for you:

```python
from livekit_wakeword_bench import SlidingWindowDetector

det = SlidingWindowDetector(score_fn=lambda window: my_model(window),
                            name="my-window-model", sample_rate=16000,
                            window_s=2.0, hop_s=0.08)
```

Every `hop_s` it scores the most recent `window_s` seconds. The stream starts with silence, like a real microphone.

### From the command line

`-d` takes a detector spec:

| spec | meaning |
|---|---|
| `livekit:path/to/model.onnx` | livekit-wakeword streaming model, run the way `WakeWordListener` runs it: each clip is fed in 80 ms chunks through `WakeWordModel.predict` (state reset per clip). Needs `uv sync --extra livekit`. |
| `openwakeword:hey_jarvis` | openWakeWord pretrained name or model path, 80 ms frames. Needs `pip install openwakeword`. |
| `my_pkg.models:make_model` | any factory function that returns a detector |
| `my_pkg.models:make_model?size=small&name=v2` | the same, with string keyword arguments |

Built-in adapters also take query arguments, for example `livekit:model.onnx?hop_s=0.16&name=hop160`.

Pass `-d` more than once to benchmark several models side by side in one report.

## Choosing what to run on

```bash
uv run lkwb bench out/my_dataset -d ...                 # a local dataset
uv run lkwb bench you/your-dataset -d ...               # another Hugging Face dataset
uv run lkwb bench -d ... --split validation             # another split
uv run lkwb bench -d ... -f noise_bucket=extreme,very_noisy -f reverb=true   # a subset
```

`--filter` matches clip metadata (see [Dataset format](dataset-format.md)). Comma-separated values mean "any of".

**Quick runs.** `--limit N` benchmarks a random subset of N clips. The subset is fixed (seed 0), so two quick runs with the same dataset and limit are comparable. Do not compare a quick run with a full run.

**Compute timing.** The real-time factor is measured per `detect()` call. It is only accurate with `--workers 1` (the default). `--workers N` runs N processes in parallel, which helps single-threaded models but distorts timing.

## What a run writes

```
results/
  report.html                 interactive report: charts, all conditions, playable failures
  <model>/
    summary.json              headline metrics (see Metrics)
    breakdown.json            every condition, with n and 95% confidence intervals
    samples.parquet           one row per clip: metadata + outcome at all 101 thresholds
    times.npz, scores.npz     raw score timeline for every clip
    run.json                  detector name and settings
```

`--json` prints a compact summary to stdout instead of tables, for scripts and agents:

```bash
uv run lkwb bench -d my_pkg.models:make_model --limit 1000 --json
```

It contains `score`, the metrics at the operating threshold, the metrics at each false-alarm target, compute cost, and the five worst conditions per axis.

## Comparing two models

```bash
uv run lkwb compare results/baseline/my-model results/candidate/my-model
uv run lkwb compare results/a/x results/b/y --json --top 20 --min-n 50
```

`compare` lists the change in every headline metric, then every condition sorted from worst regression to best improvement. Conditions with fewer than `--min-n` clips (default 20) are skipped. For the rest, inspect the confidence intervals as well as the deltas; a small change may still be inconclusive.

In Python:

```python
from livekit_wakeword_bench import BenchResult, compare

diff = compare(BenchResult.load("results/a/x"), BenchResult.load("results/b/y"))
diff["headline"]["score"]      # {"base": ..., "new": ..., "delta": ...}
diff["slices"][:5]             # worst regressions
```

If competing speech remains a weak spot, use the [augmentation guide](augmentation.md#targeting-a-weakness) to generate training examples for it. Keep the evaluation dataset fixed while you iterate.

## Re-scoring without re-running

A saved run keeps every score timeline, so you can change how it is scored without running the model again:

```python
from livekit_wakeword_bench import BenchConfigRun, BenchResult

res = BenchResult.load("results/my-model")
strict = res.rescore(BenchConfigRun(score_fpph=0.1, debounce_s=3.0, threshold=0.7))
strict.summary()["score"]
```

`lkwb report <run>... --threshold 0.7` re-renders the HTML report at another threshold.

## Reports and charts

- `results/report.html` is self-contained (charts, tables and failure audio embedded) and follows the viewer's light or dark theme.
- `lkwb report run_a run_b --source <dataset>` builds a comparison report from saved runs. `--source` lets it embed failure audio.
- `lkwb figures results/my-model` writes the README charts (SVG, dark and light) for your model.

## Python API summary

| function | does |
|---|---|
| `benchmark(detector, source=DEFAULT_DATASET, split="test", cfg=None, workers=1, limit=None, **filters)` | load, run, score; returns `BenchResult` |
| `run_detector(detector, data, ...)` | run only; returns raw score timelines |
| `evaluate(results, data, name, cfg)` | score raw timelines |
| `BenchResult.summary()` / `.breakdown()` / `.failures()` / `.frame()` | metrics, conditions, worst clips, per-clip DataFrame |
| `BenchResult.save(dir)` / `BenchResult.load(dir)` / `.rescore(cfg)` | persistence and re-scoring |
| `compare(base, new, min_n=20)` | per-metric and per-condition deltas |
| `load_detector(spec)` | build a detector from a spec string |
