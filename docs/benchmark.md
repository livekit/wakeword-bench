# Benchmark

Evaluate your detector on a [generated benchmark](generate.md) or an existing dataset. For sample categories and coverage, see [what gets generated](../README.md#what-gets-generated).

## Run it

Requires Python 3.12+. Install the package and the adapter for your model:

```bash
pip install livekit-wakeword-bench livekit-wakeword
lkwb bench your_model.onnx --data out/hey_jarvis --limit 1000  # quick check
lkwb bench your_model.onnx --data out/hey_jarvis               # full test set
```

`.onnx` files must be livekit-wakeword v2 streaming models. The default dataset is the public [Hey LiveKit test set](https://huggingface.co/datasets/binhpham/wakeword-bench-hey-livekit). Use `--data PATH_OR_REPO` for a local dataset or another Hugging Face dataset. No LiveKit credentials are needed.

`--limit` selects a fixed random subset (seed 0). Compare runs with the same dataset, split, filters and limit; use a full run before reporting results.

```python
from livekit_wakeword_bench import benchmark

result = benchmark("hey_livekit.onnx", limit=1000)
print(result.summary()["score"])
result.save("results/hey_livekit")
```

## Read the results

The CLI writes:

```text
results/
  report.html          charts and playable failures
  <model>/
    summary.json       score and headline metrics
    breakdown.json     metrics by condition, with 95% confidence intervals
    samples.parquet    metadata and outcomes for each clip
    times.npz          score timestamps
    scores.npz         score traces
```

**Score** is recall at the lowest tested threshold with no more than 0.5 false alarms per hour (FPPH). Higher is better. Change the target with `--score-fpph`.

The breakdown uses `--threshold` (default: 0.5), which can differ from the threshold used for the score.

| Metric | Meaning |
|---|---|
| Recall | Share of wake words detected from phrase onset to 1 second after its end |
| FPPH | Triggers per hour of ordinary speech, recorded speech and noise |
| Adversarial false accept rate | Share of similar-sounding phrases, such as “hey live kid,” that trigger the model; excluded from FPPH |
| Latency | Time from phrase end to trigger; negative means the trigger came before the phrase ended |
| Real-time factor | Compute time divided by audio duration; measure with `--workers 1` |

Breakdowns group clips by conditions such as noise, competing talkers, reverb, codec and accent. Red or green marks a clear difference from the average; gray means the difference is uncertain.

Treat confidence intervals wider than about 0.2 as weak evidence. Synthesized wake words do not establish recall on real users, and accent labels come from TTS metadata. Rare false alarms need many hours of negative audio to measure reliably.

## Compare models

Save each run separately:

```bash
lkwb bench baseline.onnx --limit 1000 -o results/base
lkwb bench candidate.onnx --limit 1000 -o results/new
lkwb compare results/base/baseline results/new/candidate
```

Use the model directories printed by `bench`. The comparison shows headline changes, then conditions from worst regression to best improvement. Groups with fewer than 20 clips are skipped by default (`--min-n`). Positive `improvement` values mean better results.

## Your own model

A detector needs a `name` and a `detect` method. It receives a whole float32 mono clip and controls its own streaming:

```python
import numpy as np
from livekit_wakeword_bench import Prediction, benchmark

class MyModel:
    name = "my-model"

    def detect(self, audio: np.ndarray, sample_rate: int) -> Prediction:
        times, scores = run_my_model(audio, sample_rate)  # replace with your code
        return Prediction(times, scores)

result = benchmark(MyModel(), limit=1000)
```

`scores[i]` must be the score emitted after hearing `times[i]` seconds of audio. For trigger-only models, return `Prediction.from_events([1.9, 7.4])`.

| Model | CLI argument | Extra dependency |
|---|---|---|
| livekit-wakeword v2 | `hey_livekit.onnx` | `livekit-wakeword` |
| openWakeWord | `openwakeword:hey_jarvis` | `openwakeword` |
| Custom factory | `my_pkg.models:make_model?size=small` | Your model's dependencies |

A factory must return a detector. See [custom_detector.py](../examples/custom_detector.py).

## Add false-alarm hours

Build a separate set of TV, music and speeches, targeting 24 hours:

```bash
lkwb media -o out/media
lkwb bench hey_livekit.onnx --media out/media
```

Media results appear in `summary()["media"]` and do not change the main score.

## Use in scripts

`bench --json` prints a JSON array with one object per detector. Each object includes `score`, `at_threshold`, `at_target_fpph`, `compute`, `worst_slices` and `run_dir`.

```bash
lkwb bench hey_livekit.onnx --limit 1000 --json > benchmark.json
lkwb compare results/base/baseline results/new/candidate --json > comparison.json
```

`compare --json` returns `headline` and `slices`, limited to `--top` entries per direction (default: 10). Each slice includes `base`, `new`, `delta`, `improvement` and `n`; positive `improvement` means better.
