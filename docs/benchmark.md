# Benchmark

## Run it

```bash
pip install livekit-wakeword-bench   # .onnx models also need livekit-wakeword v2
lkwb bench hey_livekit.onnx --limit 1000   # quick: 1,000 clips
lkwb bench hey_livekit.onnx                # the full test set
```

```python
from livekit_wakeword_bench import benchmark

res = benchmark("hey_livekit.onnx", limit=1000)
res.summary()["score"]
```

By default it runs on the public [Hey LiveKit test set](https://huggingface.co/datasets/binhpham/wakeword-bench-hey-livekit) (7,873 clips, 12.3 h). Use `--data` for another one: a folder you generated or a Hugging Face repo. No LiveKit credentials needed.

`--limit` picks the same random clips every time, so quick runs are comparable with each other, but not with full runs.

## Your own model

A model gets a whole clip and returns its score over time:

```python
import numpy as np
from livekit_wakeword_bench import Prediction, benchmark

class MyModel:
    name = "my-model"

    def detect(self, audio: np.ndarray, sample_rate: int) -> Prediction:
        times, scores = run_my_model(audio, sample_rate)  # stream it however your model streams
        return Prediction(times, scores)                  # scores[i] after hearing times[i] seconds

res = benchmark(MyModel(), limit=1000)
```

If your model only fires events, return `Prediction.from_events([1.9, 7.4])`.

From the command line, point at a function that returns your model: `lkwb bench my_pkg.models:make_model`.

| model | how to pass it |
|---|---|
| livekit-wakeword v2 (`.onnx`) | `hey_livekit.onnx` |
| openWakeWord | `openwakeword:hey_jarvis` (`pip install openwakeword`) |
| anything else | `my_pkg.models:make_model` (arguments: `make_model?size=small`) |

## Read the results

```
results/
  report.html          charts, every condition, and failed clips you can play
  <model>/
    summary.json       the score and headline numbers
    breakdown.json     every condition, with 95% confidence intervals
    samples.parquet    one row per clip
```

**Score:** the share of wake words caught while false alarms stay at or below 0.5 per hour. Higher is better. Change the target with `--score-fpph`.

| number | meaning |
|---|---|
| recall | wake words caught (between the start of the word and 1 s after it ends) |
| false alarms per hour | triggers on speech, real conversation and noise that don't contain the wake word |
| near-misses accepted | share of similar-sounding phrases ("hey live kid") that trigger it |
| latency | time from the end of the wake word to the trigger (negative: before it ended) |
| real-time factor | compute time ÷ audio time (use `--workers 1`) |

The breakdown splits recall and false alarms by every condition: noise level and type, another voice, a party, room, codec, phrasing, accent and voice. In the report, a condition is red or green only when its confidence interval clearly differs from the average. Gray means no clear difference. Small groups have wide intervals, so read them with care.

## Compare two models

```bash
uv run lkwb compare results/old/hey_livekit results/new/hey_livekit
```

It shows the change in the score, then the conditions that got worse and better.

## More false-alarm hours

The test set has about 6 hours of negative audio, which is too little to tell 0.5 from 0.1 false alarms per hour. Build a 24-hour set of TV, music and speeches once, then add it to any run:

```bash
uv run lkwb media -o out/media
uv run lkwb bench hey_livekit.onnx --media out/media
```

These hours are reported separately (`summary()["media"]`) and don't change the score.

## For scripts and agents

`--json` prints the score and the worst conditions, and `lkwb compare --json` prints every change. [AGENTS.md](../AGENTS.md) describes the full loop.
