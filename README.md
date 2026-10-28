# LiveKit WakeWord Bench

Test a wake word model, find where it fails, and generate data to improve it.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-dataset-yellow)](https://huggingface.co/datasets/binhpham/wakeword-bench-hey-livekit)

The benchmark reports recall and false alarms by noise, speaker, room, accent, phrase and codec. Each condition includes a 95% confidence interval. Test voices and audio sources stay separate from training.

**The score is recall at no more than 0.5 false alarms per hour. Higher is better.**

## Quick start

Requires Python 3.12+ and a local model file. For livekit-wakeword v2 streaming models:

```bash
pip install livekit-wakeword-bench livekit-wakeword
lkwb bench hey_livekit.onnx --limit 1000
```

This uses the public [Hey LiveKit dataset](https://huggingface.co/datasets/binhpham/wakeword-bench-hey-livekit). No LiveKit credentials are needed. Open `results/report.html` for charts and playable failures; `results/<model>/` contains the metrics and score traces.

Remove `--limit 1000` for a full run before reporting results. Compare runs on the same dataset and subset. Use `--workers 1` for accurate compute timing.

Other models can use the [Python detector interface](docs/benchmark.md#your-own-model), including models that only return trigger times.

## Improve a model

A model can work well in quiet audio and miss the same phrase when someone else speaks:

<img alt="A wake word is detected alone but missed with a competing voice" src="assets/teaser.svg">

Use the same test set throughout:

1. **Benchmark** a baseline and inspect `report.html` for weak conditions.
2. **Generate** training data for those conditions, then train your model.
3. **Compare** the baseline and candidate:

```bash
lkwb bench baseline.onnx --limit 1000 -o results/base
lkwb bench candidate.onnx --limit 1000 -o results/new
lkwb compare results/base/baseline results/new/candidate
```

Use the model directories written by `bench`. Never train on the test split or test sources. Wake words are synthesized; validate on real users before drawing conclusions about real-world recall. See [how to read results](docs/benchmark.md#read-the-results).

## Generate a dataset

Generation requires LiveKit credentials and uses Inference credits. Install the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/), then run:

```bash
lk cloud auth
lk app env -w
lkwb generate "hey jarvis" --size small -o out/hey_jarvis
lkwb verify out/hey_jarvis
```

The default plan is `build`. Match `--plan` to your account and leave TTS capacity for production agents. See [Generate](docs/generate.md) for capacity, pronunciation and augmentation settings, [Train](docs/train.md) to load the data, and [audio sources](docs/reference.md#audio-sources) for licensing restrictions.

## Documentation

| Task | Guide |
|---|---|
| Run and compare models | [Benchmark](docs/benchmark.md) |
| Create a dataset | [Generate](docs/generate.md) |
| Load or export training data | [Train](docs/train.md) |
| Look up commands, settings and columns | [Reference](docs/reference.md) |
| Contribute or release | [Development](docs/development.md) |

For coding agents, see [AGENTS.md](AGENTS.md).
