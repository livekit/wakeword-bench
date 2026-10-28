# LiveKit WakeWord Bench

Generate a benchmark or training dataset for your wake word, with labeled speech, near-miss phrases and realistic audio conditions. Use the benchmark to measure any detector against the same test set.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-dataset-yellow)](https://huggingface.co/datasets/binhpham/wakeword-bench-hey-livekit)

## Quick start

Requires Python 3.12+ and the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/). Generation uses LiveKit Inference credits.

```bash
pip install livekit-wakeword-bench
lk cloud auth
lk app env -w

# Generate train, validation and test splits for your phrase.
lkwb generate "hey jarvis" --size small -o out/hey_jarvis
lkwb verify out/hey_jarvis
```

For a benchmark only, add `--split test`; for training data only, add `--split train`. Use `--size tiny` for a smoke test. Choose `--plan` to match your account (`build` by default). TTS capacity is shared with production agents: leave room for them with a lower `--set tts.concurrency=N`.

For unusual names, run `lkwb init "your wake word"` to create a pronunciation config. See [Generate](docs/generate.md) for sizes, pronunciation, custom settings and outputs.

## What gets generated

The generator combines phoneme-guided TTS with recorded noise, room responses and speech. Each clip has 16 kHz audio, a label, phrase timing and metadata describing its voice and audio conditions.

| Sample | Contents | What it tests |
|---|---|---|
| Positive | Your wake word alone or in a sentence, with varied voices, delivery and speaking speed | Does the model detect the intended phrase? |
| Adversarial negative | Similar-sounding substitutions, partial phrases and your custom hard negatives | Does it reject near misses? |
| Ordinary speech | Synthesized sentences without the wake word | Does unrelated conversation trigger it? |
| Recorded speech | Segments from real speech corpora | Does natural speech trigger it? |
| Noise only | Background sounds without the target phrase | Do environmental sounds trigger it? |

Only positives have label `1`; all negatives have label `0`. Category counts depend on the size and split: training presets contain positives, adversarial negatives and synthesized speech. See the [exact sample counts](docs/generate.md#choose-sample-counts).

Positive and adversarial phrases are isolated or placed before/after surrounding speech (`p_carrier=0.35` by default). Near misses are generated from phonetic similarity; add known confusions with `hard_negatives`.

## Which conditions are covered

Validation and test clips sample combinations of these conditions. They are not an exhaustive test of every combination.

| Condition | Default coverage |
|---|---|
| Background noise | Clean background plus five SNR buckets, from quiet (20–30 dB) to noise louder than speech (−5–0 dB) |
| Competing speech | A second voice 0–20 dB quieter; party scenes with 2–4 talkers and crowd noise |
| Rooms | Recorded room impulse responses for reverberation |
| Device effects | EQ, band limiting, clipping, codec compression and varying levels |
| Voice and delivery | Multiple TTS providers, voices and accent labels; speaking speed 0.85–1.2× |
| Phrasing | Isolated phrases, phrases in sentences and phonetically similar negatives |

For example, the same wake word can be detected alone and missed with a competing speaker:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/teaser_dark.svg">
  <img alt="A wake word is detected alone but missed with a competing voice" src="assets/teaser_light.svg" width="100%">
</picture>

Training clips are stored clean by default so you can [add fresh effects while training](docs/train.md). Test voices, noise, rooms and recorded speakers are held apart from training. Never train on the test split or test sources. Some source corpora restrict commercial use; see [sources and licenses](docs/reference.md#audio-sources).

Tune the recipe with YAML or `--set key=value`. For example, generate training data with more competing speech:

```bash
lkwb generate "hey jarvis" --size small --split train -o out/train_talkers \
    --set splits.train.augment=true \
    --set augment.p_interferer=0.6 \
    --set augment.clean_weight=0.2
```

Run `lkwb knobs` for all settings. TTS is cached, so changing only augmentation reuses synthesized speech.

## How models are evaluated

**The headline score is recall at the lowest tested threshold with no more than 0.5 false positives per hour (FPPH).** Higher is better; change the target with `--score-fpph`.

| Criterion | Measurement |
|---|---|
| Wake word detection | Recall: fraction of positives detected between phrase onset and 1 second after phrase end |
| Background false alarms | Triggers per hour of ordinary speech, recorded speech and noise; counted triggers are at least 2 seconds apart by default |
| Near-miss rejection | Fraction of adversarial clips that trigger; reported separately, excluded from FPPH |
| Response time | Trigger time relative to phrase end; negative latency means early detection |
| Compute cost | Processing time / audio duration; use `--workers 1` |

Reports break results down by noise, competing speech, room, device effects, voice, accent and phrasing, with 95% confidence intervals. Breakdowns use `--threshold` (default 0.5), which may differ from the headline score's threshold.

Synthesized positives do not establish recall on real users. Accent labels come from TTS metadata. Treat confidence intervals wider than about 0.2 as weak evidence; rare false alarms need many hours of negative audio.

## Evaluate on your dataset

For a livekit-wakeword v2 streaming model:

```bash
pip install livekit-wakeword
lkwb bench path/to/your_model.onnx --data out/hey_jarvis --limit 1000
```

Open `results/report.html` for charts and playable failures. Remove `--limit` for a full run before reporting results. Compare models using the same dataset, split, filters and limit.

Other models use the [custom detector interface](docs/benchmark.md#your-own-model). Existing datasets need no LiveKit credentials. Omitting `--data` uses the public [Hey LiveKit dataset](https://huggingface.co/datasets/binhpham/wakeword-bench-hey-livekit), which is an example benchmark for that phrase.

## Documentation

| Task | Guide |
|---|---|
| Generate and check your benchmark or dataset | [Generate](docs/generate.md) |
| Look up sample fields, sources and generation settings | [Reference](docs/reference.md) |
| Load, augment or export training data | [Train](docs/train.md) |
| Evaluate detectors and compare results | [Benchmark](docs/benchmark.md) |
| Contribute or release | [Development](docs/development.md) |

For coding agents, see [AGENTS.md](AGENTS.md).
