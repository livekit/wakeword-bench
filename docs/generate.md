# Generate

Create wake word clips with synthetic speech, then mix in real noise, rooms and competing speakers.

## Connect LiveKit

Generation uses [LiveKit Inference](https://docs.livekit.io/agents/models/tts/inference/) credits. Install the package ([quick start](../README.md#quick-start)) and the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/), then connect your project:

```bash
lk cloud auth
lk app env -w  # write credentials to .env.local
```

## Make a dataset

```bash
lkwb generate "hey jarvis" --size small -o out/hey_jarvis
```

Or use Python:

```python
from livekit_wakeword_bench import generate

generate("hey jarvis", "out/hey_jarvis", size="small", plan="build")
```

Add `--split test` for a benchmark only, or `--split train` for training data only. Without `--split`, all configured splits are generated. Example configs are in [configs/](../configs/):

```bash
lkwb generate -c configs/alexa.yaml -o out/alexa
```

The output directory contains `data/<split>-*.parquet` shards, `bench_config.yaml` with the generation recipe, and a dataset card. Rows include audio, labels, phrase timestamps, voice information and augmentation metadata; see [dataset columns](reference.md#dataset-columns). Use the directory with `lkwb bench --data`, `WakeWordData`, or the export commands.

## Choose sample counts

Choose a preset explicitly with `--size`. Counts below are fixed except recorded speech, which is requested in minutes and cut into variable-length clips.

| Size | Split | Positive | Adversarial | Synthesized speech | Noise | Recorded speech (minutes) |
|---|---|---:|---:|---:|---:|---:|
| `tiny` | train | 40 | 40 | 10 | 0 | 0 |
| | test | 100 | 100 | 30 | 10 | 5 |
| `small` | train | 1,000 | 1,000 | 300 | 0 | 0 |
| | validation | 100 | 100 | 30 | 20 | 10 |
| | test | 500 | 500 | 150 | 50 | 30 |
| `default` | train | 10,000 | 10,000 | 3,000 | 0 | 0 |
| | validation | 1,000 | 1,000 | 300 | 100 | 30 |
| | test | 3,000 | 3,000 | 1,000 | 300 | 180 |

`tiny` has no validation split. Omitting `--size` uses the config's counts, which need not match `--size default`.

Override individual counts after choosing a preset:

```bash
lkwb generate "hey jarvis" --size small --split test -o out/jarvis_benchmark \
    --set splits.test.n_positive=2000 \
    --set splits.test.n_adversarial=2000 \
    --set splits.test.real_speech_minutes=180
```

See [what gets generated](../README.md#what-gets-generated) for the purpose of each category and [evaluation criteria](../README.md#how-models-are-evaluated) for how it contributes to metrics.

TTS audio is cached in `.cache/lkwb/tts`. Changing only augmentation reuses cached speech without new TTS charges.

## Set TTS capacity

Choose `--plan` to match your account. The tool's concurrency presets are:

| Plan | Connections |
|---|---|
| `build` (default) | 5 |
| `ship` | 20 |
| `scale` | 50 |
| `enterprise` | Set `tts.concurrency` explicitly |

TTS connections are shared with production voice agents. Leave capacity for them and never exceed your account's limit:

```bash
lkwb generate "hey jarvis" --size small -o out/hey_jarvis --set tts.concurrency=2
```

## Set pronunciation

The generator sends the wake word as phonemes to guide pronunciation across voices. For unknown names or made-up words, create and edit a config:

```bash
lkwb init "ok zorblax"  # writes ok_zorblax.yaml and marks unknown words
```

Add an ARPAbet pronunciation (`1` marks primary stress; `|` separates word parts):

```yaml
pronunciations:
  zorblax: "Z AO1 R B L AE2 K S"
```

Inspect the provider inputs before generating:

```bash
lkwb pronounce "ok zorblax" -c ok_zorblax.yaml
```

## Understand the splits

Each dataset includes wake words (label `1`) and negatives (label `0`): similar-sounding phrases, ordinary speech, recorded speech and noise.

Training clips are stored clean by default; [add fresh effects while training](train.md). Validation and test clips include noise, echo, competing voices and device effects. Test voices, noise, rooms and recorded speakers are separate from training.

To store more competing speech in a training dataset:

```bash
lkwb generate "hey livekit" -o out/train_v2 --split train \
    --set splits.train.augment=true \
    --set augment.p_interferer=0.6 \
    --set augment.clean_weight=0.2
```

See [settings](reference.md#settings) for more controls and [audio sources](reference.md#audio-sources) for licensing restrictions.

## Check and share

```bash
lkwb verify out/hey_jarvis
lkwb push out/hey_jarvis you/wakeword-bench-hey-jarvis --public
```

`verify` checks audio, labels and split overlap. Listen to samples as well, especially unusual pronunciations. `push --public` uploads a public Hugging Face dataset.
