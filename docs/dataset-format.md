# Dataset format

If you're investigating a missed wake word, the audio alone won't tell you how the clip was made. Each row stores the audio alongside its phrase, voice, timing and augmentation settings.

Datasets use Parquet shards, with the same layout locally and on the Hugging Face Hub. Load one by directory path or repo id:

```
my_dataset/
  README.md                    dataset card: splits, composition, sources, generation config
  bench_config.yaml            the exact config used to generate it
  data/
    train-00000.parquet        about 1,000 clips per shard
    validation-00000.parquet
    test-00000.parquet
    …
```

## Loading it

Use `WakeWordData` to decode clips and access their metadata:

```python
from livekit_wakeword_bench import WakeWordData

data = WakeWordData("binhpham/wakeword-bench-hey-livekit", split="test")   # or "my_dataset"
clip = data[0]
clip.audio, clip.sample_rate, clip.label, clip.meta["noise_bucket"]
```

With plain `datasets`:

```python
import datasets

ds = datasets.load_dataset("binhpham/wakeword-bench-hey-livekit", split="test")
```

`datasets` 4.x needs `torchcodec` to decode the `audio` column. To read metadata only, or decode the FLAC bytes yourself, cast the column first: `ds.cast_column("audio", datasets.Audio(decode=False))`.

With pyarrow or pandas, read the shards directly; `audio` is a struct of FLAC `bytes` and a `path`.

## Categories

| category | label | content |
|---|---|---|
| `positive` | 1 | the wake word, alone or in a sentence |
| `adversarial` | 0 | near-miss phrases |
| `speech` | 0 | everyday synthetic speech |
| `real_speech` | 0 | real recorded conversation |
| `noise` | 0 | background noise only |

## Columns

Every row has every column. When a column doesn't apply it holds `""`, `NaN`, `-1` or `false`.

**Identity**

| column | type | meaning |
|---|---|---|
| `id` | str | unique clip id, e.g. `test-positive-000035` |
| `split` | str | `train`, `validation` or `test` |
| `label` | int | 1 if the clip contains the wake word |
| `category` | str | see above |
| `audio` | struct | 16 kHz mono FLAC (`bytes`, `path`) |
| `duration_s` | float | clip length in seconds |

**Speech**

| column | type | meaning |
|---|---|---|
| `wake_word` | str | the dataset's wake word |
| `text` | str | everything that was spoken (for real speech: the transcript, if any) |
| `text_form` | str | `isolated`, `carrier_prefix` (wake word first), `carrier_suffix` (wake word last), `filler`, `transcript` |
| `phrase` | str | the key phrase: the wake word, or the near-miss phrase |
| `adversarial_distance` | int | how many sounds the near-miss differs from the wake word |
| `phonemes` | str | ARPAbet of the key phrase (words split by ` . `, word parts by ` \| `) |
| `tts_input` | str | the exact key phrase string sent to TTS |
| `tts_key` | str | cache keys of the TTS segments (`\|`-separated) |
| `provider`, `model` | str | TTS provider and model, or the real speech corpus |
| `voice_id`, `voice_name` | str | the voice |
| `gender`, `accent` | str | voice metadata from the provider |
| `held_out_voice` | bool | the voice never appears in train or validation |
| `speed` | float | speaking-rate multiplier |

**Timing**

| column | type | meaning |
|---|---|---|
| `onset_s`, `offset_s` | float | start and end of the whole utterance in the clip |
| `phrase_onset_s`, `phrase_offset_s` | float | start and end of the wake word (or near-miss) itself; used for recall windows and latency |

**Room, scene and device** (see [Augmentation](augmentation.md))

| column | type | meaning |
|---|---|---|
| `reverb`, `rir_source`, `rir_file`, `rir_category` | bool, str | room reverb and which room |
| `interferer`, `sir_db` | str, float | competing voice corpus and its level below the speaker (in a party: every corpus, and the loudest talker) |
| `party`, `n_talkers` | bool, int | party scene: this many talkers over the speaker plus crowd babble (`noise_bucket` is `party`) |
| `noise_bucket`, `snr_db` | str, float | noise level bucket and exact SNR |
| `noise_source`, `noise_type`, `noise_category`, `noise_file` | str | which noise recording |
| `noise2_category` | str | second layered noise, if any |
| `eq` | str | microphone EQ applied, if any |
| `bandlimit` | str | `none`, `telephone`, `lowpass_4k`, `lowpass_6k`, `highpass_200` |
| `level_dbfs` | float | peak level of the clip |
| `clipping` | bool | input overdriven |
| `codec` | str | `none`, `opus_8k`, `opus_16k`, `opus_32k`, `mp3_32k`, `mulaw_8khz` |
| `augmentations` | str | effects applied, e.g. `reverb+talker+noise` |

The authoritative list is `SCHEMA` in [`src/livekit_wakeword_bench/dataset.py`](../src/livekit_wakeword_bench/dataset.py).

## Publishing

`lkwb push <dataset> <user>/<repo>` uploads the card, the config and the shards. Repos are private unless you pass `--public`.
