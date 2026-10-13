# Loading data

You can inspect which clips contain loud noise without decoding hours of audio, then load just those clips to listen or train. `WakeWordData` reads metadata first and decodes audio when you access a sample. Filtering, shuffling and windowing return new views.

```python
from livekit_wakeword_bench import DEFAULT_DATASET, WakeWordData

test = WakeWordData(DEFAULT_DATASET, split="test")      # HF repo id or local directory
len(test)                                               # 7873
clip = test[0]                                          # Sample(audio, sample_rate, label, meta)
```

`DEFAULT_DATASET` is the public `binhpham/wakeword-bench-hey-livekit`. Clips are 16 kHz float32 mono; pass `sample_rate=` to resample on load.

## Selecting clips

```python
hard = test.filter(category="positive", noise_bucket=["very_noisy", "extreme"])
uk = test.filter(accent="UK", reverb=True)
quiet_voice = test.filter(lambda m: m["sir_db"] > 15)
subset = test.shuffle(0).take(500)
```

Keyword filters match a column exactly, or any of a list. A function filter receives the clip metadata. Column names are listed in [Dataset format](dataset-format.md).

`test.metadata()` returns all metadata as a pandas DataFrame without decoding any audio, which is the fastest way to explore a split.

## Fixed-length windows

Most models train on fixed-length inputs. `.windows(seconds)` cuts every clip to that length:

- **wake word clips** end 0 to 250 ms after the wake word, which is what a streaming model sees at the moment it should fire;
- **negative clips** are cropped at a random position;
- clips shorter than the window are padded with silence at the start.

```python
train = WakeWordData(DEFAULT_DATASET, split="train").windows(2.0)
x, y = train.take(512).arrays()          # x: (512, 32000) float32, y: (512,) int64
for xb, yb, metas in train.batches(64):  # equal-length batches
    ...
```

## Augmenting on the fly

The train split is stored clean. `Augment()` adds a fresh random mix of reverb, another voice, background noise, microphone EQ, band-limiting, clipping and codecs every time a clip is read, using training sources only (see [Augmentation](augmentation.md)).

```python
from livekit_wakeword_bench.loader import Augment

train = WakeWordData(DEFAULT_DATASET, split="train").with_transform(Augment()).windows(2.0)
```

| argument | default | meaning |
|---|---|---|
| `spec` | default augmentation settings | an `AugmentSpec`, e.g. `load_config("my.yaml").augment` |
| `cache_dir` | `.cache/lkwb` | where the audio sources live |
| `sources` | all | restrict to some source names |

The first use downloads the training sources (a few GB). Each DataLoader worker loads its own copy lazily.

A custom transform takes `fn(audio, meta, rng) -> audio`. Pass it to `with_transform()`. Calling `with_transform()` again replaces the previous transform; combine multiple effects inside one function if you need them together.

## PyTorch

```python
from torch.utils.data import DataLoader

ds = train.torch()          # or WakeWordData(...).torch(window_s=2.0)
loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=4,
                    collate_fn=WakeWordData.collate)
for audio, labels, metas in loader:   # audio: (64, 32000) tensor
    ...
```

Install with `uv sync --extra torch`. The dataset object is picklable, so `num_workers > 0` works.

## Exporting

For tools that expect files on disk:

```bash
# livekit-wakeword's training layout: positive_train/, negative_train/, positive_test/, negative_test/
uv run lkwb export-livekit out/hey_jarvis <livekit-wakeword output_dir> --model-name hey_jarvis

# plain WAV files plus a CSV with every metadata column
uv run lkwb export-wav out/hey_jarvis exported/ --split test
```

`export-livekit` writes the train split as `*_train` and the validation split as `*_test`, cropped to the spoken utterance. Negatives include near-miss phrases and everyday speech. Run livekit-wakeword's augment and train steps on that folder as usual.

## Keep evaluation separate

Use `train` for fitting the model and `validation` for development checks. Never train on the `test` split or test sources. Keep that split fixed so [benchmark comparisons](benchmarking.md#comparing-two-models) measure changes to the model on held-out audio.
