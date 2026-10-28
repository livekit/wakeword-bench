# Train

Use this package to load, augment and export data; supply your own model training code. Never train on the `test` split or test sources.

## Load clips

```python
from livekit_wakeword_bench import DEFAULT_DATASET, WakeWordData
from livekit_wakeword_bench.loader import Augment

train = WakeWordData(DEFAULT_DATASET, split="train")  # or "out/hey_jarvis"
train = train.with_transform(Augment()).windows(2.0)
x, y = train.take(512).arrays()  # audio: (N, 32000), labels: (N,); N <= 512
```

`Augment()` adds fresh noise, echo, voices and device effects on each read, using training sources. `.windows(2.0)` crops or pads clips to 2 seconds, placing positive windows near the end of the wake word.

Use `.filter(category="positive")` to select clips by stored metadata. Filters do not select effects added later by `Augment()`.

## Use PyTorch

Install the optional dependency:

```bash
pip install 'livekit-wakeword-bench[torch]'
```

With `train` from the example above:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    train.torch(), batch_size=64, shuffle=True,
    num_workers=0, collate_fn=WakeWordData.collate,
)
```

Each batch contains `(audio, labels, metadata)`. To use worker processes, increase `num_workers` and put DataLoader creation and iteration inside an `if __name__ == "__main__":` guard in your script.

## Target a weak condition

Change augmentation settings based on benchmark results:

| Weak condition | Setting to try |
|---|---|
| Competing speaker | `augment.p_interferer=0.6` |
| Several speakers | `augment.p_party=0.3` |
| Too many clean examples | `augment.clean_weight=0.2` |
| Loud noise | Increase `augment.snr_weights.extreme` |
| Echo | `augment.p_reverb=0.8` |
| Compressed audio | `augment.p_codec=0.5` |

For example, add competing speakers more often:

```python
from livekit_wakeword_bench import load_config
from livekit_wakeword_bench.config import set_knob

config = set_knob(load_config(None), "augment.p_interferer", 0.6)
train = WakeWordData(DEFAULT_DATASET, split="train")
train = train.with_transform(Augment(spec=config.augment)).windows(2.0)
```

You can also [generate augmented training clips](generate.md#understand-the-splits). After training, [compare the candidate against the baseline](benchmark.md#compare-models) on the same test data.

## Export audio

```bash
lkwb export-livekit out/hey_jarvis exported/livekit --model-name hey_jarvis
lkwb export-wav out/hey_jarvis exported/wav --split train
```

`export-livekit` writes the livekit-wakeword training layout. `export-wav` writes WAV files and a CSV.
