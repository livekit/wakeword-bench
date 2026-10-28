# Train

## Load clips

```python
from livekit_wakeword_bench import DEFAULT_DATASET, WakeWordData
from livekit_wakeword_bench.loader import Augment

train = WakeWordData(DEFAULT_DATASET, split="train")   # or "out/hey_jarvis"
train = train.with_transform(Augment()).windows(2.0)
x, y = train.take(512).arrays()                         # numpy: (512, 32000), (512,)
```

- `.windows(2.0)` cuts every clip to 2 s. Wake word clips end just after the wake word.
- `Augment()` adds fresh noise, room echo, other voices and microphone effects every time a clip is read, using only training audio.
- `.filter(...)` picks clips by any column: `train.filter(category="positive", noise_bucket="extreme")`.

## PyTorch

```python
from torch.utils.data import DataLoader

loader = DataLoader(train.torch(), batch_size=64, shuffle=True, num_workers=4,
                    collate_fn=WakeWordData.collate)
```

Install with `uv sync --extra torch`.

## Train more on a weak spot

If the benchmark shows the model failing in one condition, raise it in the training data:

| weak spot | setting |
|---|---|
| another voice | `augment.p_interferer=0.6` |
| several voices | `augment.p_party=0.3` |
| loud noise | `augment.clean_weight=0.2` |
| echoey rooms | `augment.p_reverb=0.8` |
| phone audio | `augment.p_codec=0.5` |

```python
from livekit_wakeword_bench import load_config
from livekit_wakeword_bench.config import set_knob

spec = set_knob(load_config(None), "augment.p_interferer", 0.6).augment
train = WakeWordData(DEFAULT_DATASET, split="train").with_transform(Augment(spec=spec))
```

## Export for livekit-wakeword

```bash
uv run lkwb export-livekit out/hey_jarvis <output_dir> --model-name hey_jarvis   # livekit-wakeword's layout
uv run lkwb export-wav out/hey_jarvis exported/ --split train                   # WAV files + a CSV
```

Never train on the `test` split. It only measures how well the model generalizes if the model has never heard those clips.
