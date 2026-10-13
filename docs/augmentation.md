# Augmentation

Clean TTS speech is not what a device hears. After synthesis, each clip passes through a chain of effects in the order sound actually travels: the room, then the scene around the speaker, then the device that records it, then the network that carries it.

```
speech ─► room reverb ─► + another voice ─► + background noise (+ a second noise)
       ─► microphone EQ ─► band-limit ─► input gain ─► clipping ─► codec
```

Each step fires with its own probability. Whatever fires is recorded on the clip, so every effect can be measured separately, and `augmentations` summarizes the combination (for example `reverb+talker+noise+codec`).

## The steps

| step | what it simulates | setting (default) | recorded as |
|---|---|---|---|
| room reverb | the room the speaker is in | `augment.p_reverb` (0.5) | `reverb`, `rir_source`, `rir_category` |
| another voice | a TV, a colleague, a meeting in the background | `augment.p_interferer` (0.2), `augment.sir_db` (0–20 dB) | `interferer`, `sir_db` |
| background noise | traffic, a café, an appliance, music | `augment.snr_buckets`, `snr_weights`, `clean_weight` | `noise_bucket`, `snr_db`, `noise_type`, `noise_category` |
| second noise | two noises at once (traffic and music) | `augment.p_second_noise` (0.2) | `noise2_category` |
| microphone EQ | a cheap or unusual microphone | `augment.p_eq` (0.4) | `eq` |
| band-limit | telephone, far-field or filtered audio | `augment.p_bandlimit` (0.15) | `bandlimit` |
| input gain | quiet or loud recording levels | `augment.level_dbfs` (−30 to −3 dB) | `level_dbfs` |
| clipping | an overdriven input | `augment.p_clip` (0.05) | `clipping` |
| codec | compressed calls and streams | `augment.p_codec` (0.25) | `codec` |

### Noise levels

Each clip picks one noise level bucket, then a random signal-to-noise ratio (SNR) inside it:

| bucket | SNR | meaning |
|---|---|---|
| `clean` | no noise | |
| `quiet` | 20–30 dB | noise barely noticeable |
| `moderate` | 10–20 dB | clearly audible |
| `noisy` | 5–10 dB | hard to ignore |
| `very_noisy` | 0–5 dB | noise almost as loud as speech |
| `extreme` | −5–0 dB | noise louder than speech |

All buckets are equally likely by default, so every level gets enough clips to measure. Change the mix with `augment.snr_weights` (for example `{"extreme": 3}`) and `augment.clean_weight`, or redefine the buckets with `augment.snr_buckets`.

SNR and the other voice's level (SIR) are measured against the speaker's *active* level: the loudest parts of the speech, not silence. 10 dB SNR means the speech is 10 dB louder than the noise while someone is talking.

### Noise types

Noise is sampled evenly by type first (babble, music, street, transport, domestic, office, nature, public, animals, human), then by source, then by recording. Large collections such as ESC-50's 50 classes therefore cannot crowd out small ones.

### Microphone and transmission

- **EQ** applies a random spectral tilt (±6 dB) plus one to three peaking filters (150 Hz–6 kHz, ±9 dB).
- **Band-limit** is one of: telephone (300–3,400 Hz), 4 kHz low-pass, 6 kHz low-pass, 200 Hz high-pass.
- **Codecs** are Opus at three quality levels (`opus_8k`, `opus_16k`, `opus_32k`, from lowest to highest), low-bitrate MP3 (`mp3_32k`), and 8 kHz μ-law telephone (`mulaw_8khz`). They are encoded and decoded with libsndfile, so no external tools are needed and timestamps stay exact (no codec delay).

## Which splits are augmented

| split | default | why |
|---|---|---|
| test, validation | augmented | the benchmark needs realistic conditions |
| train | clean | training code adds fresh effects every epoch with `Augment()` |

Store augmented training audio instead with `--set splits.train.augment=true`.

## Augmenting during training

`Augment()` applies the same chain to training clips each time they are read, using **training sources only**:

```python
from livekit_wakeword_bench import DEFAULT_DATASET, WakeWordData, load_config
from livekit_wakeword_bench.loader import Augment

train = WakeWordData(DEFAULT_DATASET, split="train")
train = train.with_transform(Augment())                                   # default settings
train = train.with_transform(Augment(spec=load_config("my.yaml").augment))  # your settings
```

Each access draws new random effects, so the model sees a different version of every clip in every epoch. See [Loading data](loading-data.md).

## Targeting a weakness

If a benchmark shows the model failing in a condition, raise that condition in training data:

| weakness | settings |
|---|---|
| another voice | `augment.p_interferer=0.6`, `augment.sir_db=[0,10]` |
| loud noise | `augment.snr_weights={"extreme":3,"very_noisy":2}`, `augment.clean_weight=0.2` |
| reverberant rooms | `augment.p_reverb=0.8` |
| phone audio | `augment.p_bandlimit=0.5`, `augment.p_codec=0.5` |

```bash
uv run lkwb generate "hey livekit" -o out/train_v2 --split train \
    --set splits.train.augment=true --set augment.p_interferer=0.6 --set augment.clean_weight=0.2
```
