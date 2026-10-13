# LiveKit WakeWord Bench

**Find where a wake word model fails, then generate the data to fix it.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-dataset-yellow)](https://huggingface.co/datasets/binhpham/wakeword-bench-hey-livekit)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/teaser_dark.svg">
  <img alt="One clip is detected alone and missed with a second voice; recall falls from 83% to 30% as the second voice gets louder" src="assets/teaser_light.svg">
</picture>

**Figure 1. A second voice in the room silences the model.** Left: one test clip scores 0.97 alone and 0.01 when another person speaks 5 dB quieter than the speaker. Right: over all 3,000 test wake words, recall falls from 83% to 30% as that voice gets louder.

A wake word gates every hands-free voice agent. The device listens all day and must respond to one phrase. Wake word models are usually judged by one accuracy number on clean recordings. Unfortunately, one number hides *where* a model fails, and the failures that matter (other voices, rooms, accents, near-miss phrases) are rare in clean test sets.

In contrast, livekit-wakeword-bench **generates every test clip from a known recipe**: voice, phrase, room, noise, codec. Each failure therefore becomes a measurable condition, and the same generator produces training data for exactly that condition. Generated speech, however, mispronounces brand names, and generated tests leak into training. Therefore the generator sends wake words to text-to-speech as phonemes, and keeps test voices, rooms and noise out of training.

On livekit-wakeword's `hey_livekit` model, the benchmark finds 76% recall overall, 30% with a loud second voice, and 59% of near-miss phrases accepted.

**Contributions**

- **A benchmark** with one score (recall at ≤ 0.5 false alarms per hour) and recall per condition with 95% confidence intervals. The public test set has 7,873 clips (12.3 h) of held-out voices, rooms and noise.
- **A generator** for any wake word: 157 voices from 4 TTS providers steered with phonemes, real noise at controlled levels, room reverb, competing talkers, microphone and codec effects.
- **A model-agnostic harness**: a model receives a clip and returns scores over time. Output is JSON, and `compare` reports per-condition changes between two models.

**Quick links:** [Results](#results-hey_livekit) · [Benchmark your model](#benchmark-your-model) · [Generate a dataset](#generate-a-dataset) · [Use the data](#use-the-data) · [Knobs](#knobs) · [For agents](#for-agents) · [How it works](#how-it-works)

## Results: `hey_livekit`

The goal is a model that wakes on every "Hey LiveKit" and almost never otherwise. An always-listening device tolerates about one false alarm every two hours, so the benchmark scores a model by its recall at that rate:

$$\text{score} = \max_{\theta}\ \text{recall}(\theta) \quad \text{subject to} \quad \text{false alarms per hour}(\theta) \le 0.5$$

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/tradeoff_dark.svg">
  <img alt="Recall vs false alarms per hour" src="assets/tradeoff_light.svg">
</picture>

**Figure 2. The model trades recall for silence.** At the default threshold it catches 76% of wake words but fires 29 times per hour. Within ≤ 0.5 false alarms per hour (shaded), it catches 44.5%: the score.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/failures_dark.svg">
  <img alt="Recall per condition with 95% confidence intervals" src="assets/failures_light.svg">
</picture>

**Figure 3. Phrasing, accent, loud noise and reverberation each cost 8 to 23 points of recall.** Red and green bars differ from the overall 76% (dashed line); gray bars do not. Whiskers are 95% confidence intervals.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/near_misses_dark.svg">
  <img alt="Share of near-miss phrases accepted" src="assets/near_misses_light.svg">
</picture>

**Figure 4. The model does not check which words are said.** Replacing any one word of "hey live kit" still triggers it 58 to 62% of the time; dropping a word triggers it less often (32%).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/alarm_sources_dark.svg">
  <img alt="False alarms per hour by audio source" src="assets/alarm_sources_light.svg">
</picture>

**Figure 5. Speech, not noise, causes the false alarms.** False alarms per hour at the default threshold, by source.

### What to fix

| failure | evidence | training data that targets it |
|---|---|---|
| second voice | 83% → 30% (Fig. 1) | `--set augment.p_interferer=0.6` |
| wake word after other speech | 63% vs 83% when it comes first (Fig. 3) | `--set p_carrier=0.6` |
| near-miss phrases | 59% accepted (Fig. 4) | the `adversarial` category, plus `hard_negatives` |
| UK accents, reverberant rooms, −5 to 0 dB noise | 59%, 72%, 62% (Fig. 3) | `--set augment.p_reverb=0.8`, more `extreme` noise via `augment.snr_weights` |

The knobs act on training audio when it is augmented: `--set splits.train.augment=true` at generation, or `Augment(spec)` at training time. Codecs (Opus, MP3, 8 kHz telephone), band-limiting and microphone EQ show no significant effect.

### Limitations

- **Synthetic wake words.** The positives come from TTS, not human recordings, so absolute recall on real users differs. Differences between conditions are the reliable signal. Recorded human wake words are future work.
- **Few negative hours.** False alarms are measured on 6.1 h of negative audio. Zero false alarms at threshold 0.94 bounds the rate at ≤ 0.49 per hour (95% confidence), and does not separate 0.5 from 0.1 per hour. More real speech hours tighten this.
- **Accent labels** come from TTS provider metadata.

Reproduce: `lkwb bench -d livekit:hey_livekit.onnx`, then `uv run --extra livekit python scripts/readme_figures.py results/livekit_hey_livekit --dataset binhpham/wakeword-bench-hey-livekit --model hey_livekit.onnx`.

## Benchmark your model

A model receives a whole clip and returns a score timeline. How it streams the audio is up to the model.

```python
import numpy as np
from livekit_wakeword_bench import Prediction, benchmark

class MyModel:
    name = "my-model"

    def detect(self, audio: np.ndarray, sample_rate: int) -> Prediction:
        times, scores = run_my_model(audio, sample_rate)   # scores[i] after times[i] seconds
        return Prediction(times, scores)                   # or Prediction.from_events([1.9, 7.4])

res = benchmark(MyModel(), limit=500)     # public test set, random 500 clips
res.summary()["score"]                    # the number to maximize
res.breakdown()["positive/interferer"]    # recall per condition, with 95% CIs
res.save("results/my-model")
```

The same from the command line, with `my_pkg/models.py` defining `make_model()` that returns `MyModel()`:

```bash
uv run lkwb bench -d my_pkg.models:make_model --limit 500   # quick
uv run lkwb bench -d my_pkg.models:make_model               # full test set
uv run lkwb figures results/my-model                        # the charts above, for your model
```

Built-in adapters: `livekit:<model.onnx>` (needs `uv sync --extra livekit`) and `openwakeword:<name>`. Each run writes `results/<model>/` (JSON, per-clip outcomes, score traces) and `results/report.html` with playable failures. See [examples/](examples/).

## Generate a dataset

**1. Credentials.** Speech comes from [LiveKit Inference](https://docs.livekit.io/agents/models/tts/inference/), which needs a LiveKit Cloud project. The free plan works.

```bash
brew install livekit-cli                      # macOS
curl -sSL https://get.livekit.io/cli | bash   # Linux
winget install LiveKit.LiveKitCLI             # Windows

lk cloud auth      # link your LiveKit Cloud project (once)
lk app env -w      # write LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET to .env.local
```

With several projects, `lk project set-default <name>` selects one before `lk app env -w`.

**2. Plan.** LiveKit Inference limits concurrent TTS connections per [plan](https://livekit.com/pricing), and generation stays within the limit.

| plan | flag | concurrent TTS | speed |
|---|---|---|---|
| Build (free) | `--plan build` (default) | 5 | ~4 utterances/s |
| Ship | `--plan ship` | 20 | ~15/s |
| Scale | `--plan scale` | 50 | ~30/s |
| Enterprise | `--set tts.concurrency=N` | your limit | |

**3. Generate.**

```python
from livekit_wakeword_bench import generate
generate("hey jarvis", "out/hey_jarvis", size="small", plan="ship")
```

```bash
uv run lkwb generate "hey jarvis" --size small --plan ship -o out/hey_jarvis
```

| size | train | validation | test |
|---|---|---|---|
| `tiny` | 90 | — | ~250 |
| `small` | 2.3k | ~280 | ~1.3k |
| `default` | 23k | ~2.5k | ~7.9k |

- Words missing from the pronunciation dictionary need an entry: `lkwb init "hey jarvis"` writes a config with a guess, and `lkwb pronounce` prints what each TTS receives.
- TTS results are cached, so regenerating with different augmentation costs nothing.
- `lkwb verify out/hey_jarvis` checks the audio against its labels; `lkwb push out/hey_jarvis you/repo` publishes it.

## Use the data

```python
from livekit_wakeword_bench import DEFAULT_DATASET, WakeWordData
from livekit_wakeword_bench.loader import Augment

test = WakeWordData(DEFAULT_DATASET, split="test")             # or a local directory
hard = test.filter(category="positive", noise_bucket=["very_noisy", "extreme"])

train = WakeWordData(DEFAULT_DATASET, split="train").with_transform(Augment()).windows(2.0)
x, y = train.take(512).arrays()                                 # numpy
loader_ds = train.torch()                                       # DataLoader(..., collate_fn=WakeWordData.collate)
```

`Augment()` applies fresh noise, reverb, talkers, EQ and codecs on every access, from training sources only. `lkwb export-livekit` writes the livekit-wakeword training layout; `lkwb export-wav` writes WAV files and a CSV.

## Knobs

Every config key is a knob: `--set key=value` on the command line, `key__sub=value` in `generate(...)`, or YAML with `-c`. `lkwb knobs` lists all of them.

| knob | default | effect |
|---|---|---|
| `splits.<split>.n_positive` / `n_adversarial` / `n_speech` / `n_noise` | per `--size` | clips per category |
| `splits.<split>.real_speech_minutes` | per `--size` | real conversation (false alarm denominator) |
| `splits.<split>.augment` | train: `false` | store augmented audio |
| `augment.snr_buckets` / `snr_weights` / `clean_weight` | −5 to 30 dB, uniform | noise level distribution |
| `augment.p_reverb` / `p_interferer` / `p_second_noise` | 0.5 / 0.2 / 0.2 | room, second voice, layered noise |
| `augment.p_eq` / `p_bandlimit` / `p_clip` / `p_codec` | 0.4 / 0.15 / 0.05 / 0.25 | microphone and transmission |
| `tts.plan` / `tts.concurrency` / `tts.models` | build / – / all | TTS |
| `pronunciations` / `hard_negatives` / `p_carrier` | | phrasing |

## For agents

`lkwb bench --json` prints the score and the worst conditions; `lkwb compare base/ new/ --json` lists per-condition changes between two runs, worst first. [AGENTS.md](AGENTS.md) describes the full loop: benchmark, read the worst conditions, generate targeted data, train, compare.

## How it works

<details>
<summary><b>Dataset</b></summary>

| category | label | content |
|---|---|---|
| `positive` | 1 | the wake word, alone or in a sentence |
| `adversarial` | 0 | phonetic near-misses from CMUdict ("hey live kid", "hey lime kit") |
| `speech` | 0 | everyday TTS sentences |
| `real_speech` | 0 | meetings, phone calls, web audio |
| `noise` | 0 | background noise only |

Each row holds 16 kHz audio and its recipe: `noise_bucket`, `snr_db`, `noise_type`, `reverb`, `interferer`, `codec`, `voice_name`, `accent`, `phrase`, `phonemes`, `tts_input`, and the exact wake word timing used for latency (`SCHEMA` in [`dataset.py`](src/livekit_wakeword_bench/dataset.py)).
</details>

<details>
<summary><b>Phoneme steering</b></summary>

| provider | input | voices |
|---|---|---|
| Cartesia `sonic-3` | `<<ˈ\|l\|aɪ\|v\|ˌ\|k\|ɪ\|t>>` | 43 |
| Inworld `inworld-tts-2` | `/ˈlaɪvˌkɪt/` | 26 |
| xAI `tts-1` | `/ˈlaɪvˌkɪt/` | 26 |
| Rime `mistv3` | `{l1Yv} {k1It}` | 62 |

Phonemes come from CMUdict and the `pronunciations` config (ARPAbet, `|` between morphemes). Each provider passed a test that synthesizes words from phonemes alone. Deepgram, Fish Audio and Rime Coda failed it and are excluded.
</details>

<details>
<summary><b>Augmentation</b></summary>

In physical order: room reverb → second voice → noise at an SNR bucket (+ a second noise) → microphone EQ → band-limit → gain → clipping → codec (Opus 8/16/32k, MP3, 8 kHz μ-law). Every step that fires is recorded per clip. `Augment()` uses the same chain for training.
</details>

<details>
<summary><b>Sources: train and test never share audio</b></summary>

| split | noise | rooms | real speech |
|---|---|---|---|
| train / validation | MUSAN, DNS-Noise | MIT RIRs | LibriSpeech |
| test | DEMAND, WHAM!, ESC-50, GTZAN | OpenSLR 28, Treble10 | AMI, Earnings-22, People's Speech |

WHAM!, ESC-50, Treble10 and GTZAN are non-commercial or research-only; `sources:` in the config restricts them.
</details>

<details>
<summary><b>Metrics</b></summary>

| metric | definition |
|---|---|
| score | recall at the lowest threshold with ≤ 0.5 false alarms per hour |
| recall | wake words triggered between onset and offset + 1 s |
| false alarms per hour | triggers per hour of speech and noise, 2 s debounce |
| near-misses accepted | share of adversarial clips that trigger |
| latency | first trigger minus end of wake word |
| RTF | compute time ÷ audio duration |
</details>

## Development

```bash
uv sync --all-extras --group dev
uv run pytest
uv run --group dev python scripts/qa_dataset.py out/hey_jarvis --plots out/qa
```

The library is pure Python (numpy, scipy, soundfile) and runs on any platform. Whisper and matplotlib are development-only.
