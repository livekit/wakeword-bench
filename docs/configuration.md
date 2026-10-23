# Configuration

To give your model more practice in reverberant rooms, increase `augment.p_reverb`. To make that experiment repeatable, save it in a config.

Generation settings use dotted names. Set them on the command line for a quick experiment, in Python for a training pipeline, or in YAML to keep a recipe in version control:

```bash
# 1. command line
uv run lkwb generate "hey jarvis" -o out/x --set augment.p_reverb=0.8 --set tts.plan=ship
```

```python
# 2. Python: dots become double underscores
from livekit_wakeword_bench import generate
generate("hey jarvis", "out/x", augment__p_reverb=0.8, tts__plan="ship")
```

```yaml
# 3. a YAML config, used with -c
wake_word: hey jarvis
tts:
  plan: ship
augment:
  p_reverb: 0.8
```

Precedence, from lowest to highest: built-in defaults, the YAML file, `--size`, `--plan`, then `--set` values.

`uv run lkwb knobs` prints every knob with its default; `lkwb knobs -c my.yaml` shows the values a config resolves to, and `--json` gives machine-readable output. Unknown keys are rejected: `unknown knob 'augment.p_reverbb'`.

Values are parsed as YAML, so numbers, booleans, lists and maps all work on the command line: `--set 'augment.sir_db=[0,10]'`, `--set 'augment.snr_weights={extreme: 3}'`.

## Phrase and pronunciation

| knob | default | meaning |
|---|---|---|
| `wake_word` | `hey livekit` | the phrase to detect |
| `name` | `hey_livekit` | dataset name |
| `positive_texts` | punctuation variants of the wake word | surface forms sent to TTS |
| `pronunciations` | `{}` | ARPAbet for words missing from the CMU dictionary, `\|` between word parts (see [Phoneme steering](phonemes.md)) |
| `hard_negatives` | `[]` | extra near-miss phrases, added to the automatic ones |
| `p_carrier` | 0.35 | share of wake words and near-misses placed inside a sentence |
| `language` | `en` | TTS language |
| `seed` | 1234 | random seed for sampling and augmentation |

## Text-to-speech

| knob | default | meaning |
|---|---|---|
| `tts.plan` | `build` | LiveKit Cloud plan: `build` (5 concurrent connections), `ship` (20), `scale` (50), `enterprise` (set `tts.concurrency`) |
| `tts.concurrency` | none | explicit connection limit; overrides `tts.plan` and the `LKWB_TTS_CONCURRENCY` environment variable |
| `tts.models` | all | restrict to some models, e.g. `["cartesia/sonic-3", "xai/tts-1"]` |
| `tts.test_voice_fraction` | 0.3 | share of each provider's voices reserved for the test split |
| `tts.speed` | `[0.85, 1.2]` | speaking-rate range |
| `tts.phonemes` | `true` | send wake words and near-misses as phonemes |
| `tts.sample_rate` | 24000 | TTS output rate (clips are stored at 16 kHz) |
| `tts.retries` | 3 | retries per failed request |

> [!WARNING]
> TTS connections are shared with production voice agents. Using your full connection allowance for generation can prevent agents from generating speech. Set `tts.concurrency` below the account limit to reserve capacity for production; the generator does not reserve it automatically.

## Augmentation

See [Augmentation](augmentation.md) for what each step does.

| knob | default | meaning |
|---|---|---|
| `augment.snr_buckets` | quiet 20–30, moderate 10–20, noisy 5–10, very_noisy 0–5, extreme −5–0 dB | noise level buckets |
| `augment.snr_weights` | `{}` (all 1.0) | relative weight of each bucket |
| `augment.clean_weight` | 1.0 | relative weight of no noise |
| `augment.p_reverb` | 0.5 | room reverb |
| `augment.p_interferer` | 0.2 | another voice |
| `augment.sir_db` | `[0, 20]` | how much quieter the other voice is, in dB |
| `augment.p_party` | 0.1 | party scene: several talkers over the speaker plus crowd babble |
| `augment.party_talkers` | `[2, 4]` | number of competing talkers in a party |
| `augment.party_sir_db` | `[5, 15]` | how much quieter each party talker is, in dB |
| `augment.party_babble_snr_db` | `[0, 10]` | crowd babble level below the speaker, in dB |
| `augment.p_second_noise` | 0.2 | a second, layered noise |
| `augment.p_eq` | 0.4 | microphone EQ |
| `augment.p_bandlimit` | 0.15 | band-limiting |
| `augment.level_dbfs` | `[-30, -3]` | peak level range |
| `augment.p_clip` | 0.05 | overdrive clipping |
| `augment.p_codec` | 0.25 | lossy codec |
| `augment.lead_s`, `augment.tail_s` | `[0.3, 1.5]`, `[0.4, 1.2]` | silence or noise before and after speech, in seconds |
| `augment.segment_s` | `[8, 30]` | length of real speech and noise-only clips, in seconds |

## Splits

Each split has the same knobs, under `splits.train`, `splits.validation` and `splits.test`:

| knob | meaning |
|---|---|
| `n_positive` | wake word clips |
| `n_adversarial` | near-miss clips |
| `n_speech` | everyday speech clips |
| `n_noise` | noise-only clips |
| `real_speech_minutes` | minutes of real conversation |
| `augment` | store augmented audio (`train` defaults to `false`) |

`--size` replaces all split sizes at once:

| size | train (pos / adv / speech) | validation (pos / adv / speech / noise / real min) | test (pos / adv / speech / noise / real min) |
|---|---|---|---|
| `tiny` | 40 / 40 / 10 | — | 100 / 100 / 30 / 10 / 5 |
| `small` | 1,000 / 1,000 / 300 | 100 / 100 / 30 / 20 / 10 | 500 / 500 / 150 / 50 / 30 |
| `default` | 10,000 / 10,000 / 3,000 | 1,000 / 1,000 / 300 / 100 / 30 | 3,000 / 3,000 / 1,000 / 300 / 180 |

Without `--size` or a config, the built-in defaults are train 2,000 / 2,000 / 500, validation 200 / 200 / 100 / 50 noise, test 1,000 / 1,000 / 300 / 200 noise / 60 real minutes.

## Sources and caching

| knob | default | meaning |
|---|---|---|
| `sources` | `[]` (all) | restrict audio sources by name (see [Audio sources](audio-sources.md)) |
| `cache_dir` | `.cache/lkwb` | TTS cache and downloaded sources |
| `sample_rate` | 16000 | stored audio rate |

## From Python

```python
from livekit_wakeword_bench import make_config, load_config
from livekit_wakeword_bench.config import set_knob

cfg = make_config("hey jarvis", size="small", plan="ship", knobs={"augment.p_reverb": 0.8})
cfg = make_config(config="configs/hey_livekit.yaml", knobs=["splits.test.n_positive=5000"])
cfg = set_knob(cfg, "augment.p_codec", 0.5)
cfg.tts.max_concurrency()          # resolved connection limit
```

`make_config` fills in the name, phrasing and guessed pronunciations for a bare wake word, and raises if a word still has no pronunciation.

## Benchmark settings

Benchmark scoring is configured separately, with `lkwb bench` flags or `BenchConfigRun`:

| flag | `BenchConfigRun` field | default | meaning |
|---|---|---|---|
| `--threshold` | `threshold` | 0.5 | operating threshold for breakdowns and failure examples |
| `--score-fpph` | `score_fpph` | 0.5 | false-alarm target for `score` |
| `--debounce` | `debounce_s` | 2.0 | seconds between counted triggers |
| | `max_latency_s` | 1.0 | how long after the wake word a trigger still counts |
| | `target_fpph` | (1.0, 0.5, 0.1) | extra false-alarm targets reported in the summary |
| | `fpph_categories` | real_speech, speech, noise | negative categories counted in false alarms per hour |
