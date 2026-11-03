# Reference

## Commands

Use `lkwb <command> --help` for all options. From a repository checkout, prefix commands with `uv run`.

| Command | Purpose |
|---|---|
| `lkwb bench MODEL... [--data D] [--limit N] [--media DIR] [--json]` | benchmark models |
| `lkwb compare OLD_RUN NEW_RUN [--json]` | what got better and worse |
| `lkwb report RUN... [--threshold T]` | rebuild the HTML report from saved runs |
| `lkwb figures RUN` | write the charts as SVG |
| `lkwb media -o DIR` | build a media false-alarm set (target: 24 hours) |
| `lkwb generate "WAKE WORD" -o DIR [--size SIZE] [--plan PLAN] [--set k=v]` | generate a dataset |
| `lkwb init "WAKE WORD"` | write a config file |
| `lkwb pronounce "TEXT" [-c CONFIG]` | show pronunciation inputs for each provider |
| `lkwb knobs [-c CONFIG]` | list every setting |
| `lkwb voices [--check]` | list or test the TTS voices |
| `lkwb verify DATASET` | check audio against labels |
| `lkwb push DATASET USER/REPO [--public]` | upload to the Hugging Face Hub |
| `lkwb export-livekit DATASET DIR --model-name N` | export for livekit-wakeword training |
| `lkwb export-wav DATASET DIR` | export WAV files and a CSV |
| `lkwb sources` | download the noise, room and speech recordings |

Common `bench` options:

| Option | Default | Purpose |
|---|---|---|
| `--threshold` | 0.5 | Threshold for breakdowns |
| `--score-fpph` | 0.5 | False alarms per hour allowed for the score |
| `--debounce` | 2 seconds | Minimum gap between counted triggers |
| `--filter key=value` | None | Select clips by metadata |
| `--workers` | 1 | Parallel processes; keep at 1 for accurate timing |
| `-o DIR` | `results` | Output directory |

See [Benchmark](benchmark.md#read-the-results) for metric definitions.

## Settings

Set generation options with `--set key=value`, Python keywords such as `augment__p_reverb=0.8`, or YAML (`-c CONFIG`). Run `lkwb knobs` for the full list.

| Setting | Default | Purpose |
|---|---|---|
| `wake_word` | `hey livekit` | the phrase |
| `pronunciations` | | ARPAbet for unknown words |
| `hard_negatives` | | extra near-miss phrases |
| `p_carrier` | 0.35 | share of wake words inside a sentence |
| `tts.plan` / `tts.concurrency` | `build` / unset | Connection preset (5 / 20 / 50) or explicit limit |
| `tts.test_voice_fraction` | 0.3 | voices kept for the test split only |
| `splits.<split>.n_positive`, `n_adversarial`, `n_speech`, `n_noise` | per `--size` | clips per kind |
| `splits.<split>.real_speech_minutes` | per `--size` | recorded conversation |
| `splits.<split>.augment` | train: false | store effects in the audio |
| `augment.clean_weight` / `augment.snr_weights` | 1 / each bucket: 1 | Relative weights for clean audio and each noise level |
| `augment.p_reverb` | 0.5 | room echo |
| `augment.p_interferer` | 0.2 | another voice (0–20 dB quieter) |
| `augment.p_party` | 0.1 | 2–4 voices plus crowd noise |
| `augment.p_second_noise` | 0.2 | two noises at once |
| `augment.p_eq` / `p_bandlimit` / `p_clip` / `p_codec` | 0.4 / 0.15 / 0.05 / 0.25 | microphone, phone line, overload, compression |
| `sources` | all | which recordings to use |

Noise buckets use signal-to-noise ratio (SNR): `quiet` 20–30 dB, `moderate` 10–20, `noisy` 5–10, `very_noisy` 0–5 and `extreme` −5–0. Lower SNR means louder noise relative to speech. `clean` adds no background noise.

TTS shares capacity with production agents. Keep concurrency within your account limit and leave room for production. See [Generate](generate.md#set-tts-capacity).

## Dataset columns

Each row contains a 16 kHz audio clip and metadata:

| Column | Meaning |
|---|---|
| `label`, `category` | 1 for the wake word; `positive`, `adversarial`, `speech`, `real_speech`, `noise`, `media` |
| `text`, `phrase` | what was said; the wake word or near-miss in it |
| `phrase_onset_s`, `phrase_offset_s` | when the wake word starts and ends |
| `provider`, `voice_name`, `gender`, `accent`, `held_out_voice` | the voice |
| `noise_bucket`, `snr_db`, `noise_type` | the noise |
| `interferer`, `sir_db`, `party`, `n_talkers` | other voices |
| `reverb`, `eq`, `bandlimit`, `clipping`, `codec` | room and device effects |
| `augmentations` | every effect used, e.g. `reverb+talker+noise` |
| `phonemes`, `tts_input` | the pronunciation sent to TTS |

The full list is `SCHEMA` in [`dataset.py`](../src/livekit_wakeword_bench/dataset.py). Load a dataset with `WakeWordData(...)`, `datasets.load_dataset(...)` or any parquet reader.

## TTS providers

The bundled providers accept phonemes to guide wake word pronunciation. Voice counts below describe the bundled catalog; use `lkwb voices` to inspect it.

| provider | voices | "LiveKit" is sent as |
|---|---|---|
| Cartesia `sonic-3` | 43 | `<<ˈ\|l\|aɪ\|v\|ˌ\|k\|ɪ\|t>>` |
| Inworld `inworld-tts-2` | 26 | `/ˈlaɪvˌkɪt/` |
| xAI `tts-1` | 26 | `/ˈlaɪvˌkɪt/` |
| Rime `mistv3` | 78 | `{l1Yv} {k1It}` |

Generation uses every model, and every voice, that speaks the phrase's `language`, from each model's official language list in LiveKit Inference (`MODEL_LANGUAGES` in `voices.py`). Rime speakers each speak their own languages (the `languages` field of their voices), so English uses the English speakers and Spanish, German and French use Rime's native speakers. For example, `es`, `de` and `fr` get all four models, `ja`, `pt` and `hi` get Cartesia, Inworld and xAI, and `sw` gets Inworld. `lkwb voices --language es` lists the voices for a language; set `tts.models` to choose the models yourself.

## Audio sources

Training and validation use separate source pools from testing.

| | noise | rooms | recorded speech |
|---|---|---|---|
| train | MUSAN, DNS-Noise | MIT | LibriSpeech |
| test | DEMAND, WHAM!, ESC-50, GTZAN | OpenSLR 28, Treble10 | AMI, Earnings-22, People's Speech |
| media set | AudioSet (TV, YouTube), FMA music | | VoxPopuli speeches |

WHAM!, ESC-50, Treble10 and GTZAN have non-commercial or research-only restrictions. Check source licenses for your intended use. To exclude those four sources:

```yaml
sources:
  - musan_noise
  - dns_noise
  - mit_rir
  - librispeech
  - demand
  - slr28_rir
  - ami
  - earnings22
  - peoples_speech
```
