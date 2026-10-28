# Reference

## Commands

Run `lkwb <command> --help` for every option.

| command | does |
|---|---|
| `lkwb bench MODEL... [--data D] [--limit N] [--media DIR] [--json]` | benchmark models |
| `lkwb compare OLD_RUN NEW_RUN [--json]` | what got better and worse |
| `lkwb report RUN... [--threshold T]` | rebuild the HTML report from saved runs |
| `lkwb figures RUN` | write the charts as SVG |
| `lkwb media -o DIR` | build the 24 h TV / music / speech false-alarm set |
| `lkwb generate "WAKE WORD" -o DIR [--size] [--plan] [--set k=v]` | generate a dataset |
| `lkwb init "WAKE WORD"` | write a config file |
| `lkwb pronounce "TEXT" [-c CONFIG]` | show what each TTS will say |
| `lkwb knobs [-c CONFIG]` | list every setting |
| `lkwb voices [--check]` | list or test the TTS voices |
| `lkwb verify DATASET` | check audio against labels |
| `lkwb push DATASET USER/REPO [--public]` | upload to the Hugging Face Hub |
| `lkwb export-livekit DATASET DIR --model-name N` | export for livekit-wakeword training |
| `lkwb export-wav DATASET DIR` | export WAV files and a CSV |
| `lkwb sources` | download the noise, room and speech recordings |

`lkwb bench` options: `--threshold` (0.5) sets the threshold for the breakdown, `--score-fpph` (0.5) the false-alarm target, `--debounce` (2 s) the gap between counted triggers, `--filter key=value` runs on a subset, and `--workers N` runs in parallel (timing is only accurate with 1).

## Settings

Change any setting with `--set key=value`, `generate(..., key__sub=value)` in Python, or a YAML file (`-c`). `lkwb knobs` lists them all.

| setting | default | does |
|---|---|---|
| `wake_word` | `hey livekit` | the phrase |
| `pronunciations` | | ARPAbet for unknown words |
| `hard_negatives` | | extra near-miss phrases |
| `p_carrier` | 0.35 | share of wake words inside a sentence |
| `tts.plan` / `tts.concurrency` | build | TTS connections (5 / 20 / 50, or your number) |
| `tts.test_voice_fraction` | 0.3 | voices kept for the test split only |
| `splits.<split>.n_positive`, `n_adversarial`, `n_speech`, `n_noise` | per `--size` | clips per kind |
| `splits.<split>.real_speech_minutes` | per `--size` | recorded conversation |
| `splits.<split>.augment` | train: false | store effects in the audio |
| `augment.clean_weight` / `snr_weights` | 1 / all 1 | how often each noise level is used |
| `augment.p_reverb` | 0.5 | room echo |
| `augment.p_interferer` | 0.2 | another voice (0–20 dB quieter) |
| `augment.p_party` | 0.1 | 2–4 voices plus crowd noise |
| `augment.p_second_noise` | 0.2 | two noises at once |
| `augment.p_eq` / `p_bandlimit` / `p_clip` / `p_codec` | 0.4 / 0.15 / 0.05 / 0.25 | microphone, phone line, overload, compression |
| `sources` | all | which recordings to use |

Noise levels: `quiet` 20–30 dB, `moderate` 10–20, `noisy` 5–10, `very_noisy` 0–5, `extreme` −5–0 (noise louder than speech), plus `clean`.

## Dataset columns

Each row is one 16 kHz clip with its recipe. The most useful columns:

| column | meaning |
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

Only providers that take phonemes are used, so every voice says the wake word the same way.

| provider | voices | "LiveKit" is sent as |
|---|---|---|
| Cartesia `sonic-3` | 43 | `<<ˈ\|l\|aɪ\|v\|ˌ\|k\|ɪ\|t>>` |
| Inworld `inworld-tts-2` | 26 | `/ˈlaɪvˌkɪt/` |
| xAI `tts-1` | 26 | `/ˈlaɪvˌkɪt/` |
| Rime `mistv3` | 62 | `{l1Yv} {k1It}` |

## Audio sources

Training and testing never share recordings.

| | noise | rooms | recorded speech |
|---|---|---|---|
| train | MUSAN, DNS-Noise | MIT | LibriSpeech |
| test | DEMAND, WHAM!, ESC-50, GTZAN | OpenSLR 28, Treble10 | AMI, Earnings-22, People's Speech |
| media set | AudioSet (TV, YouTube), FMA music | | VoxPopuli speeches |

WHAM!, ESC-50, Treble10 and GTZAN are for non-commercial use. To leave them out: `sources: [musan_noise, dns_noise, mit_rir, librispeech, demand, slr28_rir, ami, earnings22, peoples_speech]`.
