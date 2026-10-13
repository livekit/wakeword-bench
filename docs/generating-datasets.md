# Generating datasets

You've picked a wake word and recorded yourself saying it a few times. That is enough to try a detector, but it leaves a lot untested: other speakers, background conversation, different rooms and phrases that sound almost right.

The generator builds those examples using [LiveKit Inference](https://docs.livekit.io/agents/models/tts/inference/) speech and recorded noise, room responses and conversation. Start with a small dataset, check the pronunciation, then expand it or generate training examples for a weakness you've measured. Each clip keeps the recipe used to make it.

## 1. Connect a LiveKit Cloud project

Generation spends LiveKit Inference credits, so it needs a LiveKit Cloud project. Install the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/):

```bash
brew install livekit-cli                      # macOS
curl -sSL https://get.livekit.io/cli | bash   # Linux
winget install LiveKit.LiveKitCLI             # Windows
```

Then link your account and write the credentials into the repo:

```bash
lk cloud auth      # opens a browser to link your LiveKit Cloud account (once)
lk app env -w      # writes LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET to .env.local
```

`lk app env` fills `.env.local` from the repo's `.env.example`. With several projects, `lk project list` shows them and `lk project set-default <name>` picks one before you run `lk app env -w`. `lkwb` loads `.env.local` automatically, and `.env.local` is git-ignored.

## 2. Pick your plan

LiveKit Inference limits how many TTS connections can run at once, per [plan](https://livekit.com/pricing). Choose the plan that matches your account. The generator uses the configured limit; it doesn't check your account's allowance.

| plan | flag | concurrent TTS | rough speed |
|---|---|---|---|
| Build (free) | `--plan build` (default) | 5 | ~4 utterances/s |
| Ship | `--plan ship` | 20 | ~15 utterances/s |
| Scale | `--plan scale` | 50 | ~30 utterances/s |
| Enterprise | `--set tts.concurrency=N` | your limit | |

Set `tts.plan` in a config file, or override the connection count with `LKWB_TTS_CONCURRENCY`. Never set concurrency above your account limit. All splits share one connection budget, so they synthesize in parallel without going over the limit.

> [!WARNING]
> Production voice agents share this TTS connection capacity. By default, generation can use the full configured plan limit, leaving no connections for agents to generate speech. Set `--set tts.concurrency=N` below your limit and leave enough capacity for production traffic. For example, `--set tts.concurrency=2` uses at most two connections; choose a value that fits your workload.

## 3. Generate

```bash
uv run lkwb generate "hey jarvis" --size small --plan build -o out/hey_jarvis
```

```python
from livekit_wakeword_bench import generate

generate("hey jarvis", "out/hey_jarvis", size="small", plan="build")
```

`--size` sets how many clips each split gets:

| size | train | validation | test | use it for |
|---|---|---|---|---|
| `tiny` | 90 | — | ~250 | checking that everything works |
| `small` | 2,300 | ~280 | ~1,300 | a first dataset |
| `default` | 23,000 | ~2,500 | ~7,900 | the full benchmark |

Any setting can be overridden with `--set key=value` (see [Configuration](configuration.md)):

```bash
uv run lkwb generate "hey jarvis" -o out/hey_jarvis --size small \
    --set augment.p_interferer=0.5 --set splits.test.n_positive=2000
```

## What gets generated

The dataset supports five kinds of clips. The split settings determine how many of each are generated:

| category | label | what it is |
|---|---|---|
| `positive` | 1 | the wake word, alone ("Hey Jarvis") or inside a sentence ("Hey Jarvis, set a timer", "Okay, hey Jarvis") |
| `adversarial` | 0 | near-miss phrases that sound almost like the wake word |
| `speech` | 0 | everyday sentences that do not contain the wake word |
| `real_speech` | 0 | segments of real recorded conversation (meetings, calls, web audio) |
| `noise` | 0 | background noise with no speech |

**Near-miss phrases** come from the CMU pronouncing dictionary. For each word of the wake word, the generator finds common English words whose pronunciation differs by one or two sounds ("jarvis" → "marius", "vargas") and swaps them in. It also drops single words ("jarvis" alone). Closer near-misses are sampled more often. Phrases that *contain* the wake word acoustically ("hey jarvises") are removed, because a streaming model correctly fires before the extra sound. Add your own with `hard_negatives`.

**Phrasing** varies punctuation ("Hey Jarvis", "Hey, Jarvis!", "Hey Jarvis?") and wraps about 35% of wake words and near-misses in a carrier sentence (`p_carrier`). Carrier segments are synthesized separately and joined with a short pause, which gives exact wake word timestamps.

**Voices** come from 157 voices across Cartesia, Inworld, xAI and Rime. Speaking rate varies between 0.85× and 1.2×. About 30% of the voices of each provider are reserved for the test split (`tts.test_voice_fraction`), so the test set measures how a model handles speakers it has never heard.

**Timing.** Each wake word clip has at least 2 seconds of audio before the wake word ends, plus random extra lead time, so the wake word position is not a constant a model could learn.

**Train vs test.** The test split uses different voices, noise recordings, rooms and real speech from training and validation (see [Audio sources](audio-sources.md)). The train split is stored clean by default; training code adds fresh augmentation on the fly with `Augment()` (see [Loading data](loading-data.md)). Set `--set splits.train.augment=true` to store augmented training audio instead.

## Pronunciation

Wake words are sent to TTS as phonemes to specify the intended pronunciation. Dictionary words are handled automatically; check the generated config for brand names and invented words:

```bash
uv run lkwb init "ok zorblax"                # writes ok_zorblax.yaml with a TODO for "zorblax"
uv run lkwb pronounce "ok zorblax" -c ok_zorblax.yaml
```

See [Phoneme steering](phonemes.md).

## Config files

`lkwb init` writes a starter config you can edit and keep in version control:

```bash
uv run lkwb init "hey jarvis" --plan ship --size small    # writes hey_jarvis.yaml
uv run lkwb generate -c hey_jarvis.yaml -o out/hey_jarvis
```

The config holds the wake word, pronunciations, phrasing, hard negatives, plan and split sizes. `--set` overrides still apply on top of it.

## Caching and reproducibility

- **TTS is cached** in `.cache/lkwb/tts`, keyed by provider, voice, text and speed. Requests already in the cache incur no new TTS charge. Changes that require new utterances still spend inference credits.
- **Sampling and augmentation use a seed.** Keep the config, cached TTS and audio sources to reproduce a run; a seed alone cannot guarantee identical output from an external TTS service.
- **Audio sources** (noise, rooms, real speech) are downloaded from Hugging Face on first use into `.cache/lkwb/sources`. Prefetch them with `lkwb sources`.
- **Splits** can be generated separately with `--split train` / `--split test`. Existing shards for a split are kept unless you pass `--overwrite`.

## Checking a dataset

```bash
uv run lkwb verify out/hey_jarvis               # test split
uv run lkwb verify out/hey_jarvis --split validation
```

`verify` measures the audio and checks it against its labels:

| check | passes when |
|---|---|
| integrity | every clip decodes, has no NaN/inf, matches its duration, and its label matches its category |
| leakage | no TTS voice, noise file, room recording or noise source is shared between train/validation and test |
| level | the measured peak matches `level_dbfs` (clips with codecs or clipping are skipped) |
| SNR | SNR re-measured from the audio agrees with `snr_db` (median error is reported) |
| timing | clean, dry TTS clips are silent outside `onset_s`…`offset_s` and have sound right after `onset_s` |
| band-limit | band-limited clips have almost no energy above 4.5 kHz |

## Publishing

```bash
uv run lkwb push out/hey_jarvis you/wakeword-bench-hey-jarvis            # private
uv run lkwb push out/hey_jarvis you/wakeword-bench-hey-jarvis --public
```

The dataset card (`README.md` in the dataset directory) lists the splits, the composition of each split, the audio sources with their licenses, and the generation config. Anyone can then benchmark against the repo id without regenerating:

```bash
uv run lkwb bench you/wakeword-bench-hey-jarvis -d my_pkg.models:make_model
```

## Voices

```bash
uv run lkwb voices                  # list the catalog by provider and accent
uv run lkwb voices --check          # synthesize a short phrase with every voice
```

The catalog lives in `src/livekit_wakeword_bench/voices.json`. `--check` sends a phoneme request to each voice, so it also confirms the provider accepts phoneme input. It uses `LKWB_TTS_CONCURRENCY` (default 5) connections.
