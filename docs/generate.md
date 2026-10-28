# Generate

Make a dataset for any wake word: TTS speech from 157 voices, mixed with real noise, rooms and other talkers.

## 1. Connect LiveKit

Speech comes from [LiveKit Inference](https://docs.livekit.io/agents/models/tts/inference/), so you need a LiveKit Cloud project. Install the [LiveKit CLI](https://docs.livekit.io/intro/basics/cli/), then:

```bash
lk cloud auth      # link your account (once)
lk app env -w      # write your credentials to .env.local
```

## 2. Pick your plan

LiveKit limits how many TTS connections run at once. Use your plan:

| plan | flag | connections |
|---|---|---|
| Build (free) | `--plan build` (default) | 5 |
| Ship | `--plan ship` | 20 |
| Scale | `--plan scale` | 50 |
| Enterprise | `--set tts.concurrency=N` | yours |

> [!WARNING]
> Your production voice agents use the same connections. Leave them room: `--set tts.concurrency=N` with N below your limit.

## 3. Generate

```bash
uv run lkwb generate "hey jarvis" --size small --plan ship -o out/hey_jarvis
```

```python
from livekit_wakeword_bench import generate
generate("hey jarvis", "out/hey_jarvis", size="small", plan="ship")
```

| size | train | validation | test |
|---|---|---|---|
| `tiny` | 90 | — | ~250 |
| `small` | 2,300 | ~280 | ~1,300 |
| `default` | 23,000 | ~2,500 | ~7,900 |

Ready-made configs for common wake words are in `configs/` (alexa, hey jarvis, hey mycroft, okay nabu, hey rhasspy, hey livekit): `lkwb generate -c configs/alexa.yaml -o out/alexa`.

Speech is cached, so generating again with different noise settings is free.

## Pronunciation

Every voice says the wake word exactly the same way, because the wake word is sent to TTS as phonemes. Common English words just work. Names and made-up words need a pronunciation:

```bash
uv run lkwb init "ok zorblax"                        # writes ok_zorblax.yaml, marks unknown words
uv run lkwb pronounce "ok zorblax" -c ok_zorblax.yaml  # shows what each TTS will say
```

```yaml
pronunciations:
  zorblax: "Z AO1 R B L AE2 K S"   # ARPAbet; 1 = stressed vowel, | between word parts
```

## What you get

| clips | label | what |
|---|---|---|
| wake word | 1 | alone, or inside a sentence ("Hey Jarvis, set a timer") |
| near-misses | 0 | phrases that sound almost right ("hey marius") |
| speech | 0 | everyday sentences |
| real speech | 0 | recorded meetings and calls |
| noise | 0 | background noise only |

The test and validation splits get realistic effects: room echo, another voice, a party (several voices plus crowd noise), background noise from quiet to louder than speech, microphone and phone-line effects, and compression. The train split is stored clean, and training adds fresh effects ([Train](train.md)).

The test split uses different voices, noise recordings, rooms and real speakers than training, so it measures how well a model handles audio it has never heard.

## Check and share

```bash
uv run lkwb verify out/hey_jarvis                       # audio matches its labels, no train/test overlap
uv run lkwb push out/hey_jarvis you/wakeword-bench-hey-jarvis --public
```

Every setting can be changed: see [Reference](reference.md#settings).
