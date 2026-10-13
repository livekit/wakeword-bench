# Phoneme steering

Text-to-speech engines guess how to say words they don't know. For a brand name the guesses vary: one voice says "LiveKit" as /laɪv kɪt/ ("live" as in "live music"), another as /lɪv kɪt/ ("live" as in "I live here"), another as "life kid". A wake word benchmark built on those guesses would mix right and wrong pronunciations under the same label.

So the generator sends the wake word, and every near-miss phrase, to TTS as **phonemes**. Every voice says exactly the sounds you specify. Carrier sentences and everyday speech stay plain text.

## Supported providers

Only models that accept inline phonemes through LiveKit Inference are used:

| provider | model | what it receives for "LiveKit" | voices |
|---|---|---|---|
| Cartesia | `sonic-3` | `<<ˈ\|l\|aɪ\|v\|ˌ\|k\|ɪ\|t>>` | 43 |
| Inworld | `inworld-tts-2` | `/ˈlaɪvˌkɪt/` | 26 |
| xAI | `tts-1` | `/ˈlaɪvˌkɪt/` | 26 |
| Rime | `mistv3` | `{l1Yv} {k1It}` (with `phonemize_between_brackets`) | 62 |

Each provider was checked by synthesizing words from phonemes alone (no spelling in the text) and transcribing the result. Deepgram Aura, Fish Audio and Rime Coda read the markup literally and are not used. `lkwb voices --check` repeats a phoneme request for every voice in the catalog.

## Where pronunciations come from

1. **The CMU pronouncing dictionary** covers about 130,000 English words. Words found there need nothing.
2. **Compound words** that are not in the dictionary are split into known parts: "livekit" → "live" + "kit". Later parts get secondary stress.
3. **Your `pronunciations` config** overrides both.

```yaml
wake_word: hey livekit
pronunciations:
  livekit: "L AY1 V | K IH2 T"
```

Pronunciations use [ARPAbet](https://en.wikipedia.org/wiki/ARPABET), the same notation as the CMU dictionary. Every vowel carries a stress digit: `1` primary, `2` secondary, `0` unstressed. Use `|` between the parts of a compound word: Rime needs each part in its own bracket (`{l1Yvk2It}` comes out as "life get", `{l1Yv} {k1It}` as "live kit"), and the other providers join the parts into one word.

## Checking a pronunciation

```bash
uv run lkwb pronounce "Hey, LiveKit!" -c configs/hey_livekit.yaml
```

```
ARPAbet   HH EY1 . L AY1 V | K IH2 T
cartesia  <<ˈ|h|eɪ>>, <<ˈ|l|aɪ|v|ˌ|k|ɪ|t>>!
inworld   /ˈheɪ/, /ˈlaɪvˌkɪt/!
xai       /ˈheɪ/, /ˈlaɪvˌkɪt/!
rime      {h1e}, {l1Yv} {k1It}!
```

For a new wake word, `lkwb init` guesses pronunciations for unknown words and marks the ones it can't guess with `TODO`:

```bash
uv run lkwb init "ok zorblax"
# zorblax: "TODO"   # not in CMUdict: write ARPAbet, e.g. "K AE1 T"
```

Generation refuses to start while a word has no pronunciation or an entry has an invalid symbol or a vowel without a stress digit.

## ARPAbet quick reference

| vowels | as in | | consonants | as in |
|---|---|---|---|---|
| AA | f**a**ther | | CH | **ch**eese |
| AE | c**a**t | | DH | **th**is |
| AH | c**u**t, sof**a** | | HH | **h**at |
| AO | th**ou**ght | | JH | **j**am |
| AW | n**ow** | | NG | si**ng** |
| AY | m**y** | | SH | **sh**ip |
| EH | b**e**d | | TH | **th**in |
| ER | b**ir**d | | Y | **y**es |
| EY | d**ay** | | ZH | vi**si**on |
| IH | b**i**t | | R, L, W | **r**ed, **l**eg, **w**e |
| IY | s**ee** | | B D F G K M N P S T V Z | as written |
| OW | g**o** | | | |
| OY | b**oy** | | | |
| UH | b**oo**k | | | |
| UW | t**oo** | | | |

Example: "zorblax" stressed on the first syllable is `Z AO1 R B L AE2 K S`.

## How the conversion works

`src/livekit_wakeword_bench/phonemes.py` converts ARPAbet to each provider's syntax:

- **Cartesia** uses its 41-symbol American English set, one symbol per `|`, with the stress mark placed before the consonants that start the stressed syllable.
- **Inworld and xAI** use standard English IPA wrapped in slashes, one word per slash pair.
- **Rime** uses its own alphabet, which maps one-to-one to ARPAbet, with the stress digit before each vowel.

Set `tts.phonemes: false` to send plain text instead; pronunciation is then left to each voice.
