"""`lkwb init`: write a starter config for a new wake word."""

from __future__ import annotations

import re

from .config import PLAN_CONCURRENCY, SIZES
from .phonemes import PHONEME_PROVIDERS, render_phrase, word_parts
from .text import _cmu


def guess_pronunciations(wake_word: str) -> tuple[dict[str, str], list[str]]:
    """ARPAbet overrides for words missing from CMUdict."""
    cmu = _cmu()
    guesses: dict[str, str] = {}
    unknown: list[str] = []
    for w in re.findall(r"[A-Za-z']+", wake_word):
        lw = w.lower()
        if lw in cmu:
            continue
        parts = word_parts(lw)
        if parts:
            guesses[lw] = " | ".join(" ".join(p) for p in parts)
        else:
            unknown.append(lw)
    return guesses, unknown


def positive_texts(wake_word: str) -> list[str]:
    words = wake_word.split()
    title = " ".join(w[:1].upper() + w[1:] for w in words)
    forms = [title, f"{title}!", f"{title}?", f"{title}."]
    if len(words) > 1:
        head, rest = (
            words[0][:1].upper() + words[0][1:],
            " ".join(w[:1].upper() + w[1:] for w in words[1:]),
        )
        forms.append(f"{head}, {rest}!")
    return forms


def config_yaml(wake_word: str, plan: str = "build", size: str = "default") -> str:
    guesses, unknown = guess_pronunciations(wake_word)
    name = re.sub(r"[^a-z0-9]+", "_", wake_word.lower()).strip("_")
    tp, ta, ts, _, _ = SIZES[size]["train"]
    vp, va, vs, vn, vr = SIZES[size]["validation"]
    sp, sa, ss, sn, sr = SIZES[size]["test"]
    pron_lines = [
        f'  {w}: "{p}"   # auto-guessed from sub-words, check it' for w, p in guesses.items()
    ]
    pron_lines += [
        f'  {w}: "TODO"   # not in CMUdict: write ARPAbet, e.g. "K AE1 T"' for w in unknown
    ]
    pron = "pronunciations:\n" + "\n".join(pron_lines) if pron_lines else "pronunciations: {}"
    conc = PLAN_CONCURRENCY.get(plan)
    plan_note = f"{conc} concurrent TTS connections" if conc else "set tts.concurrency"
    texts = ", ".join(f'"{t}"' for t in positive_texts(wake_word))
    return f"""# Wake word dataset config. Generate with:
#   lkwb generate -c {name}.yaml -o out/{name}
name: {name}
wake_word: {wake_word}
# Exact pronunciation (ARPAbet, stress digits 0/1/2; "|" separates morphemes).
# Words in CMUdict need no entry. Preview: lkwb pronounce "{wake_word}" -c {name}.yaml
{pron}
# Only punctuation varies: words are sent to TTS as phonemes.
positive_texts: [{texts}]
# Hand-written confusables, on top of the automatic phonetic neighbours.
hard_negatives: []

tts:
  # Your LiveKit Cloud plan sets the TTS concurrency limit: build 5, ship 20, scale 50.
  plan: {plan}   # -> {plan_note}
  test_voice_fraction: 0.3

splits:
  train:          # stored clean; augment on the fly with livekit_wakeword_bench.loader.Augment
    n_positive: {tp}
    n_adversarial: {ta}
    n_speech: {ts}
    augment: false
  validation:
    n_positive: {vp}
    n_adversarial: {va}
    n_speech: {vs}
    n_noise: {vn}
    real_speech_minutes: {vr}
  test:
    n_positive: {sp}
    n_adversarial: {sa}
    n_speech: {ss}
    n_noise: {sn}
    real_speech_minutes: {sr}
"""


def preview(text: str, overrides: dict[str, str]) -> dict[str, str | None]:
    models = {
        "cartesia": "cartesia/sonic-3",
        "inworld": "inworld/inworld-tts-2",
        "xai": "xai/tts-1",
        "rime": "rime/mistv3",
    }
    return {p: render_phrase(text, models[p], overrides) for p in PHONEME_PROVIDERS}
