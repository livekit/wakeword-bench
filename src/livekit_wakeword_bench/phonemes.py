"""Phoneme steering: render key phrases as phonemes in each TTS provider's inline syntax."""

from __future__ import annotations

import re
from typing import Any

from .text import _cmu, _split_unknown

VOWELS = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}

# ARPAbet -> Cartesia en-US v01 (41 symbols). AH is ə stressed or not; ER is always ɝ.
_CARTESIA = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ə",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "IH": "ɪ",
    "IY": "i",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "UH": "ʊ",
    "UW": "u",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}
# Standard English IPA (Inworld / xAI): like Cartesia but ʌ for stressed AH, ɚ unstressed ER.
_IPA = {**_CARTESIA, "AH1": "ʌ", "AH2": "ʌ", "ER0": "ɚ"}
# Rime phonetic alphabet (1:1 with ARPAbet; stress digit goes before each vowel).
_RIME = {
    "AA": "a",
    "AE": "@",
    "AH": "A",
    "AO": "a",
    "AW": "W",
    "AY": "Y",
    "EH": "E",
    "ER": "R",
    "EY": "e",
    "IH": "I",
    "IY": "i",
    "OW": "o",
    "OY": "O",
    "UH": "U",
    "UW": "u",
    "B": "b",
    "CH": "C",
    "D": "d",
    "DH": "D",
    "F": "f",
    "G": "g",
    "HH": "h",
    "JH": "J",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "G",
    "P": "p",
    "R": "r",
    "S": "s",
    "SH": "S",
    "T": "t",
    "TH": "T",
    "V": "v",
    "W": "w",
    "Y": "y",
    "Z": "z",
    "ZH": "Z",
}
_RIME_SCHWA = "x"  # AH0

# Legal English syllable onsets (for placing stress marks before the onset).
_ONSETS = {
    *[(c,) for c in _CARTESIA if c not in VOWELS and c != "NG"],
    *[(a, b) for a in ("P", "B", "K", "G", "F") for b in ("L", "R")],
    *[(a, "R") for a in ("T", "D", "TH", "SH")],
    *[(a, "W") for a in ("T", "D", "K", "G", "S", "TH")],
    *[(a, "Y") for a in ("P", "B", "K", "G", "F", "V", "M", "N", "HH", "L")],
    *[("S", b) for b in ("P", "T", "K", "M", "N", "L", "W", "F")],
    ("S", "P", "L"),
    ("S", "P", "R"),
    ("S", "T", "R"),
    ("S", "K", "R"),
    ("S", "K", "W"),
    ("S", "K", "L"),
    ("S", "P", "Y"),
    ("S", "K", "Y"),
}

PHONEME_PROVIDERS = ("cartesia", "inworld", "xai", "rime")


def supports_phonemes(model: str) -> bool:
    provider = model.split("/")[0]
    if provider == "rime":
        return "mist" in model  # Coda does not support phonemizeBetweenBrackets
    if provider == "cartesia":
        return "sonic-3" in model
    return provider in ("inworld", "xai")


def provider_options(model: str) -> dict[str, Any]:
    """Extra TTS options needed for phoneme input."""
    return {"phonemize_between_brackets": True} if model.startswith("rime/") else {}


def validate_arpabet(pron: str) -> str | None:
    """Return an error message if `pron` isn't valid ARPAbet (with optional "|" parts)."""
    syms = [t for t in pron.replace("|", " ").split()]
    if not syms:
        return "empty pronunciation"
    for t in syms:
        m = re.fullmatch(r"([A-Z]+)([012]?)", t)
        if not m or m.group(1) not in _CARTESIA:
            return f"unknown ARPAbet symbol {t!r}"
        if m.group(1) in VOWELS and not m.group(2):
            return f"vowel {t!r} needs a stress digit (0/1/2)"
    return None


def word_parts(word: str, overrides: dict[str, str] | None = None) -> list[list[str]] | None:
    """ARPAbet for one word as a list of parts (a compound like "livekit" -> live | kit)."""
    w = re.sub(r"[^a-z']", "", word.lower())
    if overrides and w in overrides:
        return [p.split() for p in overrides[w].split("|") if p.strip()]
    cmu = _cmu()
    if w in cmu:
        return [cmu[w][0].split()]
    parts = _split_unknown(w, cmu)
    if len(parts) > 1 and all(p in cmu for p in parts):
        out = [cmu[parts[0]][0].split()]
        for p in parts[1:]:
            out.append([ph[:-1] + "2" if ph.endswith("1") else ph for ph in cmu[p][0].split()])
        return out
    return None


def word_arpabet(word: str) -> list[list[str]]:
    """Flat CMUdict-style pronunciations (compounds joined)."""
    parts = word_parts(word)
    return [[ph for part in parts for ph in part]] if parts else []


def _split(ph: str) -> tuple[str, str]:
    m = re.fullmatch(r"([A-Z]+)([012]?)", ph)
    if not m:
        raise ValueError(f"bad ARPAbet symbol {ph!r}")
    return m.group(1), m.group(2)


def _stress_positions(arpa: list[str]) -> dict[int, str]:
    """Index at which to insert a stress mark (before the onset of each stressed syllable)."""
    marks: dict[int, str] = {}
    prev_vowel = -1
    for i, ph in enumerate(arpa):
        base, st = _split(ph)
        if base not in VOWELS:
            continue
        if st in ("1", "2"):
            cluster = [_split(p)[0] for p in arpa[prev_vowel + 1 : i]]
            onset = 0
            for k in range(min(3, len(cluster)), 0, -1):
                if tuple(cluster[-k:]) in _ONSETS:
                    onset = k
                    break
            if prev_vowel < 0:  # word-initial: whole cluster is the onset
                onset = len(cluster)
            marks[i - onset] = "ˈ" if st == "1" else "ˌ"
        prev_vowel = i
    return marks


def _ensure_primary(arpa: list[str]) -> list[str]:
    """Every word needs one primary stress (Cartesia rule); promote the strongest vowel."""
    if any(p.endswith("1") for p in arpa):
        return arpa
    out = list(arpa)
    for i, p in enumerate(out):
        if p.endswith("2"):
            out[i] = p[:-1] + "1"
            return out
    for i, p in enumerate(out):
        if _split(p)[0] in VOWELS:
            out[i] = _split(p)[0] + "1"
            return out
    return out


def to_cartesia(arpa: list[str]) -> str:
    arpa = _ensure_primary(arpa)
    marks = _stress_positions(arpa)
    syms: list[str] = []
    for i, ph in enumerate(arpa):
        if i in marks:
            syms.append(marks[i])
        syms.append(_CARTESIA[_split(ph)[0]])
    return "<<" + "|".join(syms) + ">>"


def to_ipa(arpa: list[str]) -> str:
    arpa = _ensure_primary(arpa)
    marks = _stress_positions(arpa)
    out = []
    for i, ph in enumerate(arpa):
        if i in marks:
            out.append(marks[i])
        base, st = _split(ph)
        out.append(_IPA.get(base + st, _IPA[base]))
    return "".join(out)


def to_rime(arpa: list[str]) -> str:
    out = []
    for ph in _ensure_primary(arpa):
        base, st = _split(ph)
        if base in VOWELS:
            sym = _RIME_SCHWA if (base == "AH" and st in ("0", "")) else _RIME[base]
            out.append((st or "0") + sym)
        else:
            out.append(_RIME[base])
    return "{" + "".join(out) + "}"


def render_word(parts: list[list[str]], provider: str) -> str:
    flat = [ph for part in parts for ph in part]
    if provider == "cartesia":
        return to_cartesia(flat)
    if provider in ("inworld", "xai"):
        return f"/{to_ipa(flat)}/"
    if provider == "rime":
        # one bracket per morpheme: {l1Yvk2It} comes out as "life get"
        return " ".join(to_rime(part) for part in parts)
    raise ValueError(f"{provider} does not take phonemes")


def phrase_parts(
    text: str, overrides: dict[str, str] | None = None
) -> list[list[list[str]]] | None:
    """Per-word parts for every word of a phrase; None if any word is unknown."""
    out = []
    for w in re.findall(r"[A-Za-z']+", text):
        parts = word_parts(w, overrides)
        if parts is None:
            return None
        out.append(parts)
    return out


def arpabet_string(text: str, overrides: dict[str, str] | None = None) -> str:
    """Human-readable ARPAbet of a phrase: words separated by " . ", parts by " | "."""
    pp = phrase_parts(text, overrides) or []
    return " . ".join(" | ".join(" ".join(part) for part in parts) for parts in pp)


def render_phrase(text: str, model: str, overrides: dict[str, str] | None = None) -> str | None:
    """Rewrite each word of `text` as provider phonemes, keeping the punctuation between words (it shapes prosody)."""
    provider = model.split("/")[0]
    tokens = re.split(r"([A-Za-z']+)", text)
    pp = phrase_parts(text, {k.lower(): v for k, v in (overrides or {}).items()})
    if pp is None:
        return None
    it = iter(pp)
    out = []
    for tok in tokens:
        if re.fullmatch(r"[A-Za-z']+", tok):
            out.append(render_word(next(it), provider))
        else:
            out.append(tok)
    return "".join(out)
