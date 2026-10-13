"""Text generation: wake word surface forms, carrier sentences, adversarial phrases, filler speech."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from functools import lru_cache

# carrier segments are synthesized separately, giving exact key-phrase timestamps
CARRIER_AFTER = [
    "what's the weather like today?",
    "turn on the kitchen lights.",
    "set a timer for ten minutes.",
    "can you hear me?",
    "play some music.",
    "call mom.",
    "are you there?",
    "what time is it?",
    "start the meeting.",
    "remind me to buy milk.",
    "how long is my commute?",
    "read me the news.",
]
CARRIER_BEFORE = [
    "Okay,",
    "Um,",
    "So I said",
    "Alright then.",
    "Wait.",
    "Let me try again.",
    "Excuse me,",
    "Hmm,",
]

FILLER_SPEECH = [
    "I think we should leave a little earlier tomorrow.",
    "Hey, did you see the game last night?",
    "The kids are playing in the backyard.",
    "Can you pass me the first aid kit?",
    "We live just a few blocks from the station.",
    "She decided to quit her job and travel.",
    "Let's grab lunch after the meeting.",
    "The delivery should arrive between two and four.",
    "I can't believe how fast this year went by.",
    "Please remember to lock the door when you leave.",
    "He's been working on that project for months.",
    "Turn left at the next light, then keep going straight.",
    "Do you want coffee or tea?",
    "The live stream starts at eight tonight.",
    "My little kid loves dinosaurs.",
    "Hey there, how's it going?",
    "It's supposed to rain all weekend.",
    "That restaurant has the best noodles in town.",
    "I need to charge my phone before we go.",
    "Could you send me the slides from yesterday?",
    "The concert was absolutely incredible.",
    "We should probably update the documentation.",
    "Hey, that's my seat.",
    "Give me a lift to the airport, would you?",
    "The cat knocked a glass off the table again.",
    "I'll call you back in five minutes.",
    "What did you have for dinner?",
    "Living in the city is expensive but fun.",
    "The toolkit is in the garage.",
    "Hey Siri, what's the time?",
    "Alexa, play the news.",
    "Okay Google, set an alarm.",
    "Let me check my calendar and get back to you.",
    "The train was delayed by twenty minutes.",
    "I'm learning to play the guitar.",
    "Our team shipped the new release on Friday.",
    "Can you believe it's already October?",
    "Hey Kevin, long time no see.",
    "Hey Lee, can you hand me that?",
    "Live music every Friday night at the bar.",
    "Let's kick off with a quick round of introductions.",
    "Please hold while I transfer your call.",
    "Make sure the oven is off before you leave.",
    "The quick brown fox jumps over the lazy dog.",
    "I'd like a large pizza with mushrooms.",
    "This kit comes with everything you need.",
    "Hey look, it's snowing!",
    "Leave it on the counter, I'll get it later.",
    "Thanks for joining the call, everyone.",
    "We're going live in three, two, one.",
]


_BLOCKLIST = {
    "shit",
    "tit",
    "tits",
    "fuck",
    "cunt",
    "dick",
    "cock",
    "piss",
    "fag",
    "slut",
    "whore",
    "gay",
}


@lru_cache(maxsize=1)
def _cmu() -> dict[str, list[str]]:
    import pronouncing

    pronouncing.init_cmu()
    d: dict[str, list[str]] = {}
    for word, phones in pronouncing.pronunciations:
        d.setdefault(word, []).append(phones)
    return d


def _strip_stress(phones: str) -> list[str]:
    return [re.sub(r"\d", "", p) for p in phones.split()]


def _split_unknown(word: str, cmu: dict[str, list[str]]) -> list[str]:
    """Split an out-of-vocabulary word into known sub-words ("livekit" -> "live", "kit")."""
    if word in cmu or len(word) < 4:
        return [word]
    for i in range(len(word) - 2, 1, -1):
        head, tail = word[:i], word[i:]
        if head in cmu:
            rest = _split_unknown(tail, cmu)
            if all(r in cmu for r in rest):
                return [head, *rest]
    return [word]


def _edit_distance(a: list[str], b: list[str]) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[-1]


def adversarial_phrases(
    wake_word: str, max_per_word: int = 40, min_zipf: float = 2.5
) -> list[tuple[str, int]]:
    """Phonetically confusable phrases with their phoneme edit distance to the wake word."""
    from wordfreq import zipf_frequency

    cmu = _cmu()
    words: list[str] = []
    for w in re.findall(r"[a-z']+", wake_word.lower()):
        words.extend(_split_unknown(w, cmu))
    target = [_strip_stress(cmu[w][0]) if w in cmu else [] for w in words]

    by_len: dict[int, list[tuple[str, list[str]]]] = {}
    for w, prons in cmu.items():
        if not re.fullmatch(r"[a-z]+", w):
            continue
        ph = _strip_stress(prons[0])
        by_len.setdefault(len(ph), []).append((w, ph))

    out: dict[str, int] = {}
    for idx, (word, ph) in enumerate(zip(words, target)):
        if not ph:
            continue
        cands: list[tuple[int, str]] = []
        for n in range(max(1, len(ph) - 1), len(ph) + 2):
            for w, cph in by_len.get(n, []):
                if w == word or cph == ph or w in _BLOCKLIST:
                    continue
                d = _edit_distance(ph, cph)
                if 1 <= d <= (1 if len(ph) <= 2 else 2):
                    freq = zipf_frequency(w, "en")
                    if freq >= min_zipf:
                        cands.append((d, -freq, w))
        cands.sort()
        for d, _, w in cands[:max_per_word]:
            phrase = " ".join(words[:idx] + [w] + words[idx + 1 :])
            out[phrase] = min(out.get(phrase, 99), d)

    if len(words) > 1:
        for i in range(len(words)):
            partial = " ".join(words[:i] + words[i + 1 :])
            out.setdefault(partial, len(target[i]))
        for w, ph in zip(words, target):
            out.setdefault(w, sum(len(t) for t in target) - len(ph))
    wake_norm = " ".join(words)
    out.pop(wake_norm, None)
    out.pop(wake_word.lower(), None)
    # drop candidates that contain the wake word ("hey live kits"): not fair negatives
    wake_vars = phrase_variants(wake_word)
    out = {
        p: d
        for p, d in out.items()
        if min(_substring_distance(v, phrase_variants(p)[0]) for v in wake_vars) > 0
    }
    return sorted(out.items(), key=lambda kv: kv[1])


def _surface(phrase: str, rng: random.Random) -> str:
    """Random casing / punctuation so TTS prosody varies."""
    choice = rng.random()
    if choice < 0.4:
        phrase = phrase.title()
    elif choice < 0.55:
        phrase = phrase.capitalize()
    punct = rng.choice(["", "", ".", "!", "?", ","])
    return phrase + punct if punct != "," else phrase.replace(" ", ", ", 1)


@dataclass
class Utterance:
    """Ordered text segments; `key` is the index of the key phrase (wake word / confusable)."""

    segments: list[str]
    key: int | None
    form: str
    phrase: str
    distance: int = -1

    @property
    def text(self) -> str:
        return " ".join(self.segments)


class TextSampler:
    def __init__(
        self,
        wake_word: str,
        positive_texts: list[str],
        hard_negatives: list[str],
        p_carrier: float,
        rng: random.Random,
    ) -> None:
        self.wake_word = wake_word
        self.positive_texts = positive_texts or [wake_word]
        self.p_carrier = p_carrier
        self.rng = rng
        adv = adversarial_phrases(wake_word)
        self.adv = [(p, d) for p, d in adv] + [(h, 0) for h in hard_negatives]
        self.adv_weights = [{0: 6.0, 1: 4.0, 2: 2.0}.get(d, 1.0) for _, d in self.adv]

    def _wrap(self, key_text: str, phrase: str, distance: int = -1) -> Utterance:
        r = self.rng.random()
        if r < self.p_carrier / 2:
            return Utterance(
                [key_text.rstrip(".!?") + ",", self.rng.choice(CARRIER_AFTER)],
                0,
                "carrier_prefix",
                phrase,
                distance,
            )
        if r < self.p_carrier:
            return Utterance(
                [self.rng.choice(CARRIER_BEFORE), key_text], 1, "carrier_suffix", phrase, distance
            )
        return Utterance([key_text], 0, "isolated", phrase, distance)

    def positive(self) -> Utterance:
        w = self.rng.choice(self.positive_texts)
        return self._wrap(w, w)

    def adversarial(self) -> Utterance:
        p, d = self.rng.choices(self.adv, weights=self.adv_weights, k=1)[0]
        return self._wrap(_surface(p, self.rng), p, d)

    def speech(self) -> Utterance:
        k = self.rng.choice([1, 1, 1, 2])
        return Utterance([" ".join(self.rng.sample(FILLER_SPEECH, k))], None, "filler", "")


def word_phones(word: str) -> list[list[str]]:
    """All stress-free CMUdict pronunciations of a word; OOV words are split into known sub-words ("livekit" -> live + kit), else spelled out letter by letter."""
    cmu = _cmu()
    w = re.sub(r"[^a-z']", "", word.lower())
    if not w:
        return []
    if w in cmu:
        return [_strip_stress(p) for p in cmu[w]]
    parts = _split_unknown(w, cmu)
    if all(p in cmu for p in parts):
        seqs: list[list[str]] = [[]]
        for p in parts:
            seqs = [s + _strip_stress(v) for s in seqs for v in cmu[p][:2]]
        return seqs
    return [list(w.upper())]


def phrase_variants(text: str, limit: int = 16) -> list[list[str]]:
    """Phoneme sequences for a phrase (cartesian product over word pronunciations)."""
    seqs: list[list[str]] = [[]]
    for w in re.findall(r"[a-zA-Z']+", text):
        prons = word_phones(w)
        if not prons:
            continue
        seqs = [s + p for s in seqs for p in prons][:limit]
    return seqs


def _substring_distance(pattern: list[str], text: list[str]) -> int:
    """Min edit distance between `pattern` and any substring of `text` (Sellers' algorithm)."""
    if not pattern:
        return 0
    prev = list(range(len(pattern) + 1))
    best = prev[-1]
    for t in text:
        cur = [0]
        for j, p in enumerate(pattern, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (p != t)))
        best = min(best, cur[-1])
        prev = cur
    return best


def phonetic_distance(transcript: str, wake_word: str) -> int:
    """How far the closest stretch of `transcript` is from the wake word, in phonemes (0 = the wake word is said exactly, under some pronunciation of each word)."""
    wake = phrase_variants(wake_word)
    words = re.findall(r"[a-zA-Z']+", transcript)
    if not words or not wake:
        return 99
    text = [ph for w in words for ph in (word_phones(w) or [[]])[0]]
    return min(_substring_distance(v, text) for v in wake)
