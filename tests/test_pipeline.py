"""Offline end-to-end: synthetic dataset -> loader -> bench -> metrics -> report.

The "wake word" is a 1 kHz tone; ToneDetector scores tone energy in a sliding window.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from livekit_wakeword_bench import (
    Prediction,
    SlidingWindowDetector,
    WakeWordData,
    evaluate,
    run_detector,
)
from livekit_wakeword_bench.dataset import ShardWriter
from livekit_wakeword_bench.mix import active_rms, place_and_mix
from livekit_wakeword_bench.report import render_report
from livekit_wakeword_bench.text import TextSampler, adversarial_phrases

SR = 16000


def tone(sec: float, f: float = 1000.0) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)


@pytest.fixture(scope="module")
def ds_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ds")
    w = ShardWriter(root, "test", SR, shard_size=7)
    rng = np.random.default_rng(0)
    for i in range(10):  # positives: tone at 2.0-2.6 s
        x = 0.01 * rng.standard_normal(int(3.5 * SR)).astype(np.float32)
        x[2 * SR : int(2.6 * SR)] += tone(0.6)
        w.add(
            x,
            {
                "id": f"p{i}",
                "label": 1,
                "category": "positive",
                "onset_s": 2.0,
                "offset_s": 2.6,
                "phrase_onset_s": 2.0,
                "phrase_offset_s": 2.6,
                "noise_bucket": "quiet" if i < 5 else "extreme",
            },
        )
    for i in range(6):  # negatives: noise, one with a 500 Hz tone (should not trigger)
        x = 0.01 * rng.standard_normal(int(10 * SR)).astype(np.float32)
        if i == 0:
            x[SR : 2 * SR] += tone(1.0, 500.0)
        w.add(x, {"id": f"n{i}", "label": 0, "category": "noise", "noise_bucket": "noise_only"})
    # One "hard" negative that does contain the tone -> one false alarm.
    x = 0.01 * rng.standard_normal(int(5 * SR)).astype(np.float32)
    x[2 * SR : int(2.6 * SR)] += tone(0.6)
    w.add(
        x,
        {
            "id": "adv0",
            "label": 0,
            "category": "adversarial",
            "phrase": "tone",
            "noise_bucket": "quiet",
        },
    )
    w.flush()
    return root


def tone_detector() -> SlidingWindowDetector:
    def score(win: np.ndarray) -> float:
        seg = win[-int(0.4 * SR) :]
        spec = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(len(seg), 1 / SR)
        band = spec[(freqs > 950) & (freqs < 1050)].sum()
        return float(min(1.0, band / (spec.sum() + 1e-9) * 3))

    return SlidingWindowDetector(score, "tone", window_s=1.0, hop_s=0.05)


def test_loader(ds_dir: Path) -> None:
    data = WakeWordData(ds_dir, "test")
    assert len(data) == 17
    assert len(data.filter(category="positive", noise_bucket=["extreme"])) == 5
    s = data[0]
    assert s.audio.dtype == np.float32 and s.sample_rate == SR
    x, y = data.windows(2.0).arrays()
    assert x.shape == (17, 2 * SR) and y.sum() == 10
    # window for positives ends shortly after the phrase
    p = data.filter(category="positive").windows(1.0)[0]
    assert np.abs(p.audio[-int(0.5 * SR) :]).max() > 0.3


def test_bench(ds_dir: Path, tmp_path: Path) -> None:
    data = WakeWordData(ds_dir, "test")
    res = evaluate(run_detector(tone_detector(), data, progress=False), data, "tone")
    s = res.summary()
    a = s["at_threshold"]
    assert a["recall"] == 1.0
    assert s["n_negative"] == 7
    # adversarial clips are excluded from FPPH and reported separately
    assert a["false_accept_rate"] == 0.0 and a["fpph"] == 0.0
    assert a["adversarial_false_accept_rate"] == 1.0
    assert s["fpph_hours"] == pytest.approx(60 / 3600, abs=1e-3)
    assert a["latency_ms_p50"] is not None and -600 < a["latency_ms_p50"] < 500
    bd = res.breakdown()
    assert {r["value"] for r in bd["positive/noise_bucket"]} == {"quiet", "extreme"}
    assert bd["negative/adversarial_phrase"][0]["false_accept_rate"] == 1.0
    f = res.failures()
    assert [r["id"] for r in f["false_alarms"]] == ["adv0"]
    out = res.save(tmp_path / "run")
    from livekit_wakeword_bench import BenchResult

    again = BenchResult.load(out)
    assert again.summary()["at_threshold"] == a
    html = render_report([res, again], tmp_path / "r.html", data=data).read_text(encoding="utf-8")
    assert "adv0" in html and "data:audio/wav;base64" in html


def test_event_detector(ds_dir: Path) -> None:
    class Oracle:
        name = "oracle"

        def detect(self, audio: np.ndarray, sample_rate: int) -> Prediction:
            return (
                Prediction.from_events([2.7])
                if len(audio) == int(3.5 * SR)
                else Prediction.from_events([])
            )

    data = WakeWordData(ds_dir, "test")
    s = evaluate(run_detector(Oracle(), data, progress=False), data, "oracle").summary()
    assert s["at_threshold"]["recall"] == 1.0 and s["at_threshold"]["fpph"] == 0.0
    assert s["at_threshold"]["latency_ms_p50"] == pytest.approx(100, abs=1)


def test_snr() -> None:
    rng = np.random.default_rng(0)
    speech = tone(1.0)
    noise = rng.standard_normal(SR * 3).astype(np.float32)
    m = place_and_mix(speech, noise, 5.0, SR, 0.5, 0.5, rng=rng)
    n_part = m.audio[: int(0.5 * SR)]
    snr = 20 * math.log10(active_rms(speech, SR) / np.sqrt(np.mean(n_part**2)))
    assert abs(snr - 5.0) < 0.6


def test_adversarial() -> None:
    adv = dict(adversarial_phrases("hey livekit"))
    assert "hey livekit" not in adv and "hey live kit" not in adv
    assert "hey live kid" in adv and "hey kit" in adv
    import random

    ts = TextSampler("hey livekit", ["Hey LiveKit"], ["hey lifekit"], 1.0, random.Random(0))
    u = ts.positive()
    assert u.form.startswith("carrier") and "Hey LiveKit" in u.segments[u.key]


def test_augmenter_exact_snr_and_chain() -> None:
    """Scene mixing hits the requested SNR exactly (component-wise), and the device chain
    records what it did."""
    from livekit_wakeword_bench.augment import Augmenter, apply_codec
    from livekit_wakeword_bench.config import AugmentSpec
    from livekit_wakeword_bench.mix import active_rms

    class Pool:
        def __init__(self, x: np.ndarray) -> None:
            self.x = x

        def __bool__(self) -> bool:
            return True

        def sample(self, rng, sr=SR):  # noqa: ANN001
            class It:
                source, category, name = "fake", "street/fake", "f.flac"

            return It(), self.x

    class Pools:
        def noise(self, pool):  # noqa: ANN001
            return Pool(np.random.default_rng(1).standard_normal(SR * 4).astype(np.float32))

        def rir(self, pool):  # noqa: ANN001
            return None

        def speech(self, pool):  # noqa: ANN001
            return None

    spec = AugmentSpec(snr_buckets={"only": (7.0, 7.0)}, clean_weight=0.0, p_second_noise=0.0)
    aug = Augmenter(spec, Pools(), "test")
    speech = tone(1.0)
    canvas = np.concatenate([np.zeros(SR, np.float32), speech, np.zeros(SR, np.float32)])
    meta: dict = {}
    out = aug.scene(canvas, speech, np.random.default_rng(0), meta)
    noise = out - canvas
    snr = 20 * math.log10(active_rms(speech, SR) / np.sqrt(np.mean(noise**2)))
    assert snr == pytest.approx(7.0, abs=1e-3)
    assert meta["noise_bucket"] == "only" and meta["noise_type"] == "street"

    full = AugmentSpec(p_eq=1.0, p_bandlimit=1.0, p_codec=1.0, p_clip=0.0)
    meta2: dict = {}
    y = Augmenter(full, Pools(), "test").device(out, np.random.default_rng(3), meta2)
    assert len(y) == len(out) and np.isfinite(y).all()
    assert meta2["eq"] and meta2["bandlimit"] != "none" and meta2["codec"] != "none"
    for c in ("opus_8k", "mp3_32k", "mulaw_8khz"):
        z = apply_codec(out * 0.3, SR, c)
        assert len(z) == len(out) and np.corrcoef(z, out)[0, 1] > 0.5
        # no codec delay: timestamps (phrase_onset_s / latency) must stay valid
        cc = np.correlate(z[SR : 2 * SR], out[SR : 2 * SR] * 0.3, "full")
        assert abs(int(np.argmax(cc)) - (SR - 1)) <= 2


def test_phonetics() -> None:
    from livekit_wakeword_bench.text import phonetic_distance

    assert phonetic_distance("Hey, LiveKit!", "hey livekit") == 0
    assert phonetic_distance("okay hey live kit call mom", "hey livekit") == 0
    assert phonetic_distance("Hey life kid", "hey livekit") == 2
    assert phonetic_distance("the weather is nice", "hey livekit") > 3
    adv = dict(adversarial_phrases("hey livekit"))
    # phrases that acoustically contain the wake word are not fair negatives
    assert "hey live kits" not in adv and "hey live kitty" not in adv


def test_phoneme_rendering() -> None:
    """Pinned to the providers' documented examples."""
    from livekit_wakeword_bench.phonemes import (
        render_phrase,
        supports_phonemes,
        to_cartesia,
        to_ipa,
        to_rime,
        word_arpabet,
    )

    # Cartesia docs: cat, today (stress before the onset), America, jambalaya, letter (ɝ)
    assert to_cartesia(word_arpabet("cat")[0]) == "<<ˈ|k|æ|t>>"
    assert to_cartesia(word_arpabet("today")[0]) == "<<t|ə|ˈ|d|eɪ>>"
    assert to_cartesia(word_arpabet("america")[0]) == "<<ə|ˈ|m|ɛ|ɹ|ə|k|ə>>"
    assert to_cartesia(word_arpabet("jambalaya")[0]) == "<<ˌ|dʒ|ə|m|b|ə|ˈ|l|aɪ|ə>>"
    assert to_cartesia(word_arpabet("letter")[0]) == "<<ˈ|l|ɛ|t|ɝ>>"
    # Rime docs: custom {k1Ast0xm}, comma {k1am0x}
    assert to_rime(word_arpabet("custom")[0]) == "{k1Ast0xm}"
    assert to_rime(word_arpabet("comma")[0]) == "{k1am0x}"
    assert to_ipa(word_arpabet("crete")[0]) == "ˈkɹit"
    ov = {"livekit": "L AY1 V | K IH2 T"}
    assert render_phrase("Hey, LiveKit!", "rime/mistv3", ov) == "{h1e}, {l1Yv} {k1It}!"
    assert render_phrase("Hey, LiveKit!", "inworld/inworld-tts-2", ov) == "/ˈheɪ/, /ˈlaɪvˌkɪt/!"
    assert render_phrase("Hey xyzzyq", "cartesia/sonic-3") is None  # unknown word
    assert supports_phonemes("rime/mistv3") and not supports_phonemes("rime/coda")
    assert not supports_phonemes("deepgram/aura-2") and not supports_phonemes("fishaudio/s2-pro")


def test_silent_model_scores_zero(ds_dir: Path) -> None:
    class Silent:
        name = "silent"

        def detect(self, audio: np.ndarray, sample_rate: int) -> Prediction:
            t = np.arange(1, len(audio) // 1280 + 1) * 0.08
            return Prediction(t, np.zeros_like(t))

    data = WakeWordData(ds_dir, "test")
    s = evaluate(run_detector(Silent(), data, progress=False), data, "silent").summary()
    assert s["score"] == 0.0 and s["curve"]["recall"][0] == 0.0


def test_media_negatives(ds_dir: Path, tmp_path: Path) -> None:
    from livekit_wakeword_bench.bench import _says, add_media

    media = tmp_path / "media"
    w = ShardWriter(media, "media", SR, shard_size=10)
    rng = np.random.default_rng(1)
    for i, kind in enumerate(["music", "tv_youtube", "podcast"]):
        x = 0.01 * rng.standard_normal(30 * SR).astype(np.float32)
        if kind == "music":  # two tones 10 s apart: two false alarms in 30 s
            x[5 * SR : int(5.6 * SR)] += tone(0.6)
            x[15 * SR : int(15.6 * SR)] += tone(0.6)
        w.add(
            x,
            {"id": f"m{i}", "label": 0, "category": "media", "noise_type": kind, "provider": kind},
        )
    w.flush()
    data = WakeWordData(ds_dir, "test")
    res = evaluate(run_detector(tone_detector(), data, progress=False), data, "tone")
    score_before = res.summary()["score"]
    add_media(res, tone_detector(), data, media, progress=False)
    s = res.summary()
    m = s["media"]
    assert m["hours"] == pytest.approx(90 / 3600, abs=1e-3)
    assert m["by_type"]["music"]["at_threshold"] == pytest.approx(2 / (30 / 3600))
    assert m["by_type"]["podcast"]["at_threshold"] == 0.0
    assert s["score"] == score_before and s["at_threshold"]["fpph"] == 0.0  # not part of FPPH
    assert _says("Okay, Hey Jarvis!", ["hey jarvis"]) and not _says("hey jarvisson", ["hey jarvis"])


def test_party_scene() -> None:
    """A party scene adds 2-4 competing talkers, each at its own SIR, plus crowd babble at the
    requested SNR, and labels the clip as noise bucket "party"."""
    from livekit_wakeword_bench.augment import Augmenter
    from livekit_wakeword_bench.config import AugmentSpec
    from livekit_wakeword_bench.mix import active_rms

    rng0 = np.random.default_rng(5)

    class It:
        def __init__(self, source: str, category: str) -> None:
            self.source, self.category, self.name = source, category, "f.flac"

    class Speech:
        def __bool__(self) -> bool:
            return True

        def sample_segment(self, rng, sr, seconds):  # noqa: ANN001
            return It("ami", "speech/meeting"), rng0.standard_normal(int(seconds * sr)), ""

    class Noise:
        def __bool__(self) -> bool:
            return True

        def sample_type(self, rng, coarse, sr=SR):  # noqa: ANN001
            assert coarse == "babble"
            return It("wham", "babble/restaurant_bar"), rng0.standard_normal(SR * 6)

        def sample(self, rng, sr=SR):  # noqa: ANN001
            return It("demand", "street/traffic"), rng0.standard_normal(SR * 6)

    class Pools:
        def noise(self, pool):  # noqa: ANN001
            return Noise()

        def rir(self, pool):  # noqa: ANN001
            return None

        def speech(self, pool):  # noqa: ANN001
            return Speech()

    spec = AugmentSpec(p_party=1.0, party_babble_snr_db=(4.0, 4.0), party_sir_db=(6.0, 6.0))
    aug = Augmenter(spec, Pools(), "test")
    speech = tone(1.0)
    canvas = np.concatenate([np.zeros(SR, np.float32), speech, np.zeros(SR, np.float32)])
    for seed in range(8):
        meta: dict = {}
        out = aug.scene(canvas, speech, np.random.default_rng(seed), meta)
        assert meta["party"] and meta["noise_bucket"] == "party" and 2 <= meta["n_talkers"] <= 4
        assert meta["noise_type"] == "babble" and meta["snr_db"] == 4.0 and meta["sir_db"] == 6.0
        # talkers + babble: total interference power = n * (talker at 6 dB) + babble at 4 dB
        ref = active_rms(speech, SR)
        want = meta["n_talkers"] * (ref / 10**0.3) ** 2 + (ref / 10**0.2) ** 2
        assert np.mean((out - canvas) ** 2) == pytest.approx(
            want, rel=0.03
        )  # independent: powers add
        assert aug.describe(meta).startswith("party")
    # noise-only clips and clips that may not carry a talker never become parties
    meta = {}
    aug.scene(canvas, speech, np.random.default_rng(0), meta, allow_talker=False)
    assert not meta.get("party")


def test_party_summary(tmp_path: Path) -> None:
    """summary()["party"]: recall and false accepts on party clips (also part of the score)."""
    root = tmp_path / "ds"
    w = ShardWriter(root, "test", SR, shard_size=20)
    rng = np.random.default_rng(0)
    for i in range(8):
        x = 0.01 * rng.standard_normal(int(3.5 * SR)).astype(np.float32)
        party = i >= 4
        if i not in (6, 7):  # two party wake words are missed: recall 0.5 in the party
            x[2 * SR : int(2.6 * SR)] += tone(0.6)
        w.add(
            x,
            {
                "id": f"p{i}",
                "label": 1,
                "category": "positive",
                "phrase_onset_s": 2.0,
                "phrase_offset_s": 2.6,
                "onset_s": 2.0,
                "offset_s": 2.6,
                "party": party,
                "n_talkers": 3 if party else 0,
                "noise_bucket": "party" if party else "quiet",
            },
        )
    for i in range(4):
        x = 0.01 * rng.standard_normal(int(10 * SR)).astype(np.float32)
        if i == 0:  # one party negative triggers
            x[SR : int(1.6 * SR)] += tone(0.6)
        w.add(
            x,
            {
                "id": f"s{i}",
                "label": 0,
                "category": "speech",
                "party": i < 2,
                "noise_bucket": "party" if i < 2 else "quiet",
            },
        )
    w.flush()
    data = WakeWordData(root, "test")
    s = evaluate(run_detector(tone_detector(), data, progress=False), data, "tone").summary()
    pt = s["party"]
    assert pt["n_positive"] == 4 and pt["n_negative"] == 2
    assert pt["recall"]["at_threshold"] == 0.5
    assert pt["false_accept_rate"]["at_threshold"] == 0.5
    assert pt["recall_by_talkers"] == {"3": 0.5}
    assert s["at_threshold"]["recall"] == 0.75  # party clips count in the headline too


def test_negative_sentences_never_say_the_wake_word():
    import random

    from livekit_wakeword_bench.text import FILLER_SPEECH, TextSampler, phonetic_distance

    for wake in ("alexa", "hey siri", "okay google", "hey livekit"):
        s = TextSampler(wake, [wake], [], 0.35, random.Random(0))
        assert all(
            phonetic_distance(t, wake) > 0 for t in s.filler + s.carrier_after + s.carrier_before
        )
        for _ in range(200):
            assert phonetic_distance(s.speech().text, wake) > 0
    s = TextSampler("hey livekit", ["hey livekit"], [], 0.35, random.Random(0))
    assert (
        s.filler == FILLER_SPEECH
    )  # unaffected wake words keep the exact same sentence pool (reproducible)


def test_phrases_without_latin_words():
    """Phrases in other scripts have no English phonemes: nothing counts as saying them, and
    there are no confusables to generate."""
    import random

    from livekit_wakeword_bench.config import make_config
    from livekit_wakeword_bench.generate import Generator
    from livekit_wakeword_bench.text import FILLER_SPEECH, TextSampler, phonetic_distance

    assert phonetic_distance("okay so", "ねえ ロボット") > 0
    s = TextSampler("你好小智", ["你好小智"], [], 0.35, random.Random(0))
    assert s.filler == FILLER_SPEECH and s.carrier_after and s.carrier_before and not s.adv
    for _ in range(50):
        s.positive()
    cfg = make_config("ねえ ロボット", None, "small", None, {"language": "ja"})
    plans = Generator(cfg, "unused").plan("train", cfg.splits["train"])
    cats = {p.category for p in plans}
    assert "positive" in cats and "speech" in cats and "adversarial" not in cats


def test_voices_follow_the_language():
    """Each language gets every model (and voice) that speaks it."""
    from livekit_wakeword_bench.config import make_config
    from livekit_wakeword_bench.generate import Generator
    from livekit_wakeword_bench.voices import base_language, load_voices, voices_for_language

    def models(language: str) -> set[str]:
        return {v.model for v in voices_for_language(load_voices(), language)}

    everything = {"cartesia/sonic-3", "inworld/inworld-tts-2", "xai/tts-1", "rime/mistv3"}
    assert models("en") == models("es") == models("de") == models("fr") == everything
    assert models("pt-BR") == models("ja") == everything - {"rime/mistv3"}
    assert models("hi") == everything - {"rime/mistv3"}  # Rime has no Hindi speakers yet
    assert models("sw") == {"inworld/inworld-tts-2"} and models("nb") == models("no")
    assert models("tlh") == set() and base_language("pt_BR") == "pt"
    # Rime speakers speak their own languages; English keeps exactly its old voices
    es = {v.id for v in voices_for_language(load_voices(models=["rime/mistv3"]), "es")}
    en = {v.id for v in voices_for_language(load_voices(models=["rime/mistv3"]), "en")}
    assert {"diego", "isa"} <= es and "diego" not in en and len(en) == 62 and es & en

    def generator(language: str, **kv: object) -> Generator:
        cfg = make_config("hola asistente", None, "small", None, {"language": language, **kv})
        return Generator(cfg, "unused")

    g = generator("es", **{"tts.phonemes": False})
    assert {v.model for v in g.train_voices + g.test_voices} == everything
    g = generator("es", **{"tts.models": ["cartesia/sonic-3"]})  # explicit models win
    assert {v.model for v in g.train_voices + g.test_voices} == {"cartesia/sonic-3"}
    with pytest.raises(ValueError, match="no bundled TTS voice speaks"):
        generator("tlh")
