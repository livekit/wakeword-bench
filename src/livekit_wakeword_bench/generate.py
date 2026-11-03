"""Dataset generation: plan -> synthesize (LiveKit Inference TTS) -> mix -> write parquet shards."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from .augment import Augmenter
from .config import (
    BenchConfig,
    SplitSpec,  # noqa: F811
)
from .dataset import ShardWriter
from .mix import place_and_mix, resample, set_peak, trim_silence
from .phonemes import (
    arpabet_string,
    phrase_parts,
    provider_options,
    render_phrase,
    supports_phonemes,
    validate_arpabet,
)
from .sources import load_pools
from .text import TextSampler, Utterance
from .tts import Synthesizer, TTSRequest, speed_kwargs
from .voices import Voice, load_voices, split_voices

logger = logging.getLogger(__name__)

TRAIN_LIKE = ("train", "validation")


def _seed(*parts: Any) -> int:
    """Process-independent seed (built-in hash() of str is randomized per process)."""
    return int(hashlib.sha1(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


@dataclass
class Plan:
    id: str
    category: str
    utt: Utterance | None
    voice: Voice | None
    speed: float


class Generator:
    def __init__(self, cfg: BenchConfig, out_dir: str | Path) -> None:
        self.cfg = cfg
        self.out = Path(out_dir)
        self.sr = cfg.sample_rate
        voices = load_voices(models=cfg.tts.models or None)
        if cfg.tts.phonemes:
            dropped = sorted({v.model for v in voices if not supports_phonemes(v.model)})
            if dropped:
                logger.warning("skipping models without phoneme input: %s", dropped)
            voices = [v for v in voices if supports_phonemes(v.model)]
            bad = {w: e for w, p in cfg.pronunciations.items() if (e := validate_arpabet(p))}
            if bad:
                raise ValueError(f"invalid `pronunciations`: {bad} (see `lkwb pronounce`)")
            if render_phrase(cfg.wake_word, "cartesia/sonic-3", cfg.pronunciations) is None:
                raise ValueError(
                    f"no pronunciation for {cfg.wake_word!r}: add the unknown word(s) to "
                    "`pronunciations` (ARPAbet, e.g. livekit: 'L AY1 V | K IH2 T')"
                )
        if not voices:
            raise ValueError("voice catalog is empty for the selected models")
        self.train_voices, self.test_voices = split_voices(
            voices, cfg.tts.test_voice_fraction, cfg.seed
        )
        self.pools = load_pools(cfg.cache_path, cfg.sources or None)
        self._augmenters: dict[str, Augmenter] = {}

    def _voices_for(self, split: str) -> list[Voice]:
        return self.train_voices if split in TRAIN_LIKE else self.test_voices

    def _pick_voice(self, rng: random.Random, voices: list[Voice]) -> Voice:
        models = sorted({v.model for v in voices})
        m = rng.choice(models)
        return rng.choice([v for v in voices if v.model == m])

    def plan(self, split: str, spec: SplitSpec) -> list[Plan]:
        rng = random.Random(f"{self.cfg.seed}:{split}")
        hard = self.cfg.hard_negatives
        if self.cfg.tts.phonemes:
            known = [h for h in hard if phrase_parts(h, self.cfg.pronunciations) is not None]
            if len(known) < len(hard):
                logger.warning(
                    "hard_negatives without a pronunciation are skipped: %s",
                    sorted(set(hard) - set(known)),
                )
            hard = known
        texts = TextSampler(
            self.cfg.wake_word,
            self.cfg.positive_texts,
            hard,
            self.cfg.p_carrier,
            rng,
        )
        voices = self._voices_for(split)
        lo, hi = self.cfg.tts.speed
        plans: list[Plan] = []

        def add(cat: str, n: int, make: Any) -> None:
            for i in range(n):
                v = self._pick_voice(rng, voices)
                speed = round(rng.uniform(lo, hi) / 0.05) * 0.05  # quantized: better cache reuse
                if not speed_kwargs(v.model, speed):
                    speed = 1.0  # provider has no rate control; record the truth
                plans.append(Plan(f"{split}-{cat}-{i:06d}", cat, make(), v, speed))

        add("positive", spec.n_positive, texts.positive)
        n_adversarial = spec.n_adversarial
        if n_adversarial and not texts.adv:
            logger.warning(
                "no confusable phrases for %r (no English phonemes): skipping %d adversarial "
                "clips; add `hard_negatives` to have some",
                self.cfg.wake_word,
                n_adversarial,
            )
            n_adversarial = 0
        add("adversarial", n_adversarial, texts.adversarial)
        add("speech", spec.n_speech, texts.speech)
        for i in range(spec.n_noise):
            plans.append(Plan(f"{split}-noise-{i:06d}", "noise", None, None, 1.0))
        return plans

    def _tts_text(self, p: Plan, j: int) -> str:
        """Text sent to TTS for segment j: the key phrase as provider phonemes (exact pronunciation), carrier / filler segments as plain text."""
        assert p.utt and p.voice
        seg = p.utt.segments[j]
        if j == p.utt.key and self.cfg.tts.phonemes:
            rendered = render_phrase(seg, p.voice.model, self.cfg.pronunciations)
            if rendered is None:
                raise ValueError(f"no pronunciation for {seg!r}; add it to `pronunciations`")
            return rendered
        return seg

    def _requests(self, p: Plan) -> list[TTSRequest]:
        assert p.utt and p.voice
        extra = provider_options(p.voice.model) if self.cfg.tts.phonemes else {}
        return [
            TTSRequest(
                p.voice.model,
                p.voice.id,
                self._tts_text(p, j),
                round(p.speed, 3),
                self.cfg.language,
                extra=extra,
            )
            for j in range(len(p.utt.segments))
        ]

    def _synth(self) -> Synthesizer:
        return Synthesizer(
            self.cfg.cache_path,
            self.cfg.tts.max_concurrency(),
            self.cfg.tts.sample_rate,
            self.cfg.tts.retries,
        )

    async def _synth_all(self, plans: list[Plan]) -> set[str]:
        """Synthesize every utterance of every plan into the on-disk cache, under ONE shared concurrency limit (all splits at once)."""
        failed: set[str] = set()
        tts_plans = [p for p in plans if p.utt is not None]
        if not tts_plans:
            return failed
        conc = self.cfg.tts.max_concurrency()
        logger.info(
            "synthesizing %d utterances, %d concurrent TTS connections", len(tts_plans), conc
        )
        async with self._synth() as synth:
            bar = tqdm(total=len(tts_plans), desc="tts", unit="utt")

            async def one(p: Plan) -> None:
                ok = await asyncio.gather(*(synth.ensure(r) for r in self._requests(p)))
                if not all(ok):
                    failed.add(p.id)
                bar.update()

            await asyncio.gather(*(one(p) for p in tts_plans))
            bar.close()
        return failed

    def _pool_split(self, split: str) -> str:
        return "train" if split in TRAIN_LIKE else "test"

    def _render_speech(
        self, p: Plan, segs: list[np.ndarray], rng: np.random.Generator
    ) -> tuple[np.ndarray, float, float]:
        """Trim + join segments with natural pauses. Returns (speech, key_start_s, key_end_s)."""
        assert p.utt
        parts: list[np.ndarray] = []
        key_on = key_off = 0.0
        cursor = 0
        for j, seg in enumerate(segs):
            a = trim_silence(resample(seg, self.cfg.tts.sample_rate, self.sr), self.sr)
            if j > 0:
                gap = np.zeros(int(rng.uniform(0.1, 0.4) * self.sr), dtype=np.float32)
                parts.append(gap)
                cursor += len(gap)
            if p.utt.key == j:
                key_on, key_off = cursor / self.sr, (cursor + len(a)) / self.sr
            parts.append(a)
            cursor += len(a)
        speech = np.concatenate(parts)
        speech = speech / (np.abs(speech).max() + 1e-9) * 0.5
        return speech.astype(np.float32), key_on, key_off

    def _mix(
        self,
        p: Plan,
        segs: list[np.ndarray] | None,
        split: str,
        augment: bool,
        rng: np.random.Generator,
        real: tuple[np.ndarray, dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        a = self.cfg.augment
        meta: dict[str, Any] = {"id": p.id, "category": p.category, "wake_word": self.cfg.wake_word}
        meta["label"] = int(p.category == "positive")

        speech: np.ndarray | None = None
        key = (np.nan, np.nan)
        if p.utt is not None and segs is not None:
            speech, k_on, k_off = self._render_speech(p, segs, rng)
            key = (k_on, k_off) if p.utt.key is not None else (np.nan, np.nan)
            v = p.voice
            assert v
            meta.update(
                text=p.utt.text,
                text_form=p.utt.form,
                phrase=p.utt.phrase,
                adversarial_distance=p.utt.distance,
                provider=v.provider,
                model=v.model,
                voice_id=v.id,
                voice_name=v.name,
                gender=v.gender,
                accent=v.accent,
                held_out_voice=v in self.test_voices,
                speed=round(p.speed, 3),
                tts_key="|".join(r.key() for r in self._requests(p)),
                phonemes=arpabet_string(p.utt.phrase, self.cfg.pronunciations)
                if p.utt.key is not None
                else "",
                tts_input=self._tts_text(p, p.utt.key) if p.utt.key is not None else "",
            )
        elif real is not None:
            speech, rmeta = real
            meta.update(rmeta)

        aug = self._augmenter(split)
        if augment and speech is not None:
            speech = aug.reverb(
                speech, rng, meta
            )  # dry speech only: noise is already "in the room"

        lead = rng.uniform(*a.lead_s)
        tail = rng.uniform(*a.tail_s)
        if p.category in ("positive", "adversarial") and speech is not None:
            # >= 2 s of context before the key phrase ends, plus jitter so position isn't learnable
            if key[1] == key[1] and lead + key[1] < 2.0:
                lead = 2.0 - key[1] + float(rng.uniform(0.05, 1.0))

        total = float(rng.uniform(*a.segment_s)) if speech is None else None
        placed = place_and_mix(speech, None, None, self.sr, lead, tail, total, rng)
        audio = placed.audio
        if placed.onset_s is not None:
            meta.update(onset_s=placed.onset_s, offset_s=placed.offset_s)
            if key[0] == key[0]:
                meta.update(
                    phrase_onset_s=placed.onset_s + key[0], phrase_offset_s=placed.onset_s + key[1]
                )
        if augment or p.category == "noise":
            audio = aug.scene(
                audio,
                speech,
                rng,
                meta,
                noise_only=p.category == "noise",
                allow_talker=p.category in ("positive", "adversarial", "speech"),
            )
            audio = aug.device(audio, rng, meta)
        else:
            audio = set_peak(audio, -6.0)
            meta.update(noise_bucket="clean", level_dbfs=-6.0)
        meta["augmentations"] = aug.describe(meta)
        return audio, meta

    def _augmenter(self, split: str) -> Augmenter:
        ps = self._pool_split(split)
        if ps not in self._augmenters:
            self._augmenters[ps] = Augmenter(self.cfg.augment, self.pools, ps, self.sr)
        return self._augmenters[ps]

    def _real_speech(self, split: str, minutes: float, rng: np.random.Generator):  # type: ignore[no-untyped-def]
        pool = self.pools.speech(self._pool_split(split))
        if not pool or minutes <= 0:
            return
        need = minutes * 60
        i = 0
        while need > 0:
            seg_s = float(rng.uniform(*self.cfg.augment.segment_s))
            item, audio, transcript = pool.sample_segment(rng, self.sr, seg_s)
            need -= len(audio) / self.sr
            yield (
                Plan(f"{split}-real_speech-{i:06d}", "real_speech", None, None, 1.0),
                (
                    audio,
                    {
                        "text": transcript,
                        "text_form": "transcript",
                        "provider": item.source,
                        "voice_id": item.speaker,
                        "voice_name": item.speaker,
                        "held_out_voice": True,
                    },
                ),
            )
            i += 1

    def run(self, splits: list[str] | None = None, overwrite: bool = False) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        todo = {k: v for k, v in self.cfg.splits.items() if not splits or k in splits}
        for split in todo:
            for old in (self.out / "data").glob(f"{split}-*.parquet"):
                if not overwrite:
                    raise FileExistsError(f"{old} exists (use overwrite=True / --overwrite)")
                old.unlink()

        plans = {split: self.plan(split, spec) for split, spec in todo.items()}
        for split, ps in plans.items():
            logger.info("[%s] %d planned TTS/noise samples", split, len(ps))
        failed = asyncio.run(self._synth_all([p for ps in plans.values() for p in ps]))

        stats: dict[str, Any] = {}
        synth = self._synth()
        for split, spec in todo.items():
            writer = ShardWriter(self.out, split, self.sr)
            n_fail = 0
            for p in tqdm(plans[split], desc=f"mix {split}", unit="clip"):
                rng = np.random.default_rng(_seed(self.cfg.seed, p.id))
                segs = None
                if p.utt is not None:
                    if p.id in failed:
                        n_fail += 1
                        continue
                    segs = [synth.load(r) for r in self._requests(p)]
                wav, meta = self._mix(p, segs, split, spec.augment, rng)
                writer.add(wav, meta)
            rng = np.random.default_rng(_seed(self.cfg.seed, "real", split))
            for p, real in tqdm(
                self._real_speech(split, spec.real_speech_minutes, rng), desc=f"real speech {split}"
            ):
                prng = np.random.default_rng(_seed(self.cfg.seed, p.id))
                wav, meta = self._mix(p, None, split, spec.augment, prng, real=real)
                writer.add(wav, meta)
            writer.flush()
            stats[split] = {
                "rows": writer.n_rows,
                "hours": round(writer.seconds / 3600, 3),
                "tts_failures": n_fail,
            }
            logger.info("[%s] wrote %d rows (%.2f h)", split, writer.n_rows, writer.seconds / 3600)
        (self.out / "bench_config.yaml").write_text(
            yaml.safe_dump(self.cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
        from .card import write_card

        write_card(self.out, self.cfg, stats, self)
        return stats


def generate(
    wake_word: str | None = None,
    out: str | Path = "out/dataset",
    *,
    config: str | Path | BenchConfig | None = None,
    size: str | None = None,
    plan: str | None = None,
    splits: list[str] | None = None,
    overwrite: bool = False,
    **knobs: Any,
) -> dict[str, Any]:
    """Generate a dataset. Knobs are dotted config paths with "__" for ".":

    generate("hey jarvis", "out/hey_jarvis", size="small", plan="ship", augment__p_reverb=0.3)
    """
    from .config import make_config

    kv = {k.replace("__", "."): v for k, v in knobs.items()}
    if isinstance(config, BenchConfig):
        cfg = make_config(wake_word, None, size, plan, kv)
        cfg = config if not (wake_word or size or plan or kv) else cfg
    else:
        cfg = make_config(wake_word, config, size, plan, kv)
    return Generator(cfg, out).run(splits=splits, overwrite=overwrite)
