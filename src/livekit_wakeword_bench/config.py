"""Benchmark / training-set generation config (YAML + pydantic)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class SplitSpec(BaseModel):
    """How many samples of each category to generate for one split."""

    n_positive: int = 0
    n_adversarial: int = 0
    n_speech: int = 0
    real_speech_minutes: float = 0.0
    n_noise: int = 0
    augment: bool = True


def _default_snr_buckets() -> dict[str, tuple[float, float]]:
    return {
        "quiet": (20.0, 30.0),
        "moderate": (10.0, 20.0),
        "noisy": (5.0, 10.0),
        "very_noisy": (0.0, 5.0),
        "extreme": (-5.0, 0.0),
    }


class AugmentSpec(BaseModel):
    # each clip picks "clean" or one bucket, then a uniform SNR inside it
    snr_buckets: dict[str, tuple[float, float]] = Field(default_factory=_default_snr_buckets)
    snr_weights: dict[str, float] = Field(default_factory=dict)  # missing = 1.0
    clean_weight: float = 1.0
    p_reverb: float = 0.5
    p_interferer: float = 0.2
    sir_db: tuple[float, float] = (0.0, 20.0)
    p_second_noise: float = 0.2
    p_eq: float = 0.4
    p_bandlimit: float = 0.15
    level_dbfs: tuple[float, float] = (-30.0, -3.0)
    p_clip: float = 0.05
    p_codec: float = 0.25
    lead_s: tuple[float, float] = (0.3, 1.5)
    tail_s: tuple[float, float] = (0.4, 1.2)
    segment_s: tuple[float, float] = (8.0, 30.0)


# LiveKit Inference concurrent TTS connections per plan (livekit.com/pricing)
PLAN_CONCURRENCY = {"build": 5, "ship": 20, "scale": 50}


class TTSSpec(BaseModel):
    models: list[str] = Field(default_factory=list)
    plan: Literal["build", "ship", "scale", "enterprise"] = "build"
    # overrides plan (also env LKWB_TTS_CONCURRENCY); required for enterprise
    concurrency: int | None = None
    sample_rate: int = 24000
    test_voice_fraction: float = 0.3
    speed: tuple[float, float] = (0.85, 1.2)
    retries: int = 3
    phonemes: bool = True

    def max_concurrency(self) -> int:
        """Concurrent TTS requests: explicit value > env var > plan limit."""
        import os

        if self.concurrency:
            return self.concurrency
        env = os.getenv("LKWB_TTS_CONCURRENCY")
        if env:
            return int(env)
        if self.plan == "enterprise":
            raise ValueError("plan=enterprise: set tts.concurrency to your account's limit")
        return PLAN_CONCURRENCY[self.plan]


class BenchConfig(BaseModel):
    name: str = "hey_livekit"
    wake_word: str = "hey livekit"
    positive_texts: list[str] = Field(
        default_factory=lambda: ["Hey LiveKit", "Hey, LiveKit!", "Hey LiveKit?", "Hey, LiveKit."]
    )
    # ARPAbet for words missing from CMUdict, "|" between morphemes: {"livekit": "L AY1 V | K IH2 T"}
    pronunciations: dict[str, str] = Field(default_factory=dict)
    hard_negatives: list[str] = Field(default_factory=list)
    p_carrier: float = 0.35
    language: str = "en"
    seed: int = 1234
    sample_rate: int = 16000
    cache_dir: str = ".cache/lkwb"
    # noise / RIR / speech sources (`lkwb sources`); empty = all
    sources: list[str] = Field(default_factory=list)
    tts: TTSSpec = Field(default_factory=TTSSpec)
    augment: AugmentSpec = Field(default_factory=AugmentSpec)
    splits: dict[str, SplitSpec] = Field(
        default_factory=lambda: {
            "train": SplitSpec(n_positive=2000, n_adversarial=2000, n_speech=500, augment=False),
            "validation": SplitSpec(n_positive=200, n_adversarial=200, n_speech=100, n_noise=50),
            "test": SplitSpec(
                n_positive=1000,
                n_adversarial=1000,
                n_speech=300,
                real_speech_minutes=60,
                n_noise=200,
            ),
        }
    )

    @property
    def cache_path(self) -> Path:
        return Path(self.cache_dir)


SIZES = {
    "tiny": {
        "train": (40, 40, 10, 0, 0),
        "validation": (0, 0, 0, 0, 0),
        "test": (100, 100, 30, 10, 5),
    },
    "small": {
        "train": (1000, 1000, 300, 0, 0),
        "validation": (100, 100, 30, 20, 10),
        "test": (500, 500, 150, 50, 30),
    },
    "default": {
        "train": (10000, 10000, 3000, 0, 0),
        "validation": (1000, 1000, 300, 100, 30),
        "test": (3000, 3000, 1000, 300, 180),
    },
}  # per split: n_positive, n_adversarial, n_speech, n_noise, real_speech_minutes


def sized_splits(size: str) -> dict[str, SplitSpec]:
    out = {}
    for name, (p, a, s, n, r) in SIZES[size].items():
        if p or a or s or n or r:
            out[name] = SplitSpec(
                n_positive=p,
                n_adversarial=a,
                n_speech=s,
                n_noise=n,
                real_speech_minutes=r,
                augment=name != "train",
            )
    return out


def set_knob(cfg: BenchConfig, key: str, value: Any) -> BenchConfig:
    """Set a dotted config path, e.g. set_knob(cfg, "augment.p_reverb", 0.3)."""
    data = cfg.model_dump()
    node = data
    *parents, leaf = key.split(".")
    for p in parents:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"unknown knob {key!r}")
        node = node[p]
    if leaf not in node and parents[:1] != ["splits"] and parents[:1] != ["pronunciations"]:
        raise KeyError(f"unknown knob {key!r}")
    if isinstance(value, str):
        value = yaml.safe_load(value)
    node[leaf] = value
    return BenchConfig.model_validate(data)


def make_config(
    wake_word: str | None = None,
    config: str | Path | None = None,
    size: str | None = None,
    plan: str | None = None,
    knobs: dict[str, Any] | list[str] | None = None,
) -> BenchConfig:
    """Config from a YAML file and/or a bare wake word, then knob overrides.

    knobs: {"augment.p_reverb": 0.3} or ["augment.p_reverb=0.3", ...]
    """
    cfg = load_config(config)
    unknown: list[str] = []
    if wake_word:
        from .scaffold import guess_pronunciations, positive_texts

        guesses, unknown = guess_pronunciations(wake_word)
        cfg.wake_word = wake_word
        cfg.name = re.sub(r"[^a-z0-9]+", "_", wake_word.lower()).strip("_")
        cfg.positive_texts = positive_texts(wake_word)
        cfg.pronunciations = {**guesses, **cfg.pronunciations}
    if size:
        cfg.splits = sized_splits(size)
    if plan:
        cfg.tts.plan = plan  # type: ignore[assignment]
    items = knobs.items() if isinstance(knobs, dict) else (k.split("=", 1) for k in knobs or [])
    for k, v in items:
        cfg = set_knob(cfg, k.strip(), v)
    if wake_word:
        missing = [w for w in unknown if w not in cfg.pronunciations]
        if missing:
            raise ValueError(
                f"no pronunciation for {missing}: pass a knob like "
                f'"pronunciations.{missing[0]}=K AE1 T" or use `lkwb init`'
            )
    return cfg


def load_config(path: str | Path | None) -> BenchConfig:
    if path is None:
        return BenchConfig()
    data = yaml.safe_load(Path(path).read_text()) or {}
    return BenchConfig.model_validate(data)
