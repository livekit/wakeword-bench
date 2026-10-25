"""Augmentation chain, applied in physical order: room -> scene -> device -> transmission."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal

from .config import AugmentSpec
from .mix import BANDLIMITS, active_rms, apply_rir, bandlimit, fit_length, resample, set_peak

CODECS = ("opus_8k", "opus_16k", "opus_32k", "mp3_32k", "mulaw_8khz")


def _db_gain(x: np.ndarray, ref_rms: float, ratio_db: float) -> np.ndarray:
    """Scale x so that ref_rms / rms(x) == ratio_db (dB)."""
    rms = float(np.sqrt(np.mean(x**2)) + 1e-9)
    return (x * (ref_rms / (rms * 10 ** (ratio_db / 20)))).astype(np.float32)


def mic_eq(audio: np.ndarray, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Random device coloration: spectral tilt + 1-3 peaking bands (+/- 9 dB)."""
    out = audio.astype(np.float64)
    desc = []
    tilt = float(rng.uniform(-6, 6))  # dB at 8 kHz relative to 1 kHz, via a shelf
    if abs(tilt) > 1:
        sos = signal.butter(1, 1500, btype="highpass", fs=sr, output="sos")
        hp = signal.sosfilt(sos, out)
        out = out + (10 ** (tilt / 20) - 1) * hp
        desc.append(f"tilt{tilt:+.0f}")
    for _ in range(int(rng.integers(1, 4))):
        f0 = float(np.exp(rng.uniform(np.log(150), np.log(6000))))
        gain = float(rng.uniform(-9, 9))
        q = float(rng.uniform(0.7, 4))
        a = 10 ** (gain / 40)
        w0 = 2 * np.pi * f0 / sr
        alpha = np.sin(w0) / (2 * q)
        b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
        den = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
        out = signal.lfilter(b, den, out)
        desc.append(f"pk{f0:.0f}{gain:+.0f}")
    return out.astype(np.float32), " ".join(desc)


def apply_codec(audio: np.ndarray, sr: int, codec: str) -> np.ndarray:
    """Round-trip through a lossy codec (soundfile/libsndfile, no external binaries)."""
    n = len(audio)
    x = np.clip(audio, -1, 1)
    if codec == "mulaw_8khz":  # narrowband telephone: 8 kHz, 8-bit mu-law
        y = resample(x, sr, 8000)
        mu = 255.0
        q = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
        q = np.round(q * 127) / 127
        y = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
        out = resample(y.astype(np.float32), 8000, sr)
    else:
        fmt, sub = ("OGG", "OPUS") if codec.startswith("opus") else ("MP3", "MPEG_LAYER_III")
        # compression_level: 0 = best quality / highest bitrate, 1 = smallest.
        level = {"opus_8k": 1.0, "opus_16k": 0.75, "opus_32k": 0.45, "mp3_32k": 0.9}[codec]
        buf = io.BytesIO()
        sf.write(buf, x, sr, format=fmt, subtype=sub, compression_level=level)
        buf.seek(0)
        out, sr2 = sf.read(buf, dtype="float32")
        out = out.mean(axis=1) if out.ndim > 1 else out
        if sr2 != sr:
            out = resample(out, sr2, sr)
    out = out[:n]
    if len(out) < n:
        out = np.concatenate([out, np.zeros(n - len(out), np.float32)])
    return out.astype(np.float32)


@dataclass
class AugResult:
    audio: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


class Augmenter:
    """Applies the chain using one pool ("train" or "test") of noise / RIR / speech."""

    def __init__(self, spec: AugmentSpec, pools: Any, pool: str, sr: int = 16000) -> None:
        self.spec = spec
        self.pools = pools
        self.pool = pool
        self.sr = sr

    def noise_bucket(self, rng: np.random.Generator) -> tuple[str, float | None]:
        a = self.spec
        names = ["clean", *a.snr_buckets]
        w = np.array([a.clean_weight] + [a.snr_weights.get(n, 1.0) for n in a.snr_buckets])
        name = names[int(rng.choice(len(names), p=w / w.sum()))]
        if name == "clean":
            return name, None
        lo, hi = a.snr_buckets[name]
        return name, float(rng.uniform(lo, hi))

    def reverb(self, x: np.ndarray, rng: np.random.Generator, meta: dict[str, Any]) -> np.ndarray:
        rirs = self.pools.rir(self.pool) if self.pools else None
        if not rirs or rng.random() >= self.spec.p_reverb:
            return x
        item, rir = rirs.sample(rng, self.sr)
        meta.update(
            reverb=True, rir_source=item.source, rir_file=item.name, rir_category=item.category
        )
        return apply_rir(x, rir)

    def _noise(self, rng: np.random.Generator) -> tuple[Any, np.ndarray] | None:
        pool = self.pools.noise(self.pool) if self.pools else None
        return pool.sample(rng, self.sr) if pool else None

    def scene(
        self,
        canvas: np.ndarray,
        speech_ref: np.ndarray | None,
        rng: np.random.Generator,
        meta: dict[str, Any],
        noise_only: bool = False,
        allow_talker: bool = True,
    ) -> np.ndarray:
        """Add competing talker + noise to a canvas. SNR/SIR are relative to speech_ref."""
        a = self.spec
        ref = (
            active_rms(speech_ref, self.sr) if speech_ref is not None and len(speech_ref) else None
        )
        out = canvas.astype(np.float32).copy()

        talkers = self.pools.speech(self.pool) if self.pools else None
        if allow_talker and ref is not None and talkers and rng.random() < a.p_party:
            return self._party(out, ref, talkers, rng, meta)
        if allow_talker and ref is not None and talkers and rng.random() < a.p_interferer:
            item, talk, _ = talkers.sample_segment(rng, self.sr, len(out) / self.sr + 0.5)
            talk = fit_length(talk, len(out), rng)
            if rng.random() < 0.5:  # e.g. TV in another part of the room
                talk = self.reverb(talk, rng, {})[: len(out)]
            sir = float(rng.uniform(*a.sir_db))
            out = out + _db_gain(talk, ref, sir)
            meta.update(interferer=item.source, sir_db=sir)

        if noise_only:
            bucket, snr = "noise_only", None
        else:
            bucket, snr = self.noise_bucket(rng)
        if bucket != "clean":
            got = self._noise(rng)
            if got is None:
                if noise_only:
                    raise RuntimeError(f"no {self.pool} noise pool for noise-only samples")
                bucket, snr = "clean", None
            else:
                item, nz = got
                nz = fit_length(nz, len(out), rng)
                if ref is not None and snr is not None:
                    nz = _db_gain(nz, ref, snr)
                out = out + nz
                meta.update(
                    noise_source=item.source,
                    noise_type=item.category.split("/")[0],
                    noise_category=item.category,
                    noise_file=item.name,
                )
                if rng.random() < a.p_second_noise:  # layered scene: e.g. traffic + music
                    got2 = self._noise(rng)
                    if got2 is not None:
                        item2, nz2 = got2
                        nz2 = fit_length(nz2, len(out), rng)
                        lvl = float(np.sqrt(np.mean(nz**2)) + 1e-9)
                        out = out + _db_gain(nz2, lvl, float(rng.uniform(0, 10)))
                        meta["noise2_category"] = item2.category
        meta.update(noise_bucket=bucket, snr_db=snr if snr is not None else np.nan)
        return out

    def _party(
        self,
        out: np.ndarray,
        ref: float,
        talkers: Any,
        rng: np.random.Generator,
        meta: dict[str, Any],
    ) -> np.ndarray:
        """Several overlapping talkers close by plus crowd babble: a kitchen, a party, an office.
        Each talker is `party_sir_db` below the target speech; the babble is `party_babble_snr_db`."""
        a = self.spec
        n = int(rng.integers(a.party_talkers[0], a.party_talkers[1] + 1))
        sirs, srcs = [], []
        for _ in range(n):
            item, talk, _ = talkers.sample_segment(rng, self.sr, len(out) / self.sr + 0.5)
            talk = fit_length(talk, len(out), rng)
            if rng.random() < 0.5:
                talk = self.reverb(talk, rng, {})[: len(out)]
            sir = float(rng.uniform(*a.party_sir_db))
            out = out + _db_gain(talk, ref, sir)
            sirs.append(sir)
            srcs.append(item.source)
        snr = None
        pool = self.pools.noise(self.pool) if self.pools else None
        got = pool.sample_type(rng, "babble", self.sr) if pool else None
        if got is not None:
            item, nz = got
            snr = float(rng.uniform(*a.party_babble_snr_db))
            out = out + _db_gain(fit_length(nz, len(out), rng), ref, snr)
            meta.update(
                noise_source=item.source,
                noise_type=item.category.split("/")[0],
                noise_category=item.category,
                noise_file=item.name,
            )
        meta.update(
            party=True,
            n_talkers=n,
            interferer="+".join(sorted(set(srcs))),
            sir_db=min(sirs),  # the loudest competing talker
            noise_bucket="party",
            snr_db=snr if snr is not None else np.nan,
        )
        return out

    def device(self, x: np.ndarray, rng: np.random.Generator, meta: dict[str, Any]) -> np.ndarray:
        """Microphone / capture / transmission chain."""
        a = self.spec
        out = x
        if rng.random() < a.p_eq:
            out, desc = mic_eq(out, self.sr, rng)
            meta["eq"] = desc
        if rng.random() < a.p_bandlimit:
            kind = str(rng.choice(list(BANDLIMITS)))
            out = bandlimit(out, self.sr, kind)
            meta["bandlimit"] = kind
        level = float(rng.uniform(*a.level_dbfs))
        out = set_peak(out, level)
        meta["level_dbfs"] = level
        if rng.random() < a.p_clip:  # overdriven input: push past full scale, hard clip
            drive = float(rng.uniform(3, 12))
            out = np.clip(out * 10 ** (drive / 20) / (10 ** (level / 20)), -1, 1).astype(np.float32)
            meta.update(clipping=True, level_dbfs=0.0)
        if rng.random() < a.p_codec:
            codec = str(rng.choice(CODECS))
            out = apply_codec(out, self.sr, codec)
            meta["codec"] = codec
        return out

    def describe(self, meta: dict[str, Any]) -> str:
        tags = []
        if meta.get("reverb"):
            tags.append("reverb")
        if meta.get("party"):
            tags.append("party")
        elif meta.get("interferer"):
            tags.append("talker")
        if meta.get("noise_bucket") not in (None, "clean", "noise_only"):
            tags.append("noise")
        if meta.get("noise2_category"):
            tags.append("noise2")
        if meta.get("eq") not in (None, "", "none"):
            tags.append("eq")
        if meta.get("bandlimit") not in (None, "none"):
            tags.append("bandlimit")
        if meta.get("clipping"):
            tags.append("clip")
        if meta.get("codec") not in (None, "", "none"):
            tags.append("codec")
        return "+".join(tags) or "none"

    def augment_clip(
        self, audio: np.ndarray, rng: np.random.Generator, meta: dict[str, Any] | None = None
    ) -> np.ndarray:
        """Full chain on an already-placed clip (on-the-fly training augmentation)."""
        meta = {} if meta is None else meta
        wet = self.reverb(audio, rng, meta)[: len(audio)]
        out = self.scene(wet, wet if np.abs(wet).max() > 1e-4 else None, rng, meta)
        out = self.device(out, rng, meta)
        meta["augmentations"] = self.describe(meta)
        return out
