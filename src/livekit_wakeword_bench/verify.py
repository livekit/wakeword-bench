"""Dataset QA: measure the audio and check it agrees with its labels."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .loader import WakeWordData

SR = 16000


def _frame_db(x: np.ndarray, sr: int = SR, hop_s: float = 0.01) -> np.ndarray:
    h = int(hop_s * sr)
    n = len(x) // h
    if n == 0:
        return np.zeros(0)
    return 10 * np.log10((x[: n * h].reshape(n, h) ** 2).mean(axis=1) + 1e-12)


def _hf_ratio(x: np.ndarray, cutoff: float, sr: int = SR) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float(spec[f > cutoff].sum() / (spec.sum() + 1e-12))


def measure_snr(audio: np.ndarray, meta: dict[str, Any], sr: int = SR) -> float | None:
    """Blind SNR estimate: noise power from the lead-in (before the utterance); speech power from noise-subtracted 20 ms frames within 35 dB of the loudest (the mixer's definition)."""
    on, off = meta["onset_s"], meta["offset_s"]
    if on != on or on < 0.25:
        return None
    noise = audio[: int((on - 0.05) * sr)]
    sp = audio[int(on * sr) : int(off * sr)]
    if len(noise) < sr // 5 or len(sp) < sr // 5:
        return None
    pn = float(np.mean(noise**2))
    h = int(0.02 * sr)
    fr = (sp[: len(sp) // h * h].reshape(-1, h) ** 2).mean(axis=1) - pn  # noise-subtracted
    fr = np.clip(fr, 1e-12, None)
    db = 10 * np.log10(fr)
    ps = float(fr[db > db.max() - 35].mean())
    if ps <= 0 or pn <= 0:
        return None
    return 10 * math.log10(ps / pn)


def verify(
    source: str | Path,
    split: str = "test",
    limit: int | None = None,
) -> dict[str, Any]:
    data = WakeWordData(source, split=split)
    if limit:
        data = data.shuffle(0).take(limit)
    report: dict[str, Any] = {"split": split, "n": len(data)}
    bad: dict[str, list[str]] = defaultdict(list)
    snr_pairs: list[tuple[float, float, str]] = []
    level_err: list[float] = []
    onset_err: list[float] = []
    hf: dict[str, list[float]] = defaultdict(list)

    for s in data:
        a, m = s.audio, s.meta
        if not np.isfinite(a).all():
            bad["integrity:nonfinite"].append(s.id)
            continue
        if abs(len(a) / s.sample_rate - m["duration_s"]) > 0.01:
            bad["integrity:duration"].append(s.id)
        if (m["category"] == "positive") != bool(m["label"]):
            bad["integrity:label"].append(s.id)
            peak = 20 * math.log10(np.abs(a).max() + 1e-12)
            level_err.append(peak - m["level_dbfs"])
            if abs(peak - m["level_dbfs"]) > 0.5:
                bad["level"].append(s.id)
        clean_chain = (
            not m.get("interferer")
            and not m.get("noise2_category")
            and not m.get("clipping")
            and m.get("codec", "none") == "none"
        )
        if (
            m["snr_db"] == m["snr_db"]
            and clean_chain
            and not m.get("eq")
            and m.get("bandlimit", "none") == "none"
            and m["category"] != "real_speech"
        ):
            est = measure_snr(a, m)
            if est is not None:
                snr_pairs.append((m["snr_db"], est, s.id))
        # clean dry clips: silent outside [onset, offset] (codecs ring, so skip them)
        if (
            m["noise_bucket"] == "clean"
            and m["category"] in ("positive", "adversarial", "speech")  # TTS utterances
            and m.get("codec", "none") == "none"
            and not m.get("reverb")
            and not m.get("interferer")
            and m["onset_s"] == m["onset_s"]
        ):
            db = _frame_db(a)
            if len(db):
                ref = db.max()
                i_on, i_off = int(m["onset_s"] * 100), int(m["offset_s"] * 100)
                before = db[: max(0, i_on - 1)]
                after = db[i_off + 1 :]
                head = db[i_on : i_on + 25]
                outside = max(
                    before.max() if len(before) else -200, after.max() if len(after) else -200
                )
                onset_err.append(float(outside - ref))
                if outside > ref - 50:
                    bad["timing:sound_outside_labels"].append(s.id)
                if len(head) and head.max() < ref - 45:
                    bad["timing:no_sound_after_onset"].append(s.id)
        bl, codec = m.get("bandlimit", "none"), m.get("codec", "none")
        if bl in ("lowpass_4k", "telephone") or codec == "mulaw_8khz":
            hf["limited(>4.5k)"].append(_hf_ratio(a, 4500))
        elif bl == "none" and codec == "none" and not m.get("eq"):
            hf["full(>4.5k)"].append(_hf_ratio(a, 4500))

    if split == "test":
        leak = _leakage(source)
        report["leakage"] = leak
        if any(leak.values()):
            bad["leakage"].append(str(leak))

    if snr_pairs:
        lab = np.array([p[0] for p in snr_pairs])
        est = np.array([p[1] for p in snr_pairs])
        err = est - lab
        report["snr"] = {
            "n": len(snr_pairs),
            "median_abs_err_db": round(float(np.median(np.abs(err))), 2),
            "p90_abs_err_db": round(float(np.percentile(np.abs(err), 90)), 2),
            "corr": round(float(np.corrcoef(lab, est)[0, 1]), 3),
        }
        report["snr"]["by_bucket"] = {}
        for name, (lo, hi) in [
            ("<0", (-99, 0)),
            ("0-10", (0, 10)),
            ("10-20", (10, 20)),
            (">=20", (20, 99)),
        ]:
            sel = (lab >= lo) & (lab < hi)
            if sel.any():
                report["snr"]["by_bucket"][name] = round(float(np.median(err[sel])), 2)
    if level_err:
        report["level"] = {
            "n": len(level_err),
            "max_abs_err_db": round(float(np.max(np.abs(level_err))), 3),
        }
    if onset_err:
        report["timing"] = {
            "n": len(onset_err),
            "max_level_outside_labels_db": round(float(np.max(onset_err)), 1),
        }
    if hf:
        report["bandlimit"] = {
            k: {"n": len(v), "median_hf_energy": float(np.median(v))} for k, v in hf.items()
        }

    report["failures"] = {k: {"count": len(v), "ids": v[:20]} for k, v in bad.items()}
    return report


def _leakage(source: str | Path) -> dict[str, int]:
    test = WakeWordData(source, "test").metadata()
    trains = []
    for sp in ("train", "validation"):
        try:
            trains.append(WakeWordData(source, sp).metadata())
        except FileNotFoundError:
            pass
    if not trains:
        return {}
    import pandas as pd

    tr = pd.concat(trains)
    out = {}
    tts = lambda df: set(zip(df[df.model != ""].model, df[df.model != ""].voice_id))  # noqa: E731
    out["shared_tts_voices"] = len(tts(tr) & tts(test))
    out["shared_noise_files"] = len((set(tr.noise_file) - {""}) & (set(test.noise_file) - {""}))
    out["shared_rir_files"] = len((set(tr.rir_file) - {""}) & (set(test.rir_file) - {""}))
    out["shared_noise_sources"] = len(
        (set(tr.noise_source) - {""}) & (set(test.noise_source) - {""})
    )
    return out
