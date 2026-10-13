"""Signal processing: resampling, trimming, reverb, noise mixing at a target SNR, band-limiting."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import numpy as np
from scipy import signal


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    g = gcd(sr_in, sr_out)
    return signal.resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)


def to_mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim > 1 else audio


def trim_silence(
    audio: np.ndarray,
    sr: int,
    thresh_db: float = -40.0,
    pad_s: float = 0.03,
    min_run: int = 3,
) -> np.ndarray:
    """Trim leading/trailing silence relative to the clip's peak frame energy."""
    frame = int(0.01 * sr)
    n = len(audio) // frame
    if n < 3:
        return audio
    e = (audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    e_db = 10 * np.log10(e + 1e-12)
    on = (e_db > e_db.max() + thresh_db).astype(np.int32)
    if min_run > 1 and n >= min_run:
        run = np.convolve(on, np.ones(min_run, dtype=np.int32), mode="valid") == min_run
        starts = np.flatnonzero(run)
        if len(starts):
            active = np.array([starts[0], starts[-1] + min_run - 1])
        else:
            active = np.flatnonzero(on)
    else:
        active = np.flatnonzero(on)
    if len(active) == 0:
        return audio
    pad = int(pad_s * sr)
    a = max(0, active[0] * frame - pad)
    b = min(len(audio), (active[-1] + 1) * frame + pad)
    return audio[a:b]


def active_rms(audio: np.ndarray, sr: int, thresh_db: float = -35.0) -> float:
    """RMS over frames that are within thresh_db of the loudest frame (speech-active power)."""
    frame = int(0.02 * sr)
    n = len(audio) // frame
    if n == 0:
        return float(np.sqrt(np.mean(audio**2) + 1e-12))
    e = (audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    e_db = 10 * np.log10(e + 1e-12)
    sel = e[e_db > e_db.max() + thresh_db]
    return float(np.sqrt(sel.mean() + 1e-12))


def apply_rir(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolve with a room impulse response, keeping the direct path aligned at t=0."""
    rir = rir.astype(np.float32)
    peak = int(np.argmax(np.abs(rir)))
    rir = rir[peak:]
    rir = rir / (np.sqrt(np.sum(rir**2)) + 1e-9)
    out = signal.fftconvolve(audio, rir)[: len(audio) + len(rir) // 4]
    return (out * (active_rms(audio, 16000) / (active_rms(out, 16000) + 1e-9))).astype(np.float32)


def fit_length(noise: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Random crop, or loop with short crossfades, to exactly n samples."""
    if len(noise) >= n:
        start = int(rng.integers(0, len(noise) - n + 1))
        return noise[start : start + n]
    reps = []
    total = 0
    while total < n:
        reps.append(noise)
        total += len(noise)
    out = np.concatenate(reps)
    start = int(rng.integers(0, len(noise)))
    out = np.concatenate([out, noise])[start : start + n]
    return out


BANDLIMITS = {
    "telephone": (300.0, 3400.0),
    "lowpass_4k": (None, 4000.0),
    "lowpass_6k": (None, 6000.0),
    "highpass_200": (200.0, None),
}


def bandlimit(audio: np.ndarray, sr: int, kind: str) -> np.ndarray:
    lo, hi = BANDLIMITS[kind]
    if lo and hi:
        sos = signal.butter(6, [lo, hi], btype="bandpass", fs=sr, output="sos")
    elif hi:
        sos = signal.butter(8, hi, btype="lowpass", fs=sr, output="sos")
    else:
        sos = signal.butter(4, lo, btype="highpass", fs=sr, output="sos")
    return signal.sosfilt(sos, audio).astype(np.float32)


def set_peak(audio: np.ndarray, dbfs: float) -> np.ndarray:
    peak = np.abs(audio).max() + 1e-9
    return (audio * (10 ** (dbfs / 20) / peak)).astype(np.float32)


@dataclass
class MixResult:
    audio: np.ndarray
    onset_s: float | None
    offset_s: float | None


def place_and_mix(
    speech: np.ndarray | None,
    noise: np.ndarray | None,
    snr_db: float | None,
    sr: int,
    lead_s: float,
    tail_s: float,
    total_s: float | None = None,
    rng: np.random.Generator | None = None,
) -> MixResult:
    """Place speech on a canvas (lead + speech + tail) and mix in noise at snr_db."""
    rng = rng or np.random.default_rng()
    if speech is None:
        n = int(total_s * sr)
        canvas = np.zeros(n, dtype=np.float32)
        onset = offset = None
    else:
        lead, tail = int(lead_s * sr), int(tail_s * sr)
        canvas = np.zeros(lead + len(speech) + tail, dtype=np.float32)
        canvas[lead : lead + len(speech)] = speech
        onset, offset = lead / sr, (lead + len(speech)) / sr

    if noise is not None:
        nz = fit_length(noise.astype(np.float32), len(canvas), rng)
        if speech is not None and snr_db is not None:
            s_rms = active_rms(speech, sr)
            n_rms = float(np.sqrt(np.mean(nz**2)) + 1e-9)
            nz = nz * (s_rms / (n_rms * 10 ** (snr_db / 20)))
        canvas = canvas + nz
    return MixResult(canvas.astype(np.float32), onset, offset)
