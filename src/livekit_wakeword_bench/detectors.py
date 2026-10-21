"""Model-agnostic detector interface + adapters.

The benchmark hands each detector a *whole clip*. How the model consumes it (chunk size,
ring buffers, streaming state, VAD gating, ...) is entirely up to the detector. It returns a
`Prediction`: a score timeline, where `times[i]` is the position in the clip (seconds) at
which `scores[i]` was produced, i.e. how much audio the model had heard. That timeline is
what recall/FPPH sweeps and detection latency are computed from.

Models that only emit trigger events can return `Prediction.from_events([...])`.

Detector spec strings (CLI `--detector`, or `load_detector()`):

    livekit:path/to/hey_livekit.onnx          livekit-wakeword ONNX classifier
    openwakeword:hey_jarvis                   openWakeWord pretrained name or model path
    my_pkg.my_module:make_detector            any factory returning a Detector
    my_pkg.my_module:make_detector?key=value  ... with string kwargs
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qsl

import numpy as np


@dataclass
class Prediction:
    times: np.ndarray  # seconds into the clip at which each score was emitted (ascending)
    scores: np.ndarray  # confidence in [0, 1]

    @classmethod
    def from_events(
        cls, times: Sequence[float], confidence: Sequence[float] | None = None
    ) -> Prediction:
        t = np.asarray(times, dtype=np.float32)
        c = np.ones_like(t) if confidence is None else np.asarray(confidence, dtype=np.float32)
        return cls(t, c)


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, audio: np.ndarray, sample_rate: int) -> Prediction:
        """Run on a full clip (float32 mono in [-1, 1]); return the score timeline."""
        ...


# ---- adapters -------------------------------------------------------------------------------
def _resample(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return audio
    from .mix import resample

    return resample(audio, sr, target)


class SlidingWindowDetector:
    """Simulate a mic stream for a stateless `score_fn(window) -> float` model."""

    def __init__(
        self,
        score_fn: Callable[[np.ndarray], float],
        name: str,
        sample_rate: int = 16000,
        window_s: float = 2.0,
        hop_s: float = 0.08,
    ) -> None:
        self.score_fn = score_fn
        self.name = name
        self.sample_rate = sample_rate
        self.window = int(round(window_s * sample_rate))
        self.hop = int(round(hop_s * sample_rate))

    def detect(self, audio: np.ndarray, sample_rate: int) -> Prediction:
        x = _resample(audio.astype(np.float32), sample_rate, self.sample_rate)
        x = np.concatenate([np.zeros(self.window, dtype=np.float32), x])
        ends = np.arange(self.window + self.hop, len(x) + 1, self.hop)
        scores = np.array([self.score_fn(x[e - self.window : e]) for e in ends], dtype=np.float32)
        return Prediction((ends - self.window) / self.sample_rate, scores)


class LiveKitStreamingDetector:
    """livekit-wakeword v2 streaming model: `WakeWordModel.predict` on consecutive 80 ms chunks,
    exactly like `WakeWordListener` (state reset at the start of each clip)."""

    def __init__(self, model: Any, key: str, name: str | None = None) -> None:
        self._model = model
        self._key = key
        # streaming call: `predict()` in releases that also have `predict_clip()`, `process()` before
        self._stream = model.predict if hasattr(model, "predict_clip") else model.process
        self.name = name or f"livekit:{key}"

    def detect(self, audio: np.ndarray, sample_rate: int) -> Prediction:
        x = _resample(audio.astype(np.float32), sample_rate, 16000)
        self._model.reset()
        n = 1280
        times, scores = [], []
        for i in range(0, len(x) - n + 1, n):
            scores.append(float(self._stream(x[i : i + n])[self._key]))
            times.append((i + n) / 16000)
        return Prediction(np.array(times, np.float32), np.array(scores, np.float32))


def livekit_detector(
    model_path: str, name: str | None = None, **kw: Any
) -> SlidingWindowDetector | LiveKitStreamingDetector:
    """livekit-wakeword ONNX model via its public API, fed the same way `WakeWordListener` does:
    v2 streaming models streamed in 80 ms chunks, v1 classifiers scored on a 2 s window every
    80 ms."""
    from pathlib import Path

    from livekit.wakeword import WakeWordModel

    model = WakeWordModel(models=[model_path])
    key = Path(model_path).stem
    if getattr(model, "is_streaming", False):
        return LiveKitStreamingDetector(model, key, name)

    clip = getattr(model, "predict_clip", None) or model.predict  # whole-window scoring

    def score(window: np.ndarray) -> float:
        return float(clip(window)[key])

    return SlidingWindowDetector(
        score,
        name=name or f"livekit:{key}",
        window_s=float(kw.get("window_s", 2.0)),
        hop_s=float(kw.get("hop_s", 0.08)),
    )


class OpenWakeWordDetector:
    """openWakeWord (pip install openwakeword): natively streaming, 80 ms frames."""

    def __init__(self, model: str, name: str | None = None, **kw: Any) -> None:
        from openwakeword.model import Model

        self._model = Model(
            wakeword_models=[model], inference_framework=kw.get("framework", "onnx")
        )
        self._key = next(iter(self._model.models))
        self.name = name or f"openwakeword:{self._key}"

    def detect(self, audio: np.ndarray, sample_rate: int) -> Prediction:
        x = _resample(audio, sample_rate, 16000)
        pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        self._model.reset()
        n = 1280
        times, scores = [], []
        for i in range(0, len(pcm) - n + 1, n):
            scores.append(float(self._model.predict(pcm[i : i + n])[self._key]))
            times.append((i + n) / 16000)
        return Prediction(np.array(times, np.float32), np.array(scores, np.float32))


_BUILTIN: dict[str, Callable[..., Any]] = {
    "livekit": livekit_detector,
    "openwakeword": OpenWakeWordDetector,
}


def load_detector(spec: str) -> Detector:
    """Build a detector from a spec string (see module docstring)."""
    spec, _, query = spec.partition("?")
    kwargs = dict(parse_qsl(query))
    kind, _, arg = spec.partition(":")
    if kind in _BUILTIN:
        det = _BUILTIN[kind](arg, **kwargs)
    else:
        module, _, attr = spec.rpartition(":")
        if not module:
            raise ValueError(f"bad detector spec {spec!r}")
        det = getattr(importlib.import_module(module), attr)(**kwargs)
    if not isinstance(det, Detector):
        raise TypeError(f"{spec!r} did not produce a Detector (needs .name and .detect())")
    return det
