"""Async TTS through LiveKit Inference, with an on-disk cache keyed by request params."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TTSRequest:
    model: str  # "provider/model"
    voice: str
    text: str
    speed: float = 1.0
    language: str = "en"
    extra: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def provider(self) -> str:
        return self.model.split("/")[0]

    def key(self) -> str:
        blob = json.dumps(
            {**asdict(self), "speed": round(self.speed, 3)}, sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha1(blob.encode()).hexdigest()[:20]


def speed_kwargs(model: str, speed: float) -> dict[str, Any]:
    """Map a generic speaking-rate multiplier onto each provider's own option."""
    provider = model.split("/")[0]
    if abs(speed - 1.0) < 1e-3:
        return {}
    if provider == "cartesia":
        return {"speed": float(np.clip(speed, 0.6, 1.5))}
    if provider == "rime" and "mist" in model:
        return {"speed_alpha": float(np.clip(1.0 / speed, 0.5, 2.0))}  # >1 is slower
    if provider == "inworld":
        return {"speaking_rate": float(np.clip(speed, 0.55, 1.5))}
    if provider in ("xai", "fishaudio"):
        return {"speed": float(np.clip(speed, 0.6, 1.5))}
    return {}  # deepgram: no rate control


class Synthesizer:
    """Concurrent, cached, retrying synthesis. Returns float32 mono at the TTS sample rate."""

    def __init__(
        self,
        cache_dir: Path,
        concurrency: int = 16,
        sample_rate: int = 24000,
        retries: int = 3,
        timeout_s: float = 30.0,
    ) -> None:
        self.timeout_s = timeout_s
        self.cache_dir = Path(cache_dir) / "tts"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.retries = retries
        self._sem = asyncio.Semaphore(concurrency)
        self._http: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> Synthesizer:
        self._http = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._http:
            await self._http.close()

    def cache_path(self, req: TTSRequest) -> Path:
        k = req.key()
        return self.cache_dir / k[:2] / f"{k}.flac"

    def _make_tts(self, req: TTSRequest) -> Any:
        """One TTS instance per request, closed right after."""
        from livekit.agents import APIConnectOptions, inference

        extra = {**speed_kwargs(req.model, req.speed), **req.extra}
        return inference.TTS(
            model=req.model,
            voice=req.voice,
            language=req.language,
            sample_rate=self.sample_rate,
            http_session=self._http,
            extra_kwargs=extra,
            conn_options=APIConnectOptions(max_retry=0, timeout=20.0),
        )

    async def ensure(self, req: TTSRequest) -> bool:
        """Synthesize into the cache without keeping audio in memory. True on success."""
        return await self.synthesize(req, load=False) is not None

    def load(self, req: TTSRequest) -> np.ndarray:
        audio, _ = sf.read(self.cache_path(req), dtype="float32")
        return audio

    async def synthesize(self, req: TTSRequest, load: bool = True) -> np.ndarray | None:
        path = self.cache_path(req)
        if path.exists():
            if not load:
                return np.zeros(0, dtype=np.float32)
            audio, _ = sf.read(path, dtype="float32")
            return audio
        async with self._sem:
            for attempt in range(self.retries):
                try:
                    tts = self._make_tts(req)

                    async def collect() -> list[np.ndarray]:
                        return [
                            np.frombuffer(ev.frame.data, dtype=np.int16)
                            async for ev in tts.synthesize(req.text)
                        ]

                    try:
                        chunks = await asyncio.wait_for(collect(), timeout=self.timeout_s)
                    finally:
                        try:
                            await tts.aclose()
                        except Exception:  # noqa: BLE001
                            pass
                    if not chunks:
                        raise RuntimeError("empty audio")
                    audio = np.concatenate(chunks).astype(np.float32) / 32768.0
                    if len(audio) < 0.15 * self.sample_rate or np.abs(audio).max() < 1e-3:
                        raise RuntimeError(f"degenerate audio ({len(audio)} samples)")
                    path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(path, audio, self.sample_rate)
                    return audio
                except (TimeoutError, Exception) as e:  # noqa: BLE001
                    logger.warning(
                        "TTS failed (%s/%s %r, attempt %d): %s",
                        req.model,
                        req.voice,
                        req.text[:40],
                        attempt + 1,
                        e,
                    )
                    if "BAD_REQUEST" in str(e) or "not found" in str(e).lower():
                        break  # non-retryable (bad voice id, unsupported option, ...)
                    await asyncio.sleep(1.5 * (attempt + 1))
        return None
