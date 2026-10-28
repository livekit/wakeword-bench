"""Dev-only ASR backend (NOT part of the library): faster-whisper (CPU, cross-platform).
Transcripts are cached per (model, key) so each clip is transcribed once."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_W: dict[str, Any] = {}


def _init(model: str, threads: int) -> None:
    from faster_whisper import WhisperModel

    _W["model"] = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)


def _transcribe(job: tuple[str, str]) -> tuple[str, str]:
    key, path = job
    import soundfile as sf

    from livekit_wakeword_bench.mix import resample

    audio, sr = sf.read(path, dtype="float32")
    audio = resample(audio, sr, 16000)
    # pad so very short clips don't get dropped by the decoder
    audio = np.concatenate([np.zeros(4000, np.float32), audio, np.zeros(4000, np.float32)])
    segs, _ = _W["model"].transcribe(
        audio, language="en", beam_size=5, vad_filter=False, condition_on_previous_text=False
    )
    text = " ".join(seg.text for seg in segs).strip()
    return key, text.strip()


class ASRCache:
    def __init__(self, cache_dir: Path, model: str) -> None:
        self.path = Path(cache_dir) / "asr" / f"faster-whisper__{model.replace('/', '_')}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line:
                    r = json.loads(line)
                    self.data[r["key"]] = r["text"]

    def add(self, key: str, text: str) -> None:
        self.data[key] = text
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "text": text}) + "\n")


def transcribe_all(
    jobs: dict[str, Path],
    cache_dir: Path,
    model: str = "base.en",
    workers: int = 8,
) -> dict[str, str]:
    """Transcribe {key: audio_path}; cached. Returns {key: transcript}."""
    from tqdm import tqdm

    cache = ASRCache(cache_dir, model)
    todo = [(k, str(p)) for k, p in jobs.items() if k not in cache.data]
    if todo:
        logger.info("ASR on %d clips (faster-whisper %s, %d workers)", len(todo), model, workers)
        with ProcessPoolExecutor(max(1, workers), initializer=_init, initargs=(model, 2)) as ex:
            for k, t in tqdm(
                ex.map(_transcribe, todo, chunksize=16), total=len(todo), desc="asr", unit="clip"
            ):
                cache.add(k, t)
    return {k: cache.data[k] for k in jobs}
