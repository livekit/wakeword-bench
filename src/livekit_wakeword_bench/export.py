"""Export a generated dataset to training-pipeline layouts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .loader import WakeWordData

_LKWW_LAYOUT = {
    "positive_train": ("train", 1),
    "negative_train": ("train", 0),
    "positive_test": ("validation", 1),
    "negative_test": ("validation", 0),
}


def _crop_utterance(audio: np.ndarray, sr: int, meta: dict) -> np.ndarray:
    on, off = meta.get("onset_s"), meta.get("offset_s")
    if on is None or off is None or on != on or off != off:
        return audio
    return audio[int(on * sr) : int(off * sr)]


def export_livekit_wakeword(
    source: str | Path, out_dir: str | Path, model_name: str, include_speech: bool = True
) -> dict[str, int]:
    """Write `<out_dir>/<model_name>/{positive,negative}_{train,test}/clip_NNNNNN.wav`."""
    root = Path(out_dir) / model_name
    counts: dict[str, int] = {}
    for dirname, (split, label) in _LKWW_LAYOUT.items():
        try:
            data = WakeWordData(source, split=split)
        except FileNotFoundError:
            continue
        cats = ["positive"] if label else ["adversarial"] + (["speech"] if include_speech else [])
        data = data.filter(category=cats)
        d = root / dirname
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for s in tqdm(data, total=len(data), desc=dirname):
            sf.write(
                d / f"clip_{n:06d}.wav",
                _crop_utterance(s.audio, s.sample_rate, s.meta),
                s.sample_rate,
            )
            n += 1
        counts[dirname] = n
    return counts


def export_wav_folder(source: str | Path, out_dir: str | Path, split: str) -> Path:
    """Generic export: `<out_dir>/<split>/<id>.wav` + `<split>/metadata.csv` (all columns)."""
    data = WakeWordData(source, split=split)
    d = Path(out_dir) / split
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        w: csv.DictWriter | None = None
        for s in tqdm(data, total=len(data), desc=f"export {split}"):
            sf.write(d / f"{s.id}.wav", s.audio, s.sample_rate)
            row = {"file": f"{s.id}.wav", **s.meta}
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
            w.writerow(row)
    return d
