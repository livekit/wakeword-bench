"""Voice catalog for LiveKit Inference TTS + deterministic held-out voice split."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

_ACCENT_MAP = {
    "american (en-us)": "US",
    "british (en-gb)": "UK",
    "gb": "UK",
    "australian (en-au)": "AU",
    "indian": "IN",
    "indian (hinglish)": "IN",
    "filipino (en-ph)": "PH",
    "es-mx (spanish)": "MX",
    "mexican": "MX",
    "middle eastern": "ME",
    "german": "DE",
    "french": "FR",
}


def normalize_accent(raw: str | None) -> str:
    if not raw:
        return "unknown"
    r = raw.strip()
    if r.lower() in _ACCENT_MAP:
        return _ACCENT_MAP[r.lower()]
    if r.startswith("US"):
        return "US"
    return r.upper() if len(r) <= 3 else r


@dataclass(frozen=True)
class Voice:
    model: str
    id: str
    name: str
    gender: str
    accent: str
    accent_detail: str

    @property
    def provider(self) -> str:
        return self.model.split("/")[0]

    @property
    def key(self) -> str:
        return f"{self.model}:{self.id}"


def load_voices(path: str | Path | None = None, models: list[str] | None = None) -> list[Voice]:
    """Load the voice catalog (bundled voices.json unless `path` is given)."""
    if path is None:
        text = resources.files(__package__).joinpath("voices.json").read_text()
    else:
        text = Path(path).read_text()
    data: dict[str, list[dict]] = json.loads(text)
    out: list[Voice] = []
    for model, voices in data.items():
        if models and model not in models:
            continue
        for v in voices:
            out.append(
                Voice(
                    model=model,
                    id=v["id"],
                    name=v.get("name") or v["id"],
                    gender=v.get("gender") or "unknown",
                    accent=normalize_accent(v.get("accent")),
                    accent_detail=v.get("accent_detail") or v.get("accent") or "",
                )
            )
    return out


def _h(s: str, seed: int) -> int:
    return int(hashlib.sha1(f"{seed}:{s}".encode()).hexdigest(), 16)


def split_voices(
    voices: list[Voice], test_fraction: float, seed: int = 0
) -> tuple[list[Voice], list[Voice]]:
    """Deterministically partition voices into (train_voices, test_voices), per model."""
    train: list[Voice] = []
    test: list[Voice] = []
    by_model: dict[str, list[Voice]] = {}
    for v in voices:
        by_model.setdefault(v.model, []).append(v)
    for vs in by_model.values():
        ranked = sorted(vs, key=lambda v: _h(v.key, seed))
        n_test = max(1, math.ceil(test_fraction * len(vs))) if len(vs) > 1 else 0
        test.extend(ranked[:n_test])
        train.extend(ranked[n_test:])
    return train, test
