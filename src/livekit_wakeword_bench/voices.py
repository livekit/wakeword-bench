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
    languages: frozenset[str] = frozenset({"en"})

    @property
    def provider(self) -> str:
        return self.model.split("/")[0]

    @property
    def key(self) -> str:
        return f"{self.model}:{self.id}"


# Languages each bundled model speaks through LiveKit Inference, as language codes without a
# region ("pt" for pt-BR): https://docs.livekit.io/agents/models/tts/ (checked 2026-11-03).
# A voice can narrow its model's set with "languages" in voices.json: Rime speakers each speak
# their own languages (https://users.rime.ai/data/voices/all-v2.json).
MODEL_LANGUAGES: dict[str, frozenset[str]] = {
    "cartesia/sonic-3": frozenset(
        "en de es fr ja pt zh hi ko it nl pl ru sv tr tl bg ro ar cs el fi hr ms sk da ta uk hu "
        "no vi bn th he ka id te gu kn ml mr pa".split()
    ),
    "inworld/inworld-tts-2": frozenset(
        "en zh ja ko ru it es pt fr de pl nl hi he ar af sq am hy as az eu be bn bg my yue ca "
        "ceb hr cs da yih arz et fil fi gl gbm ka el gu afb ht acw hu is id jv kn kk kok lo lv "
        "ayl lt lb mk mai mg ms ml mr mn ars ne uzn nb nn or acx ps fa pms pa ro sr sd si sk sl "
        "sw sv ta te th aeb tr uk ur uz vi vro cy".split()
    ),
    "xai/tts-1": frozenset("en ar bn zh fr de hi id it ja ko pt ru es tr vi".split()),
    "rime/mistv3": frozenset("en es fr de hi".split()),
}
# the same language under the codes different providers use
_ALIASES = {"no": "nb", "nb": "no", "tl": "fil", "fil": "tl", "iw": "he", "he": "iw"}


def base_language(language: str) -> str:
    """ "pt-BR" / "pt_br" -> "pt"."""
    return language.strip().lower().replace("_", "-").split("-")[0]


def voices_for_language(voices: list[Voice], language: str) -> list[Voice]:
    """The voices that speak `language`: from every model that supports it, so a phrase gets as
    many models (and voices) as its language allows. Models missing from `MODEL_LANGUAGES` are
    treated as English-only."""
    code = base_language(language)
    codes = {code, _ALIASES.get(code, code)}
    return [v for v in voices if codes & v.languages]


def load_voices(path: str | Path | None = None, models: list[str] | None = None) -> list[Voice]:
    """Load the voice catalog (bundled voices.json unless `path` is given)."""
    if path is None:
        text = resources.files(__package__).joinpath("voices.json").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
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
                    languages=frozenset(
                        v.get("languages") or MODEL_LANGUAGES.get(model, frozenset({"en"}))
                    ),
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
