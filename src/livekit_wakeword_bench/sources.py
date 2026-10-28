"""Background noise, room impulse responses and real speech pulled from Hugging Face."""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .mix import resample, to_mono

logger = logging.getLogger(__name__)
SR = 16000


@dataclass
class Item:
    source: str
    kind: str  # noise | rir | speech
    category: str  # noise taxonomy, room type, or corpus subset
    name: str  # relative path within the source dir
    duration_s: float
    speaker: str = ""
    transcript: str = ""
    group: str = ""  # recording/meeting/call id: consecutive items in a group are contiguous


@dataclass
class SourceSpec:
    name: str
    kind: str
    pool: str  # train | test
    fetch: Callable[[], Iterator[tuple[str, str, np.ndarray, int, dict[str, Any]]]]
    description: str = ""


def _hf_files(repo: str, patterns: list[str]) -> list[str]:
    import fnmatch

    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return sorted(f for f in files if any(fnmatch.fnmatch(f, p) for p in patterns))


def _hf_get(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset")


def _audio_files(
    repo: str, patterns: list[str], categorize: Callable[[str], str], limit: int | None = None
) -> Callable[[], Iterator[tuple[str, str, np.ndarray, int, dict[str, Any]]]]:
    def fetch() -> Iterator[tuple[str, str, np.ndarray, int, dict[str, Any]]]:
        from huggingface_hub import snapshot_download

        files = _hf_files(repo, patterns)[:limit]
        local = Path(
            snapshot_download(repo, repo_type="dataset", allow_patterns=files, max_workers=16)
        )
        for f in files:
            try:
                audio, sr = sf.read(local / f, dtype="float32", always_2d=False)
            except Exception as e:  # noqa: BLE001
                logger.warning("skip %s/%s: %s", repo, f, e)
                continue
            yield f, categorize(f), audio, sr, {}

    return fetch


def _parquet_rows(
    repo: str,
    patterns: list[str],
    audio_col: str,
    row_fn: Callable[[dict[str, Any]], tuple[str, str, dict[str, Any]] | None],
    max_files: int | None = None,
    max_rows: int | None = None,
) -> Callable[[], Iterator[tuple[str, str, np.ndarray, int, dict[str, Any]]]]:
    """Rows of parquet shards. row_fn(row) -> (name, category, extra) or None to skip."""

    def fetch() -> Iterator[tuple[str, str, np.ndarray, int, dict[str, Any]]]:
        import pyarrow.parquet as pq

        n = 0
        for f in _hf_files(repo, patterns)[:max_files]:
            pf = pq.ParquetFile(_hf_get(repo, f))
            for batch in pf.iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    got = row_fn(row)
                    if got is None:
                        continue
                    name, cat, extra = got
                    cell = row[audio_col]
                    data = cell["bytes"] if isinstance(cell, dict) else cell
                    try:
                        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("skip row %s: %s", name, e)
                        continue
                    yield name, cat, audio, sr, extra
                    n += 1
                    if max_rows and n >= max_rows:
                        return

    return fetch


def _musan_category(path: str) -> str:
    parts = path.split("/")
    return f"{parts[0]}/{parts[1]}" if len(parts) > 2 else parts[0]


def _mit_room(path: str) -> str:
    stem = Path(path).stem.lower()
    for k in (
        "bedroom",
        "office",
        "living",
        "bathroom",
        "kitchen",
        "car",
        "hall",
        "street",
        "outdoor",
        "restaurant",
        "stair",
        "classroom",
        "store",
        "lobby",
    ):
        if k in stem:
            return k
    return "other"


SOURCES: dict[str, SourceSpec] = {}


def register(spec: SourceSpec) -> None:
    SOURCES[spec.name] = spec


# noise categories are "<noise_type>/<detail>"

_DEMAND = {
    "DKITCHEN": "domestic/kitchen",
    "DLIVING": "domestic/living_room",
    "DWASHING": "domestic/washing_machine",
    "NFIELD": "nature/field",
    "NPARK": "nature/park",
    "NRIVER": "nature/river",
    "OHALLWAY": "office/hallway",
    "OMEETING": "office/meeting_room",
    "OOFFICE": "office/office",
    "PCAFETER": "babble/cafeteria",
    "PRESTO": "babble/restaurant",
    "PSTATION": "public/station",
    "SCAFE": "babble/cafe_terrace",
    "SPSQUARE": "public/square",
    "STRAFFIC": "street/traffic",
    "TBUS": "transport/bus",
    "TCAR": "transport/car",
    "TMETRO": "transport/metro",
}
_ESC50_GROUPS = ["animals", "nature", "human", "domestic", "street"]


def _demand_row(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    name = row["name"]  # e.g. SPSQUARE_16k/ch11_05.ogg -> keep one mic channel per scene
    if "/ch01_" not in name:
        return None
    env = name.split("_")[0]
    return name, _DEMAND.get(env, f"mixed/{env.lower()}"), {}


def _wham_row(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    return f"wham/{row['name']}", "babble/restaurant_bar", {}


def _esc50_row(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    label = int(row["label"])
    sound = str(row.get("sound") or label)
    clip = Path(row["audio"].get("path") or f"{label}_{_esc50_row.n}").stem  # type: ignore[attr-defined]
    _esc50_row.n = getattr(_esc50_row, "n", 0) + 1  # type: ignore[attr-defined]
    return f"esc50/{clip}", f"{_ESC50_GROUPS[label // 10]}/{sound}", {}


def _gtzan_row(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    return f"gtzan/{row['audio']['path']}", f"music/{row.get('genre', 'unknown')}", {}


def _dns_row(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]] | None:
    src = str(row.get("source_file", ""))
    if "freesound" in src.lower():  # Freesound overlaps ESC-50 (test); keep AudioSet only
        return None
    return f"dns/{Path(src).stem}_{row['audio'].get('path', '')}", "mixed/audioset", {}


def _slr28_row(row: dict[str, Any], keep_every: int = 25) -> tuple[str, str, dict[str, Any]] | None:
    path = row["path"]
    parts = path.split("/")
    if "real_rirs_isotropic_noises" in path:
        if "noise" in parts[-1].lower():
            return None
        db = parts[-1].split("_")[0]
        return path, f"real/{db}", {}
    if "simulated_rirs" in path:
        _slr28_row.n = getattr(_slr28_row, "n", 0) + 1  # type: ignore[attr-defined]
        if _slr28_row.n % keep_every:  # type: ignore[attr-defined]
            return None
        return path, f"simulated/{parts[2]}", {}
    return None  # pointsource_noises come from MUSAN


def _treble_row(row: dict[str, Any], keep_every: int = 6) -> tuple[str, str, dict[str, Any]] | None:
    _treble_row.n = getattr(_treble_row, "n", 0) + 1  # type: ignore[attr-defined]
    if _treble_row.n % keep_every:  # type: ignore[attr-defined]
        return None
    room = str(row.get("Room", "room")).replace(" ", "_")
    return f"treble10/{room}/{_treble_row.n:05d}", f"treble10/{room}", {}  # type: ignore[attr-defined]


def _ami_row(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    return (
        f"ami/{row['meeting_id']}/{row['audio_id']}",
        "meeting_far_field",
        {
            "speaker": row.get("speaker_id", ""),
            "transcript": row.get("text", ""),
            "group": row["meeting_id"],
        },
    )


def _earnings_row(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    return (
        f"earnings22/{row['file_id']}",
        f"earnings_call/{row.get('major_dialect_family') or 'unknown'}",
        {"speaker": row["file_id"], "transcript": "", "group": row["file_id"]},
    )


def _peoples_row(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    return f"peoples/{row['id']}", "mixed_web", {"speaker": "", "transcript": row.get("text", "")}


def _libri_row(row: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    return (
        f"librispeech/{row['id']}",
        "read_audiobook",
        {
            "speaker": str(row.get("speaker_id", "")),
            "transcript": row.get("text", ""),
            "group": str(row.get("chapter_id", "")),
        },
    )


# train / validation
register(
    SourceSpec(
        "musan_noise",
        "noise",
        "train",
        _audio_files(
            "FluidInference/musan", ["noise/**/*.wav"], lambda p: f"mixed/musan_{p.split('/')[1]}"
        ),
        "MUSAN noise (CC BY 4.0). What livekit-wakeword trains with.",
    )
)
register(
    SourceSpec(
        "dns_noise",
        "noise",
        "train",
        _parquet_rows(
            "humanify/DNS-Noise",
            ["noise/data_000005.parquet", "noise/data_000011.parquet"],
            "audio",
            _dns_row,
        ),
        "DNS Challenge noise, AudioSet part only (CC BY 4.0)",
    )
)
register(
    SourceSpec(
        "mit_rir",
        "rir",
        "train",
        _audio_files(
            "davidscripka/MIT_environmental_impulse_responses",
            ["16khz/*.wav"],
            lambda p: f"mit/{_mit_room(p)}",
        ),
        "MIT environmental impulse responses",
    )
)
register(
    SourceSpec(
        "librispeech",
        "speech",
        "train",
        _parquet_rows(
            "openslr/librispeech_asr", ["all/test.clean/0000.parquet"], "audio", _libri_row
        ),
        "LibriSpeech test-clean (CC BY 4.0), read speech for validation FPPH",
    )
)

# test (disjoint from train)
register(
    SourceSpec(
        "demand",
        "noise",
        "test",
        _parquet_rows("philgzl/demand", ["data/16k-*.parquet"], "audio", _demand_row),
        "DEMAND multichannel environmental noise, 1 channel per scene (CC BY 4.0)",
    )
)
register(
    SourceSpec(
        "wham",
        "noise",
        "test",
        _parquet_rows(
            "philgzl/wham",
            ["data/test-000-*.parquet", "data/test-001-*.parquet"],
            "audio",
            _wham_row,
        ),
        "WHAM! restaurant/bar babble, test split (CC BY-NC 4.0)",
    )
)
register(
    SourceSpec(
        "esc50",
        "noise",
        "test",
        _parquet_rows(
            "confit/esc50-parquet",
            ["fold1/test-*.parquet", "fold2/test-*.parquet"],
            "audio",
            _esc50_row,
        ),
        "ESC-50 environmental sounds, folds 1-2 (CC BY-NC)",
    )
)
register(
    SourceSpec(
        "gtzan",
        "noise",
        "test",
        _parquet_rows("confit/gtzan-parquet", ["data/validation-*.parquet"], "audio", _gtzan_row),
        "GTZAN music, 10 genres (research use)",
    )
)
register(
    SourceSpec(
        "slr28_rir",
        "rir",
        "test",
        _parquet_rows(
            "quinnlue/rirs",
            [
                "data/train-00000-*.parquet",
                "data/train-00003-*.parquet",
                "data/train-00007-*.parquet",
            ],
            "audio",
            _slr28_row,
        ),
        "OpenSLR 28: real RIRs (RWCP/REVERB/AIR) + simulated rooms (Apache 2.0)",
    )
)
register(
    SourceSpec(
        "treble10_rir",
        "rir",
        "test",
        _parquet_rows(
            "treble-technologies/Treble10-RIR", ["data/rir_mono-*.parquet"], "audio", _treble_row
        ),
        "Treble10 furnished-room RIRs (CC BY-NC-SA 4.0)",
    )
)
register(
    SourceSpec(
        "ami",
        "speech",
        "test",
        _parquet_rows(
            "edinburghcstr/ami",
            ["sdm/test-00000-*.parquet", "sdm/test-00001-*.parquet"],
            "audio",
            _ami_row,
        ),
        "AMI meeting corpus, far-field single mic (CC BY 4.0)",
    )
)
register(
    SourceSpec(
        "earnings22",
        "speech",
        "test",
        _parquet_rows(
            "distil-whisper/earnings22",
            ["full/test-00000-*.parquet"],
            "audio",
            _earnings_row,
            max_rows=12,
        ),
        "Earnings-22 accented earnings calls (CC BY-SA 4.0)",
    )
)
register(
    SourceSpec(
        "peoples_speech",
        "speech",
        "test",
        _parquet_rows(
            "MLCommons/peoples_speech", ["test/test-00000-*.parquet"], "audio", _peoples_row
        ),
        "People's Speech test shard (CC BY)",
    )
)


def source_dir(cache: Path, name: str) -> Path:
    return Path(cache) / "sources" / name


def prepare_source(name: str, cache: Path, force: bool = False) -> list[Item]:
    spec = SOURCES[name]
    d = source_dir(cache, name)
    manifest = d / "manifest.jsonl"
    if manifest.exists() and not force:
        return [
            Item(**json.loads(line))
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line
        ]
    d.mkdir(parents=True, exist_ok=True)
    items: list[Item] = []
    from tqdm import tqdm

    for rel, cat, audio, sr, extra in tqdm(spec.fetch(), desc=f"prepare {name}", unit="file"):
        audio = resample(to_mono(np.asarray(audio, dtype=np.float32)), sr, SR)
        if len(audio) < SR // 10 or not np.isfinite(audio).all():
            continue
        out_rel = str(Path(rel).with_suffix(".flac")).replace(":", "_")
        out = d / out_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        peak = np.abs(audio).max()
        if peak > 1.0:
            audio = audio / peak
        sf.write(out, audio, SR, subtype="PCM_16")
        items.append(
            Item(
                name,
                spec.kind,
                cat,
                out_rel,
                round(len(audio) / SR, 3),
                speaker=str(extra.get("speaker", "")),
                transcript=str(extra.get("transcript", "")),
            )
        )
    manifest.write_text("".join(json.dumps(asdict(i)) + "\n" for i in items), encoding="utf-8")
    return items


class AudioPool:
    """Hierarchical sampling: coarse type -> source -> fine category -> item, each uniform."""

    def __init__(self, cache: Path, items: list[Item]) -> None:
        self.cache = Path(cache)
        self.items = items
        self.tree: dict[str, dict[str, dict[str, list[Item]]]] = {}
        for it in items:
            coarse = it.category.split("/")[0]
            self.tree.setdefault(coarse, {}).setdefault(it.source, {}).setdefault(
                it.category, []
            ).append(it)
        self._cache: dict[str, np.ndarray] = {}

    def __bool__(self) -> bool:
        return bool(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def load(self, it: Item) -> np.ndarray:
        key = f"{it.source}/{it.name}"
        if key not in self._cache:
            a, _ = sf.read(source_dir(self.cache, it.source) / it.name, dtype="float32")
            if len(self._cache) > 512:
                self._cache.clear()
            self._cache[key] = a
        return self._cache[key]

    def pick(self, rng: np.random.Generator) -> Item:
        def choose(d: dict[str, Any]) -> Any:
            keys = sorted(d)
            return d[keys[int(rng.integers(len(keys)))]]

        group: list[Item] = choose(choose(choose(self.tree)))
        return group[int(rng.integers(len(group)))]

    def sample(self, rng: np.random.Generator, sr: int = SR) -> tuple[Item, np.ndarray]:
        it = self.pick(rng)
        return it, self.load(it)

    def sample_type(
        self, rng: np.random.Generator, coarse: str, sr: int = SR
    ) -> tuple[Item, np.ndarray] | None:
        """Like `sample`, restricted to one coarse type (e.g. "babble"); None if the pool has none."""
        sub = self.tree.get(coarse)
        if not sub:
            return None
        srcs = sub[sorted(sub)[int(rng.integers(len(sub)))]]
        group = srcs[sorted(srcs)[int(rng.integers(len(srcs)))]]
        it = group[int(rng.integers(len(group)))]
        return it, self.load(it)

    def sample_segment(
        self, rng: np.random.Generator, sr: int, seconds: float
    ) -> tuple[Item, np.ndarray, str]:
        """A `seconds`-long stretch of real speech."""
        src = sorted({it.source for it in self.items})[
            int(rng.integers(len({it.source for it in self.items})))
        ]
        idx = [i for i, it in enumerate(self.items) if it.source == src]
        i = idx[int(rng.integers(len(idx)))]
        first = self.items[i]
        parts: list[np.ndarray] = []
        texts: list[str] = []
        total = 0
        a = self.load(first)
        a = a[int(rng.integers(0, max(1, len(a) - int(seconds * sr) // 2))) :]
        while True:
            parts.append(a)
            texts.append(self.items[i].transcript)
            total += len(a)
            if total >= seconds * sr:
                break
            nxt = i + 1
            same = (
                nxt < len(self.items)
                and self.items[nxt].group
                and self.items[nxt].group == self.items[i].group
            )
            i = nxt if same else idx[int(rng.integers(len(idx)))]
            if not same:  # unrelated utterance: leave a pause
                parts.append(np.zeros(int(rng.uniform(0.2, 0.8) * sr), dtype=np.float32))
            a = self.load(self.items[i])
        return first, np.concatenate(parts)[: int(seconds * sr)], " ".join(t for t in texts if t)


class Pools:
    def __init__(self, cache: Path, enabled: list[str] | None = None) -> None:
        self.cache = Path(cache)
        self._pools: dict[tuple[str, str], AudioPool] = {}
        groups: dict[tuple[str, str], list[Item]] = {}
        for name, spec in SOURCES.items():
            if enabled is not None and name not in enabled:
                continue
            items = prepare_source(name, self.cache)
            groups.setdefault((spec.kind, spec.pool), []).extend(items)
        for k, items in groups.items():
            self._pools[k] = AudioPool(self.cache, items)

    def _get(self, kind: str, pool: str) -> AudioPool | None:
        p = self._pools.get((kind, pool))
        return p if p else None

    def noise(self, pool: str) -> AudioPool | None:
        return self._get("noise", pool)

    def rir(self, pool: str) -> AudioPool | None:
        return self._get("rir", pool)

    def speech(self, pool: str) -> AudioPool | None:
        return self._get("speech", pool)


def load_pools(cache: Path, enabled: list[str] | None = None) -> Pools:
    return Pools(cache, enabled)
