"""Long-form real-world negatives: hours of TV/YouTube audio, music and podcasts.

The test split's FPPH covers about an hour of read and meeting speech. Always-on devices mostly
false-trigger on media playing nearby (TV, radio, music with vocals, talk shows), so this set
measures false alarms per hour on ~24 h of it. It is phrase-independent: build it once, then pass
it to `benchmark(..., media=DIR)` for any wake word.

    lkwb media -o out/media                 # ~24 h (default mix below)
    lkwb bench hey_jarvis.onnx --data out/hey_jarvis --media out/media

Sources (all test-only; none is used to train or augment anything in the bench):

| source | content | license |
|---|---|---|
| `audioset_eval` | AudioSet evaluation clips (YouTube: TV, ads, music, crowds, vehicles) | CC BY 4.0 |
| `fma` | Free Music Archive tracks with vocals | per-track Creative Commons |
| `voxpopuli_<lang>` | VoxPopuli test splits: European Parliament speeches (6 languages + accented English), with transcripts | CC0 |

Clips are about 60 s (shorter source clips are concatenated, longer ones cut). Podcasts carry
transcripts, so a clip that actually says the wake word is excluded at benchmark time; AudioSet
and music have none, so a rare real mention there counts as a false alarm (check `failures()`).
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import soundfile as sf

from .mix import resample, to_mono

logger = logging.getLogger(__name__)
SR = 16000
CLIP_S = 60.0

# hours per source in the default build (~24 h)
DEFAULT_HOURS: dict[str, float] = {
    "audioset_eval": 10.0,
    "fma": 4.0,
    "voxpopuli_en_accented": 4.0,
    "voxpopuli_de": 1.5,
    "voxpopuli_fr": 1.5,
    "voxpopuli_es": 1.0,
    "voxpopuli_it": 1.0,
    "voxpopuli_pl": 1.0,
}
KIND = {"audioset_eval": "tv_youtube", "fma": "music"}  # voxpopuli_*: speeches


def _hf(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset")


def _files(repo: str, prefix: str) -> list[str]:
    from huggingface_hub import HfApi

    return sorted(
        f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(prefix)
    )


def _decode(data: bytes) -> tuple[np.ndarray, int] | None:
    try:
        a, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001
        try:  # mp3 inside tar files: fall back to librosa/audioread when libsndfile can't
            import librosa

            a, sr = librosa.load(io.BytesIO(data), sr=None, mono=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("skip undecodable clip: %s", e)
            return None
    return to_mono(np.asarray(a, dtype=np.float32)), int(sr)


def _audioset() -> Iterator[tuple[np.ndarray, str, str]]:
    repo = "agkphysics/AudioSet"
    import pyarrow.parquet as pq

    for f in _files(repo, "data/eval/"):
        for row in pq.read_table(_hf(repo, f)).to_pylist():
            got = _decode(row["audio"]["bytes"])
            if got:
                yield resample(got[0], got[1], SR), "/".join(row["human_labels"][:2]), ""


def _fma() -> Iterator[tuple[np.ndarray, str, str]]:
    repo = "benjamin-paine/free-music-archive-small"
    import pyarrow.parquet as pq

    for f in _files(repo, "data/"):
        for row in pq.read_table(_hf(repo, f)).to_pylist():
            if row.get("instrumental"):  # vocals are what false-trigger a wake word
                continue
            got = _decode(row["audio"]["bytes"])
            if got:
                tags = ",".join((row.get("tags") or [])[:3])
                yield resample(got[0], got[1], SR), f"music/{tags or 'untagged'}", ""


def _emilia(lang: str) -> Iterator[tuple[np.ndarray, str, str]]:
    repo = "amphion/Emilia-Dataset"
    # the last shards: furthest from anything a model might have seen as training speech
    for f in reversed(_files(repo, f"Emilia/{lang.upper()}/")):
        with tarfile.open(_hf(repo, f)) as tf:
            members = {m.name: m for m in tf.getmembers() if m.isfile()}
            for name, m in sorted(members.items()):
                if not name.endswith((".mp3", ".wav", ".flac")):
                    continue
                meta = members.get(name.rsplit(".", 1)[0] + ".json")
                text = ""
                if meta is not None:
                    fh = tf.extractfile(meta)
                    text = json.loads(fh.read()).get("text", "") if fh else ""
                fh = tf.extractfile(m)
                got = _decode(fh.read()) if fh else None
                if got:
                    yield resample(got[0], got[1], SR), f"podcast/{lang}", text


def _voxpopuli(lang: str) -> Iterator[tuple[np.ndarray, str, str]]:
    repo = "facebook/voxpopuli"
    import pyarrow.parquet as pq

    for f in _files(repo, f"{lang}/test-"):
        for row in pq.read_table(_hf(repo, f)).to_pylist():
            got = _decode(row["audio"]["bytes"])
            if got:
                yield resample(got[0], got[1], SR), f"speech/{lang}", str(row.get("raw_text") or "")


def _stream(source: str) -> Iterator[tuple[np.ndarray, str, str]]:
    if source == "audioset_eval":
        return _audioset()
    if source == "fma":
        return _fma()
    if source.startswith("voxpopuli_"):
        return _voxpopuli(source.removeprefix("voxpopuli_"))
    if source.startswith("emilia_"):  # gated: needs approved access to amphion/Emilia-Dataset
        return _emilia(source.removeprefix("emilia_"))
    raise ValueError(f"unknown media source {source!r}")


def _clips(source: str, hours: float) -> Iterator[tuple[np.ndarray, str, str]]:
    """~CLIP_S clips: short items concatenated, long ones cut, until `hours` is reached."""
    need, n = hours * 3600 * SR, int(CLIP_S * SR)
    buf: list[np.ndarray] = []
    cats: list[str] = []
    texts: list[str] = []
    have = 0
    for a, cat, text in _stream(source):
        while len(a):
            take = a[: n - have]
            a = a[len(take) :]
            buf.append(take)
            cats.append(cat)
            texts.append(text)
            have += len(take)
            if have >= n:
                yield np.concatenate(buf), cats[0], " ".join(t for t in texts if t)
                need -= have
                if need <= 0:
                    return
                buf, cats, texts, have = [], [], [], 0


def build_media(out: str | Path, hours: dict[str, float] | None = None) -> Path:
    """Write the media negatives as a bench dataset (split "media", category "media").
    Resumable: sources already written are skipped."""
    import pyarrow.parquet as pq

    from .dataset import ShardWriter

    out = Path(out)
    marker = out / "media_done.txt"  # sources written to the end, one per line
    done = set(marker.read_text().split()) if marker.exists() else set()
    shards = sorted((out / "data").glob("media-*.parquet"))
    for f in shards:  # an interrupted source leaves partial shards: drop them, redo it
        if pq.read_table(f, columns=["provider"]).column(0)[0].as_py() not in done:
            f.unlink()
    shards = [f for f in shards if f.exists()]
    w = ShardWriter(out, "media", SR, shard_size=200)
    w._shard = int(shards[-1].stem.split("-")[-1]) + 1 if shards else 0
    for source, h in (hours or DEFAULT_HOURS).items():
        if source in done:
            logger.info("[media] %s exists, skip", source)
            continue
        n = 0
        for audio, cat, text in _clips(source, h):
            w.add(
                audio,
                {
                    "id": f"media-{source}-{n:05d}",
                    "label": 0,
                    "category": "media",
                    "text": text,
                    "text_form": "transcript" if text else "",
                    "provider": source,
                    "noise_type": KIND.get(source, "speeches"),
                    "noise_category": cat,
                },
            )
            n += 1
        w.flush()
        with marker.open("a") as fh:
            fh.write(source + "\n")
        logger.info("[media] %s: %d clips (%.1f h)", source, n, n * CLIP_S / 3600)
    return out
