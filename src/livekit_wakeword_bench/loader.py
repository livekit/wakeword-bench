"""Data loader for benchmark / training datasets (local directory or Hugging Face repo).

from livekit_wakeword_bench import WakeWordData

data = WakeWordData("livekit/wakeword-bench-hey-livekit", split="test")
noisy = data.filter(noise_bucket=["very_noisy", "extreme"], category="positive")
for s in noisy:                      # Sample(audio=float32 @16k, label, meta)
    ...

# Fixed-length training windows (wake word aligned near the window end):
x, y = data.filter(split_category="positive").windows(2.0).arrays()

# PyTorch
ds = data.torch(window_s=2.0)        # torch.utils.data.Dataset
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import decode_audio, load_bench
from .mix import resample


@dataclass
class Sample:
    audio: np.ndarray  # float32 mono
    sample_rate: int
    label: int
    meta: dict[str, Any]

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate


def crop_window(
    audio: np.ndarray,
    sr: int,
    window_s: float,
    onset_s: float | None,
    offset_s: float | None,
    rng: np.random.Generator,
    end_jitter_s: tuple[float, float] = (0.0, 0.25),
) -> np.ndarray:
    """Cut a fixed-length window."""
    n = int(window_s * sr)
    if offset_s is not None and not np.isnan(offset_s):
        end = int((offset_s + rng.uniform(*end_jitter_s)) * sr)
        end = min(max(end, n if len(audio) >= n else len(audio)), len(audio))
        start = max(0, end - n)
    else:
        start = int(rng.integers(0, max(1, len(audio) - n + 1)))
        end = start + n
    w = audio[start:end]
    if len(w) < n:
        w = np.concatenate([np.zeros(n - len(w), dtype=np.float32), w])
    return w.astype(np.float32)


class WakeWordData:
    """Lazy, filterable view over a benchmark split. Audio is decoded on access."""

    def __init__(
        self,
        source: str | Path | Any,
        split: str = "test",
        sample_rate: int = 16000,
        window_s: float | None = None,
        seed: int = 0,
        transform: Callable[[np.ndarray, dict[str, Any], np.random.Generator], np.ndarray]
        | None = None,
    ) -> None:
        self.ds = load_bench(source, split) if isinstance(source, (str, Path)) else source
        self.split = split
        self.sample_rate = sample_rate
        self.window_s = window_s
        self.seed = seed
        self.transform = transform

    def _view(self, ds: Any, window_s: float | None = None) -> WakeWordData:
        return WakeWordData(
            ds,
            self.split,
            self.sample_rate,
            window_s if window_s is not None else self.window_s,
            self.seed,
            self.transform,
        )

    def with_transform(
        self,
        fn: Callable[[np.ndarray, dict[str, Any], np.random.Generator], np.ndarray] | None,
    ) -> WakeWordData:
        """Apply fn(audio, meta, rng) -> audio on access (e.g. `Augment` for training)."""
        v = self._view(self.ds)
        v.transform = fn
        return v

    def filter(
        self, fn: Callable[[dict[str, Any]], bool] | None = None, **equals: Any
    ) -> WakeWordData:
        """Filter by metadata. kwargs match exactly, or membership if given a list/tuple/set.

        data.filter(category="positive", noise_bucket=["noisy", "extreme"], provider="rime")
        data.filter(lambda m: m["snr_db"] < 0)
        """

        def keep(m: dict[str, Any]) -> bool:
            for k, v in equals.items():
                if isinstance(v, (list, tuple, set, frozenset)):
                    if m[k] not in v:
                        return False
                elif m[k] != v:
                    return False
            return fn(m) if fn else True

        cols = [c for c in self.ds.column_names if c != "audio"]
        meta_only = self.ds.select_columns(cols)
        idx = [i for i, m in enumerate(meta_only) if keep(m)]
        return self._view(self.ds.select(idx))

    def windows(self, window_s: float = 2.0) -> WakeWordData:
        """Return a view whose samples are fixed-length windows (see `crop_window`)."""
        return self._view(self.ds, window_s)

    def shuffle(self, seed: int | None = None) -> WakeWordData:
        return self._view(self.ds.shuffle(seed=self.seed if seed is None else seed))

    def take(self, n: int) -> WakeWordData:
        return self._view(self.ds.select(range(min(n, len(self.ds)))))

    def __len__(self) -> int:
        return len(self.ds)

    def metadata(self) -> Any:
        """All metadata as a pandas DataFrame (no audio decoding)."""
        cols = [c for c in self.ds.column_names if c != "audio"]
        return self.ds.select_columns(cols).to_pandas()

    def _make(self, row: dict[str, Any], i: int) -> Sample:
        audio, sr = decode_audio(row["audio"])
        if sr != self.sample_rate:
            audio = resample(audio, sr, self.sample_rate)
        meta = {k: v for k, v in row.items() if k != "audio"}
        if self.transform is not None:
            audio = self.transform(audio, meta, np.random.default_rng())
        if self.window_s:
            rng = np.random.default_rng((self.seed, i))
            off = meta.get("phrase_offset_s")
            if off is None or off != off:
                off = meta.get("offset_s")
            audio = crop_window(
                audio, self.sample_rate, self.window_s, None, off if meta["label"] else None, rng
            )
        return Sample(audio, self.sample_rate, int(row["label"]), meta)

    def __getitem__(self, i: int) -> Sample:
        return self._make(self.ds[int(i)], int(i))

    def __iter__(self) -> Iterator[Sample]:
        for i, row in enumerate(self.ds):
            yield self._make(row, i)

    def batches(self, batch_size: int = 64) -> Iterator[tuple[np.ndarray, np.ndarray, list[dict]]]:
        """Yield (audio[B, T], labels[B], metas). Requires `.windows()` for equal lengths."""
        if not self.window_s:
            raise ValueError("call .windows(window_s) first so all samples have equal length")
        buf: list[Sample] = []
        for s in self:
            buf.append(s)
            if len(buf) == batch_size:
                yield _stack(buf)
                buf = []
        if buf:
            yield _stack(buf)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialise everything as (audio[N, T], labels[N]). Requires `.windows()`."""
        xs, ys = [], []
        for x, y, _ in self.batches(256):
            xs.append(x)
            ys.append(y)
        return np.concatenate(xs), np.concatenate(ys)

    @staticmethod
    def collate(batch: list[tuple[Any, int, dict]]) -> tuple[Any, Any, list[dict]]:
        """DataLoader collate_fn: (audio[B, T], labels[B], metas)."""
        import torch

        return (
            torch.stack([b[0] for b in batch]),
            torch.tensor([b[1] for b in batch]),
            [b[2] for b in batch],
        )

    def torch(self, window_s: float | None = None) -> Any:
        """A torch.utils.data.Dataset returning (audio_tensor, label, meta)."""
        return TorchDataset(self.windows(window_s) if window_s else self)


class Augment:
    """On-the-fly training augmentation: the full `Augmenter` chain (reverb, competing talker, noise at SNR buckets, layered noise, mic EQ, band-limit, gain, clipping, codecs) drawn from the *train* noise / RIR / speech pools only, never the test ones."""

    def __init__(
        self,
        cache_dir: str | Path = ".cache/lkwb",
        spec: Any = None,
        sources: list[str] | None = None,
        sample_rate: int = 16000,
    ) -> None:
        from .config import AugmentSpec

        self.cache_dir = Path(cache_dir)
        self.spec = spec or AugmentSpec()
        self.sources = sources
        self.sr = sample_rate
        self._aug: Any = None

    def __call__(
        self, audio: np.ndarray, meta: dict[str, Any], rng: np.random.Generator
    ) -> np.ndarray:
        from .augment import Augmenter
        from .sources import load_pools

        if self._aug is None:  # lazy: each DataLoader worker loads its own pools
            pools = load_pools(self.cache_dir, self.sources)
            self._aug = Augmenter(self.spec, pools, "train", self.sr)
        aug_meta: dict[str, Any] = {}
        out = self._aug.augment_clip(audio, rng, aug_meta)
        meta["augmentations"] = aug_meta.get("augmentations", "")
        meta["noise_bucket"] = aug_meta.get("noise_bucket", meta.get("noise_bucket"))
        meta["snr_db"] = aug_meta.get("snr_db", meta.get("snr_db"))
        return out


class TorchDataset:
    """Map-style dataset for torch DataLoader (picklable, works with num_workers > 0)."""

    def __init__(self, view: WakeWordData) -> None:
        self.view = view

    def __len__(self) -> int:
        return len(self.view)

    def __getitem__(self, i: int) -> tuple[Any, int, dict]:
        import torch

        s = self.view[i]
        return torch.from_numpy(s.audio), s.label, s.meta


def _stack(samples: Sequence[Sample]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    return (
        np.stack([s.audio for s in samples]),
        np.array([s.label for s in samples], dtype=np.int64),
        [s.meta for s in samples],
    )
