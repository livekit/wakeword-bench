"""Benchmark runner: stream every clip through a Detector, then compute metrics + breakdowns."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import DEFAULT_DATASET
from .detectors import Detector, load_detector
from .loader import Sample, WakeWordData

THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 2)
# a trigger needs score >= threshold AND score > 0 (so a silent model never "fires" at 0)
_EFF = np.maximum(THRESHOLDS, 1e-9)

POSITIVE_AXES = [
    "noise_bucket",
    "noise_type",
    "noise_category",
    "reverb",
    "rir_category",
    "interferer",
    "n_talkers",
    "eq_applied",
    "bandlimit",
    "clipping",
    "codec",
    "augmentations",
    "level_bucket",
    "speed_bucket",
    "text_form",
    "provider",
    "gender",
    "accent",
    "held_out_voice",
    "voice",
]
NEGATIVE_AXES = [
    "category",
    "noise_bucket",
    "noise_type",
    "noise_category",
    "reverb",
    "codec",
    "augmentations",
    "provider",
    "adversarial_distance",
]


@dataclass
class BenchConfigRun:
    debounce_s: float = 2.0  # min time between two counted triggers (matches listener default)
    max_latency_s: float = 1.0  # a positive counts if triggered before wake_offset + this
    threshold: float = 0.5  # primary operating point for breakdowns
    target_fpph: tuple[float, ...] = (1.0, 0.5, 0.1)
    score_fpph: float = 0.5
    # adversarial clips are scored as a false-accept rate, not FPPH
    fpph_categories: tuple[str, ...] = ("real_speech", "speech", "noise")


@dataclass
class SampleResult:
    id: str
    times: np.ndarray  # stream position (s) of each score
    scores: np.ndarray
    compute_s: float  # wall time of detect() on this clip
    audio_s: float


def _run(det: Detector, sample: Sample) -> SampleResult:
    t0 = time.perf_counter()
    pred = det.detect(sample.audio, sample.sample_rate)
    dt = time.perf_counter() - t0
    return SampleResult(
        sample.id,
        np.asarray(pred.times, dtype=np.float32),
        np.asarray(pred.scores, dtype=np.float32),
        dt,
        sample.duration_s,
    )


_W: dict[str, Any] = {}


def _worker_init(spec: str, source: str, split: str) -> None:
    _W["det"] = load_detector(spec)
    _W["data"] = WakeWordData(source, split=split)


def _worker_run(indices: list[int]) -> list[SampleResult]:
    return [_run(_W["det"], _W["data"][i]) for i in indices]


def run_detector(
    spec: str | Detector,
    data: WakeWordData,
    source: str | None = None,
    workers: int = 1,
    progress: bool = True,
) -> list[SampleResult]:
    """Run the detector on every clip."""
    from tqdm import tqdm

    n = len(data)
    if workers <= 1:
        det = load_detector(spec) if isinstance(spec, str) else spec
        return [_run(det, s) for s in tqdm(data, total=n, disable=not progress, desc="bench")]
    if not isinstance(spec, str) or source is None:
        raise ValueError("workers>1 needs a detector spec string and a dataset source")
    ids = list(range(n))
    batches = [ids[i : i + 32] for i in range(0, n, 32)]
    # Workers re-open the *unfiltered* split, so map back via the original row ids.
    all_ids = WakeWordData(source, split=data.split).ds["id"]
    pos = {sid: i for i, sid in enumerate(all_ids)}
    want = [pos[sid] for sid in data.ds["id"]]
    batches = [[want[i] for i in b] for b in batches]
    out: list[SampleResult] = []
    with ProcessPoolExecutor(
        workers, initializer=_worker_init, initargs=(spec, source, data.split)
    ) as ex:
        for res in tqdm(
            ex.map(_worker_run, batches), total=len(batches), disable=not progress, desc="bench"
        ):
            out.extend(res)
    return out


def _events(scores: np.ndarray, times: np.ndarray, thr: float, debounce_s: float) -> list[int]:
    idx = np.flatnonzero(scores >= thr)
    out: list[int] = []
    last = -math.inf
    for i in idx:
        t = float(times[i])
        if t - last >= debounce_s:
            out.append(int(i))
            last = t
    return out


def _bucket_level(x: float) -> str:
    if x != x:
        return "n/a"
    return "loud(>-10dB)" if x > -10 else "normal(-20..-10)" if x > -20 else "soft(<-20dB)"


def _bucket_speed(x: float) -> str:
    return "slow(<0.95)" if x < 0.95 else "fast(>1.05)" if x > 1.05 else "normal"


@dataclass
class BenchResult:
    detector: str
    rows: list[dict[str, Any]]  # per-sample metadata + derived results
    traces: dict[str, tuple[np.ndarray, np.ndarray]]  # id -> (times, scores)
    cfg: BenchConfigRun = field(default_factory=BenchConfigRun)

    def frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame(self.rows)

    def summary(self) -> dict[str, Any]:
        df = self.frame()
        pos = df[df.label == 1]
        neg = df[df.label == 0]
        amb = neg[neg.category.isin(self.cfg.fpph_categories)]
        adv = neg[neg.category == "adversarial"]
        neg_hours = float(amb.duration_s.sum()) / 3600
        nan = np.full(len(THRESHOLDS), np.nan)

        def stack(col: Any) -> np.ndarray:
            return np.stack(col.to_numpy()) if len(col) else np.zeros((0, len(THRESHOLDS)))

        recall = stack(pos.detected).mean(0) if len(pos) else nan
        ev = stack(amb.events)
        fpph = ev.sum(0) / neg_hours if neg_hours > 0 else nan
        far = (ev > 0).mean(0) if len(amb) else nan
        adv_far = (stack(adv.events) > 0).mean(0) if len(adv) else nan

        def op(thr_idx: int) -> dict[str, Any]:
            t = float(THRESHOLDS[thr_idx])
            lat = latencies(pos, thr_idx)
            return {
                "threshold": t,
                "recall": float(recall[thr_idx]),
                "fpph": float(fpph[thr_idx]),
                "false_accept_rate": float(far[thr_idx]),
                "adversarial_false_accept_rate": float(adv_far[thr_idx]),
                "latency_ms_p50": _pct(lat, 50),
                "latency_ms_p90": _pct(lat, 90),
            }

        ti = int(np.argmin(np.abs(THRESHOLDS - self.cfg.threshold)))
        targets = {}
        for tgt in self.cfg.target_fpph:
            ok = np.flatnonzero(fpph <= tgt)
            targets[f"fpph<={tgt}"] = op(int(ok[0])) if len(ok) else None
        comp = df.compute_s.to_numpy(dtype=np.float64)
        aud = df.duration_s.to_numpy(dtype=np.float64)
        ok = np.flatnonzero(fpph <= self.cfg.score_fpph)
        score = float(recall[ok[0]]) if len(ok) else 0.0
        media = self._media(
            neg[neg.category == "media"], stack, ti, int(ok[0]) if len(ok) else None
        )
        party = self._party(df, stack, ti, int(ok[0]) if len(ok) else None)
        return {
            "detector": self.detector,
            "score": round(score, 4),
            "score_definition": f"recall at FPPH <= {self.cfg.score_fpph}",
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
            "n_adversarial": int(len(adv)),
            "fpph_hours": round(neg_hours, 3),
            "at_threshold": op(ti),
            "media": media,
            "party": party,
            "at_target_fpph": targets,
            "compute": {
                "real_time_factor": round(float(comp.sum() / max(aud.sum(), 1e-9)), 5),
                "ms_per_audio_s_p50": _pct(comp / aud * 1000, 50),
                "ms_per_audio_s_p95": _pct(comp / aud * 1000, 95),
                "total_compute_s": round(float(comp.sum()), 2),
                "total_audio_s": round(float(aud.sum()), 2),
            },
            "curve": {
                "thresholds": THRESHOLDS.tolist(),
                "recall": _clean(recall),
                "fpph": _clean(fpph),
                "false_accept_rate": _clean(far),
                "adversarial_false_accept_rate": _clean(adv_far),
            },
        }

    @staticmethod
    def _media(med: Any, stack: Any, ti: int, score_idx: int | None) -> dict[str, Any] | None:
        """False alarms per hour on the long-form media negatives (`benchmark(media=...)`), at the
        operating threshold and at the threshold that defines `score`; not part of `score`."""
        if not len(med):
            return None

        def fa(g: Any) -> np.ndarray:
            return stack(g.events).sum(0) / (float(g.duration_s.sum()) / 3600)

        def at(curve: np.ndarray) -> dict[str, float | None]:
            return {
                "at_threshold": round(float(curve[ti]), 3),
                "at_score_threshold": None
                if score_idx is None
                else round(float(curve[score_idx]), 3),
            }

        out: dict[str, Any] = {"hours": round(float(med.duration_s.sum()) / 3600, 3), **at(fa(med))}
        out["by_type"] = {
            str(k): {"hours": round(float(g.duration_s.sum()) / 3600, 3), **at(fa(g))}
            for k, g in med.groupby("noise_type")
        }
        out["curve"] = _clean(fa(med))
        return out

    @staticmethod
    def _party(df: Any, stack: Any, ti: int, score_idx: int | None) -> dict[str, Any] | None:
        """Wake words said among several overlapping talkers + crowd babble (`augment.p_party`):
        recall on party positives and the share of party negatives that trigger, at the operating
        threshold and at the threshold that defines `score`. Party clips are part of `score` too."""
        if "party" not in df or not df.party.fillna(False).astype(bool).any():
            return None
        p = df[df.party.fillna(False).astype(bool)]
        pos, neg = p[p.label == 1], p[p.label == 0]

        def at(curve: np.ndarray | None) -> dict[str, float | None]:
            if curve is None:
                return {"at_threshold": None, "at_score_threshold": None}
            return {
                "at_threshold": round(float(curve[ti]), 4),
                "at_score_threshold": None
                if score_idx is None
                else round(float(curve[score_idx]), 4),
            }

        rec = stack(pos.detected).mean(0) if len(pos) else None
        far = (stack(neg.events) > 0).mean(0) if len(neg) else None
        out: dict[str, Any] = {
            "n_positive": int(len(pos)),
            "n_negative": int(len(neg)),
            "recall": at(rec),
            "false_accept_rate": at(far),
        }
        if len(pos) and "n_talkers" in pos:
            out["recall_by_talkers"] = {
                str(int(k)): round(float(stack(g.detected).mean(0)[ti]), 4)
                for k, g in pos.groupby("n_talkers")
            }
        return out

    def breakdown(self, threshold: float | None = None) -> dict[str, list[dict[str, Any]]]:
        """Per-axis slices at a threshold: recall for positives, FA rate / FPPH for negatives."""
        df = self.frame()
        ti = int(np.argmin(np.abs(THRESHOLDS - (threshold or self.cfg.threshold))))
        pos = df[df.label == 1].copy()
        neg = df[df.label == 0].copy()
        out: dict[str, list[dict[str, Any]]] = {}
        if len(pos):
            pos["hit"] = [d[ti] for d in pos.detected]
            for ax in POSITIVE_AXES:
                if ax not in pos:
                    continue
                rows = []
                for val, g in pos.groupby(ax, dropna=False):
                    lat = latencies(g, ti)
                    rows.append(
                        {
                            "value": str(val),
                            "n": int(len(g)),
                            "recall": float(g.hit.mean()),
                            "ci95": wilson(float(g.hit.sum()), len(g)),
                            "mean_score": float(g.max_score.mean()),
                            "latency_ms_p50": _pct(lat, 50),
                        }
                    )
                # sort by CI upper bound so tiny slices don't outrank confidently bad ones
                out[f"positive/{ax}"] = sorted(rows, key=lambda r: (r["ci95"][1], r["recall"]))
        if len(neg):
            neg["ev"] = [e[ti] for e in neg.events]
            for ax in NEGATIVE_AXES:
                if ax not in neg:
                    continue
                rows = []
                for val, g in neg.groupby(ax, dropna=False):
                    h = g.duration_s.sum() / 3600
                    rows.append(
                        {
                            "value": str(val),
                            "n": int(len(g)),
                            "hours": round(float(h), 3),
                            "false_accept_rate": float((g.ev > 0).mean()),
                            "ci95": wilson(float((g.ev > 0).sum()), len(g)),
                            "fpph": float(g.ev.sum() / max(h, 1e-9)),
                            "mean_score": float(g.max_score.mean()),
                        }
                    )
                out[f"negative/{ax}"] = sorted(
                    rows, key=lambda r: (-r["ci95"][0], -r["false_accept_rate"])
                )
            adv = neg[neg.category == "adversarial"]
            if len(adv):
                rows = []
                for val, g in adv.groupby("phrase"):
                    rows.append(
                        {
                            "value": str(val),
                            "n": int(len(g)),
                            "false_accept_rate": float((g.ev > 0).mean()),
                            "ci95": wilson(float((g.ev > 0).sum()), len(g)),
                            "mean_score": float(g.max_score.mean()),
                        }
                    )
                rows.sort(key=lambda r: (-r["ci95"][0], -r["false_accept_rate"]))
                out["negative/adversarial_phrase"] = rows[:40]
        return out

    def failures(self, threshold: float | None = None, k: int = 25) -> dict[str, list[dict]]:
        """Worst misses (lowest-scoring positives) and worst false alarms (highest negatives)."""
        df = self.frame()
        thr = threshold or self.cfg.threshold
        pos = df[(df.label == 1) & (df.max_score < thr)].sort_values("max_score").head(k)
        neg = (
            df[(df.label == 0) & (df.max_score >= thr)]
            .sort_values("max_score", ascending=False)
            .head(k)
        )
        cols = [c for c in df.columns if c not in ("detected", "events", "first_hit_s")]
        return {
            "missed_positives": pos[cols].to_dict("records"),
            "false_alarms": neg[cols].to_dict("records"),
        }

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df = self.frame()
        df.to_parquet(out / "samples.parquet")
        np.savez_compressed(out / "times.npz", **{k: v[0] for k, v in self.traces.items()})
        np.savez_compressed(
            out / "scores.npz", **{k: v[1].astype(np.float32) for k, v in self.traces.items()}
        )
        meta = {"detector": self.detector, "cfg": self.cfg.__dict__}
        (out / "run.json").write_text(json.dumps(meta, indent=2))
        (out / "summary.json").write_text(json.dumps(self.summary(), indent=2))
        (out / "breakdown.json").write_text(json.dumps(self.breakdown(), indent=2))
        return out

    def rescore(self, cfg: BenchConfigRun | None = None) -> BenchResult:
        """Recompute every metric from the stored score traces (no model run), e.g. with a
        different debounce / latency window / score target."""
        comp = {r["id"]: r.get("compute_s", 0.0) for r in self.rows}
        results = [
            SampleResult(r["id"], *self.traces[r["id"]], comp[r["id"]], r["duration_s"])
            for r in self.rows
        ]
        return evaluate(results, self.rows, self.detector, cfg or self.cfg)

    @classmethod
    def load(cls, out_dir: str | Path) -> BenchResult:
        import pandas as pd

        out = Path(out_dir)
        meta = json.loads((out / "run.json").read_text())
        df = pd.read_parquet(out / "samples.parquet")
        rows = df.to_dict("records")
        for r in rows:
            for k in ("detected", "events", "first_hit_s"):
                r[k] = np.asarray(r[k])
        sc = np.load(out / "scores.npz")
        tm = np.load(out / "times.npz")
        cfg = BenchConfigRun(**{**meta["cfg"], "target_fpph": tuple(meta["cfg"]["target_fpph"])})
        traces = {k: (tm[k], sc[k].astype(np.float32)) for k in sc.files}
        return cls(meta["detector"], rows, traces, cfg)


def compare(base: BenchResult, new: BenchResult, min_n: int = 20) -> dict[str, Any]:
    """Deltas new - base: headline metrics and every breakdown slice (n >= min_n), sorted
    worst regression first. Positive slices compare recall, negative slices false accepts."""
    sa, sb = base.summary(), new.summary()
    head = {}
    for k in (
        "recall",
        "fpph",
        "false_accept_rate",
        "adversarial_false_accept_rate",
        "latency_ms_p50",
    ):
        a, b = sa["at_threshold"][k], sb["at_threshold"][k]
        head[k] = {"base": a, "new": b, "delta": None if a is None or b is None else b - a}
    head["score"] = {"base": sa["score"], "new": sb["score"], "delta": sb["score"] - sa["score"]}
    ba, bb = base.breakdown(), new.breakdown()
    slices = []
    for key, rows in bb.items():
        old = {r["value"]: r for r in ba.get(key, [])}
        metric = "recall" if key.startswith("positive/") else "false_accept_rate"
        sign = 1 if metric == "recall" else -1  # >0 = better
        for r in rows:
            o = old.get(r["value"])
            if o is None or min(r["n"], o["n"]) < min_n:
                continue
            d = r[metric] - o[metric]
            slices.append(
                {
                    "slice": f"{key}={r['value']}",
                    "metric": metric,
                    "base": o[metric],
                    "new": r[metric],
                    "delta": d,
                    "improvement": sign * d,
                    "n": r["n"],
                }
            )
    slices.sort(key=lambda x: x["improvement"])
    return {"base": base.detector, "new": new.detector, "headline": head, "slices": slices}


def latencies(pos: Any, ti: int) -> np.ndarray:
    """Detection latency (ms) = first trigger time - wake word end, for samples with known wake word timing that were detected at threshold index ti."""
    if not len(pos):
        return np.zeros(0)
    hit = np.array([f[ti] for f in pos.first_hit_s], dtype=np.float64)
    off = pos.phrase_offset_s.to_numpy(dtype=np.float64)
    m = ~np.isnan(hit) & ~np.isnan(off)
    return (hit[m] - off[m]) * 1000


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def _pct(x: np.ndarray, q: float) -> float | None:
    return round(float(np.percentile(x, q)), 2) if len(x) else None


def _clean(a: np.ndarray) -> list[float | None]:
    return [None if (v != v) else round(float(v), 5) for v in a]


_DERIVED = (
    "detected",
    "first_hit_s",
    "events",
    "compute_s",
    "level_bucket",
    "speed_bucket",
    "voice",
    "eq_applied",
    "max_score",
    "max_score_t",
    "early_max_score",
)


def evaluate(
    results: Iterable[SampleResult],
    data: WakeWordData | list[dict[str, Any]],
    detector_name: str,
    cfg: BenchConfigRun | None = None,
) -> BenchResult:
    """Turn raw score timelines into per-sample, per-threshold outcomes."""
    cfg = cfg or BenchConfigRun()
    records = data if isinstance(data, list) else data.metadata().to_dict("records")
    meta = {m["id"]: {k: v for k, v in m.items() if k not in _DERIVED} for m in records}
    rows: list[dict[str, Any]] = []
    traces: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for r in results:
        m = dict(meta[r.id])
        s, t = r.scores, r.times
        traces[r.id] = (t, s)
        detected = np.zeros(len(THRESHOLDS), dtype=bool)
        first_hit = np.full(len(THRESHOLDS), np.nan)
        events = np.zeros(len(THRESHOLDS), dtype=np.int32)
        if m["label"] == 1:
            on, off = m["phrase_onset_s"], m["phrase_offset_s"]
            if on != on or off != off:  # unknown phrase timing: use the whole utterance
                on, off = m["onset_s"], m["offset_s"]
            region = (t >= on) & (t <= off + cfg.max_latency_s)
            rs = np.where(region, s, -1.0)
            if len(rs):
                runmax = np.maximum.accumulate(rs)
                idx = np.searchsorted(runmax, _EFF, side="left")
                ok = idx < len(s)
                detected[ok] = True
                first_hit[ok] = t[idx[ok]]
            m["max_score"] = float(rs.max()) if region.any() else 0.0
            pre = s[t < on]
            m["early_max_score"] = float(pre.max()) if len(pre) else 0.0
        else:
            for j, thr in enumerate(THRESHOLDS):
                if len(s) == 0 or s.max() < _EFF[j]:
                    break
                events[j] = len(_events(s, t, _EFF[j], cfg.debounce_s))
            m["max_score"] = float(s.max()) if len(s) else 0.0
            m["max_score_t"] = float(t[int(np.argmax(s))]) if len(s) else 0.0
        m["detected"] = detected
        m["first_hit_s"] = first_hit
        m["events"] = events
        m["compute_s"] = r.compute_s
        m["level_bucket"] = _bucket_level(m.get("level_dbfs", math.nan))
        m["speed_bucket"] = _bucket_speed(m.get("speed", 1.0))
        m["voice"] = f"{m.get('provider', '')}/{m.get('voice_name', '')}"
        m["eq_applied"] = bool(m.get("eq"))
        rows.append(m)
    return BenchResult(detector_name, rows, traces, cfg)


def benchmark(
    model: str | Detector,
    dataset: str | Path = DEFAULT_DATASET,
    split: str = "test",
    cfg: BenchConfigRun | None = None,
    workers: int = 1,
    limit: int | None = None,
    media: str | Path | None = None,
    **filters: Any,
) -> BenchResult:
    """Benchmark a model on a dataset (local dir or HF repo id, default: the public set).

    `model`: a livekit-wakeword v2 `.onnx` path, a detector spec, or a Detector object.

    res = benchmark("hey_livekit.onnx", limit=500)   # quick run on the public set
    res.summary()["score"]; res.breakdown(); res.failures(); res.save("results/x")

    `media`: a long-form negatives set (`lkwb media`): also report false alarms per hour on it
    (`summary()["media"]`). Clips whose transcript says the wake word are left out.
    """
    data = WakeWordData(dataset, split=split)
    if filters:
        data = data.filter(**filters)
    if limit:
        data = data.shuffle(0).take(limit)
    detector: str | Detector = model
    det_name = model if isinstance(model, str) else model.name
    if workers <= 1:
        detector = load_detector(model)
        det_name = detector.name
    res = run_detector(detector, data, source=str(dataset), workers=workers)
    out = evaluate(res, data, det_name, cfg)
    if media is not None:
        add_media(out, detector, data, media, workers=workers)
    return out


def add_media(
    res: BenchResult,
    detector: str | Detector,
    data: WakeWordData,
    media: str | Path,
    workers: int = 1,
    progress: bool = True,
) -> BenchResult:
    """Run `detector` on a media negatives set and add its clips to `res` (see `benchmark`)."""
    words = sorted({str(w) for w in data.metadata().wake_word if w})
    md = WakeWordData(str(media), split="media").filter(lambda m: not _says(m["text"], words))
    raw = run_detector(detector, md, source=str(media), workers=workers, progress=progress)
    extra = evaluate(raw, md, res.detector, res.cfg)
    res.rows.extend(extra.rows)
    res.traces.update(extra.traces)
    return res


def _says(text: str, words: list[str]) -> bool:
    """Does a transcript contain one of the wake words (case/punctuation-insensitive)?"""
    import re

    norm = " " + " ".join(re.findall(r"\w+", str(text).lower())) + " "
    return any(" " + " ".join(re.findall(r"\w+", w.lower())) + " " in norm for w in words)
