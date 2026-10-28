"""Build the README figures from a saved benchmark run (dev tool).

    uv run --group dev python scripts/readme_figures.py results/hey_livekit \
        --dataset binhpham/wakeword-bench-hey-livekit --model path/to/hey_livekit.onnx

Teaser: one test clip the model detects, replayed with a second real voice at --sir dB.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from livekit_wakeword_bench import WakeWordData, load_detector
from livekit_wakeword_bench.bench import BenchResult
from livekit_wakeword_bench.mix import active_rms, fit_length
from livekit_wakeword_bench.sources import load_pools
from livekit_wakeword_bench.viz import (
    README_AXES,
    failures,
    hstack,
    near_misses,
    recall_by_talker,
    traces,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="assets")
    ap.add_argument("--sir", type=float, default=5.0)
    ap.add_argument("--cache", default=".cache/lkwb")
    args = ap.parse_args()

    res = BenchResult.load(args.run)
    out = Path(args.out)
    (out / "failures.svg").write_text(
        failures(res, axes=README_AXES, significant_only=True), encoding="utf-8"
    )
    (out / "near_misses.svg").write_text(near_misses(res), encoding="utf-8")

    det = load_detector(args.model)
    thr = res.cfg.threshold
    data = WakeWordData(args.dataset, "test").filter(
        category="positive", noise_bucket="clean", reverb=False, interferer="", text_form="isolated"
    )
    talkers = load_pools(Path(args.cache)).speech("test")
    rng = np.random.default_rng(3)
    for s in data:
        alone = det.detect(s.audio, s.sample_rate)
        if alone.scores.max() < 0.9:
            continue
        _, talk, _ = talkers.sample_segment(rng, s.sample_rate, len(s.audio) / s.sample_rate + 1)
        talk = fit_length(talk, len(s.audio), rng)
        ref = active_rms(
            s.audio[
                int(s.meta["onset_s"] * s.sample_rate) : int(s.meta["offset_s"] * s.sample_rate)
            ],
            s.sample_rate,
        )
        talk = talk * ref / (np.sqrt(np.mean(talk**2)) + 1e-9) / 10 ** (args.sir / 20)
        mixed = s.audio + talk
        both = det.detect(mixed, s.sample_rate)
        if both.scores.max() >= thr * 0.7:
            continue
        ex = [
            {
                "label": "“Hey LiveKit”",
                "audio": s.audio,
                "sr": s.sample_rate,
                "times": alone.times,
                "scores": alone.scores,
            },
            {
                "label": "same clip + a second voice",
                "audio": mixed,
                "sr": s.sample_rate,
                "times": both.times,
                "scores": both.scores,
            },
        ]
        fig = hstack([traces(ex, thr, w=560), recall_by_talker(res)])
        (out / "teaser.svg").write_text(fig, encoding="utf-8")
        print(
            f"teaser from {s.id} ({s.meta['provider']}/{s.meta['voice_name']}): "
            f"{alone.scores.max():.2f} alone, {both.scores.max():.2f} with a voice at {args.sir:g} dB"
        )
        break


if __name__ == "__main__":
    main()
