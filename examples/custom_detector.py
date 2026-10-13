"""Benchmark your own wake word model on the public "hey livekit" benchmark.

    uv run python examples/custom_detector.py                 # quick: 300 random test clips
    uv run python examples/custom_detector.py --limit 0       # full test split

The bench hands your detector a whole clip; how it streams the audio is up to the model.
It returns a score timeline: scores[i] was emitted after hearing times[i] seconds of audio.
"""

from __future__ import annotations

import argparse

import numpy as np

from livekit_wakeword_bench import Prediction, benchmark
from livekit_wakeword_bench.dataset import DEFAULT_DATASET


class EnergyDetector:
    """A deliberately naive "model": fires on loud 80 ms frames. Replace with yours."""

    name = "energy-baseline"

    def detect(self, audio: np.ndarray, sample_rate: int) -> Prediction:
        hop = int(0.08 * sample_rate)
        n = len(audio) // hop
        frames = audio[: n * hop].reshape(n, hop)
        rms = np.sqrt((frames**2).mean(axis=1))
        scores = np.clip(rms / (rms.max() + 1e-9), 0, 1)  # your model's confidences go here
        times = (np.arange(n) + 1) * hop / sample_rate  # stream position of each score
        return Prediction(times, scores)
        # Models that only emit triggers: return Prediction.from_events([1.92, 7.4])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="local dir or HF repo id")
    ap.add_argument("--limit", type=int, default=300, help="random subset size, 0 = all")
    args = ap.parse_args()

    res = benchmark(EnergyDetector(), args.dataset, limit=args.limit or None)
    s = res.summary()
    print("at threshold 0.5:", s["at_threshold"])
    print("recall at FPPH targets:", {k: v and v["recall"] for k, v in s["at_target_fpph"].items()})
    print("worst noise buckets:", res.breakdown()["positive/noise_bucket"][:3])
    res.save("results/energy-baseline")  # then: lkwb report results/energy-baseline


if __name__ == "__main__":
    main()
