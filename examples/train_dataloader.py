"""PyTorch training data from a benchmark dataset, with on-the-fly augmentation.

    uv sync --extra torch
    uv run python examples/train_dataloader.py

The train split is stored clean. `Augment` applies the full augmentation chain (reverb,
competing talker, noise at SNR buckets, mic EQ, band-limit, codecs...) using only the
*train* noise / RIR / speech sources, so the test split stays unseen. The first run
downloads those sources (a few GB) into .cache/lkwb.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from livekit_wakeword_bench import WakeWordData
from livekit_wakeword_bench.dataset import DEFAULT_DATASET
from livekit_wakeword_bench.loader import Augment


def main(dataset: str = DEFAULT_DATASET) -> None:
    train = (
        WakeWordData(dataset, split="train")
        .with_transform(Augment())  # fresh augmentation on every access
        .windows(2.0)  # 2 s windows; positives end just after the wake word
    )
    loader = DataLoader(
        train.torch(),
        batch_size=64,
        shuffle=True,
        num_workers=4,
        collate_fn=WakeWordData.collate,
    )
    audio, labels, metas = next(iter(loader))
    print(audio.shape, labels.float().mean().item(), metas[0]["augmentations"])

    # Hard slices for targeted fine-tuning or evaluation:
    val = WakeWordData(dataset, split="validation")
    hard = val.filter(category="positive", noise_bucket=["very_noisy", "extreme"])
    print(f"{len(hard)} hard validation positives")


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:2])  # optional: local dataset dir or another HF repo id
