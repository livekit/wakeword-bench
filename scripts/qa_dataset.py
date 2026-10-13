"""Dataset QA (dev tool, not part of the library).

    uv run --group dev python scripts/qa_dataset.py out/hey_livekit --split test \
        [--asr-n 20] [--plots out/qa]

1. `lkwb verify` checks (integrity, leakage, SNR / level / timing / band-limit vs labels)
2. ASR on the *augmented* clips: is the wake word still intelligible at each noise level,
   and do adversarials stay distinguishable? (A human-intelligibility proxy: if Whisper still
   hears it at 0 dB SNR, a wake word model reasonably should too.)
3. Spectrogram gallery, one example per augmentation combination, for eyeballing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from _asr_backend import transcribe_all  # noqa: E402

from livekit_wakeword_bench import WakeWordData  # noqa: E402
from livekit_wakeword_bench.text import phonetic_distance  # noqa: E402
from livekit_wakeword_bench.verify import verify  # noqa: E402

SR = 16000


def asr_check(data: WakeWordData, n: int, model: str, workers: int) -> dict:
    df = data.metadata()
    sel = df[df.category.isin(["positive", "adversarial"])].sample(min(n, len(df)), random_state=0)
    idx = {sid: i for i, sid in enumerate(data.ds["id"])}
    tmp = Path(tempfile.mkdtemp(prefix="lkwb_qa_"))
    jobs = {}
    for _, m in sel.iterrows():
        s = data[idx[m["id"]]]
        a = s.audio[int(max(0.0, m["onset_s"] - 0.3) * SR) : int((m["offset_s"] + 0.3) * SR)]
        p = tmp / f"{m['id']}.flac"
        sf.write(p, a / (np.abs(a).max() + 1e-9) * 0.5, SR)
        jobs[f"qa::{data.split}::{m['id']}"] = p
    texts = transcribe_all(jobs, tmp, model, workers)  # throwaway cache
    wake = df.wake_word.iloc[0]
    rows = []
    for _, m in sel.iterrows():
        t = texts[f"qa::{data.split}::{m['id']}"]
        rows.append({**m.to_dict(), "asr": t, "dist": phonetic_distance(t, wake)})
    out: dict = {"n": len(rows)}
    pos = [r for r in rows if r["category"] == "positive"]
    adv = [r for r in rows if r["category"] == "adversarial"]
    out["positive_heard(dist<=1)"] = round(float(np.mean([r["dist"] <= 1 for r in pos])), 3)
    out["adversarial_heard_as_wake(dist==0)"] = round(
        float(np.mean([r["dist"] == 0 for r in adv])), 3
    )
    for axis in ("noise_bucket", "augmentations", "codec", "provider"):
        g = defaultdict(list)
        for r in pos:
            g[str(r[axis])].append(r["dist"] <= 1)
        out[f"positive_heard_by_{axis}"] = {
            k: f"{np.mean(v):.2f} (n={len(v)})"
            for k, v in sorted(g.items(), key=lambda kv: np.mean(kv[1]))
        }
    out["clean_positive_misses"] = [
        {
            "id": r["id"],
            "voice": f"{r['model']}:{r['voice_name']}",
            "text": r["text"],
            "asr": r["asr"],
        }
        for r in pos
        if r["dist"] > 1 and r["noise_bucket"] in ("clean", "quiet") and not r["reverb"]
    ][:15]
    return out


def plot_gallery(data: WakeWordData, out_dir: Path, per_key: int = 1) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = data.metadata()
    pos = df[df.category == "positive"]
    picks = pos.groupby("augmentations").head(per_key).sort_values("augmentations").head(24)
    idx = {sid: i for i, sid in enumerate(data.ds["id"])}
    cols = 3
    rows = math.ceil(len(picks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 2.7), squeeze=False)
    for ax, (_, m) in zip(axes.flat, picks.iterrows()):
        a = data[idx[m["id"]]].audio
        ax.specgram(a + 1e-7, NFFT=512, Fs=SR, noverlap=352, cmap="magma", vmin=-140)
        for t in (m["phrase_onset_s"], m["phrase_offset_s"]):
            if t == t:
                ax.axvline(t, color="cyan", lw=1)
        snr = f"{m['snr_db']:.0f} dB" if m["snr_db"] == m["snr_db"] else m["noise_bucket"]
        ax.set_title(
            f"{m['augmentations']}\n{m['noise_category'] or '-'} · {snr} · {m['codec']}", fontsize=8
        )
        ax.set_ylim(0, 8000)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(picks) :]:
        ax.axis("off")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gallery_{data.split}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--split", default="test")
    ap.add_argument("--asr-n", type=int, default=20, help="clips to transcribe (0 to skip)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plots", default=None)
    args = ap.parse_args()

    report = {"verify": verify(args.source, args.split)}
    data = WakeWordData(args.source, split=args.split)
    if args.asr_n:
        model = args.model or "base.en"
        report["asr"] = asr_check(data, args.asr_n, model, args.workers)
    if args.plots:
        report["gallery"] = str(plot_gallery(data, Path(args.plots)))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
