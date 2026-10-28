# Development

## Set up and check

From the repository root:

```bash
uv sync --all-extras --group dev
uv run pytest
uv run ruff check src scripts tests examples
uv run ruff format --check src scripts tests examples
```

For livekit-wakeword v2 models, also run `uv pip install livekit-wakeword`, or `uv pip install -e ../livekit-wakeword` for a local checkout. Reinstall after `uv sync`, which removes this undeclared dependency.

Tests use a small synthetic dataset and run offline. CI tests Linux, macOS and Windows on Python 3.12 and 3.13, then checks the built wheel.

## Code map

Files are under `src/livekit_wakeword_bench/` unless noted.

| Area | Files |
|---|---|
| Benchmark and model adapters | `bench.py`, `detectors.py` |
| Charts and reports | `viz.py`, `report.py` |
| Data generation | `generate.py`, `text.py`, `phonemes.py`, `tts.py`, `voices.py` |
| Audio sources and effects | `sources.py`, `augment.py`, `mix.py`, `media.py` |
| Storage and loading | `dataset.py`, `loader.py` |
| Configuration and CLI | `config.py`, `cli.py`, `scaffold.py` |
| Exports and checks | `export.py`, `card.py`, `verify.py` |
| Dataset QA and README charts | `scripts/qa_dataset.py`, `scripts/readme_figures.py` |

Keep `src/` portable across supported platforms. Put QA dependencies, such as Whisper and matplotlib, in the `dev` group. Keep training and test voices, noise, rooms and speakers separate.

## Extend the package

| Add | Change |
|---|---|
| Model adapter | Register it in `_BUILTIN` in `detectors.py`; implement `name` and `detect(audio, sample_rate) -> Prediction` |
| Audio source | Register a `SourceSpec` in `sources.py`; preserve train/test separation |
| TTS voice | Update `voices.json`, then run `lkwb voices --check` |
| TTS provider | Add phoneme syntax in `phonemes.py` and a test based on the provider's documented examples |
| Breakdown | Add a column to `POSITIVE_AXES` or `NEGATIVE_AXES` in `bench.py` |

To regenerate README figures from a local model:

```bash
uv run lkwb bench hey_livekit.onnx
uv run --group dev python scripts/readme_figures.py results/hey_livekit \
    --dataset binhpham/wakeword-bench-hey-livekit --model hey_livekit.onnx
```

## Releasing

Versions come from Git tags. Pushing a `v*` tag triggers [publish.yml](../.github/workflows/publish.yml): build, version check, clean wheel install, PyPI publish and GitHub release.

One-time setup:

1. Add a PyPI [trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) for `livekit-wakeword-bench`: owner `livekit`, repository `wakeword-bench`, workflow `publish.yml`, environment `pypi`.
2. Create the `pypi` environment in GitHub Settings → Environments. Add reviewers if approval is required.

After checks pass, tag the intended commit on `main`. Replace the example version with the next release:

```bash
git checkout main
git pull
git tag v0.1.0
git push origin v0.1.0
```

Pre-releases use the same process, for example `v0.2.0rc1`.
