# Development

```bash
uv sync --all-extras --group dev
uv run pytest
uv run ruff check src scripts tests examples
uv run ruff format --check src scripts tests examples
```

The tests run offline on a small synthetic dataset. CI runs them on Linux, macOS and Windows with Python 3.12 and 3.13, and checks that the built wheel installs and runs.

## Rules

- The library runs on Linux, macOS and Windows. No platform-specific code or dependencies in `src/`.
- Whisper, matplotlib and other QA tools go in the `dev` group only, never in the package.
- Train and test audio never share voices, noise, rooms or speakers.

## Layout

```
src/livekit_wakeword_bench/
  bench.py        benchmark, metrics, breakdowns, compare
  detectors.py    Detector protocol, livekit-wakeword and openWakeWord adapters
  viz.py          charts
  report.py       HTML report
  generate.py     plan → TTS → mix → parquet
  text.py         wake word forms, sentences, near-miss phrases
  phonemes.py     ARPAbet → each provider's phoneme syntax
  tts.py          LiveKit Inference TTS: concurrency, cache, retries
  voices.py       voice list and held-out test voices
  sources.py      noise, room and speech recordings
  augment.py      effects chain
  mix.py          signal processing
  media.py        long-form TV, music and speech negatives
  dataset.py      on-disk format
  loader.py       WakeWordData, Augment, PyTorch
  config.py       settings, sizes, plans
  export.py, card.py, scaffold.py, verify.py, cli.py
scripts/          dev tools: qa_dataset.py (Whisper check), readme_figures.py (README charts)
```

## Adding things

- **A model adapter:** add it to `_BUILTIN` in `detectors.py`. A model only needs `name` and `detect(audio, sample_rate) -> Prediction`.
- **An audio source:** `register(SourceSpec(...))` in `sources.py`. Keep train and test pools separate.
- **A TTS voice:** add it to `voices.json`, then run `lkwb voices --check`.
- **A TTS provider:** it must accept phonemes. Add its syntax to `phonemes.py` with a test pinned to the provider's documented examples.
- **A breakdown:** add the column to `POSITIVE_AXES` or `NEGATIVE_AXES` in `bench.py`.

## Releasing

The version comes from the git tag, so there is nothing to edit before a release. Pushing a `v*` tag runs [`publish.yml`](../.github/workflows/publish.yml). It builds the package, checks that the version matches the tag, installs the wheel in a clean environment, publishes to PyPI and creates a GitHub release with notes.

One-time setup:

1. On PyPI, add a [trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/) for `livekit-wakeword-bench`: owner `livekit`, repository `wakeword-bench`, workflow `publish.yml`, environment `pypi`.
2. On GitHub, create the `pypi` environment (Settings → Environments). Add required reviewers if releases need approval.

Release:

```bash
git checkout main && git pull
git tag v0.1.0
git push origin v0.1.0
```

Pre-releases work the same way: `v0.2.0rc1`.
