# Development

```bash
uv sync --all-extras --group dev
uv run pytest
uv run ruff check src scripts tests examples
uv run ruff format src scripts tests examples
```

The tests run offline in a few seconds. They build a small synthetic dataset and check the loader, the benchmark metrics, the report, mixing at an exact SNR, codecs without delay, phoneme rendering against the providers' documented examples, and that a silent model scores 0.

## Platform policy

The library is pure Python (numpy, scipy, soundfile, pyarrow) and runs on Linux, macOS and Windows. Keep it that way: no platform-specific code or dependencies in `src/` or in the uv dependency groups. Whisper, matplotlib and other QA tools are allowed only in the `dev` group.

## Layout

```
src/livekit_wakeword_bench/
  cli.py          lkwb commands
  config.py       settings, knobs, sizes, plans
  generate.py     plan → TTS → mix → parquet shards
  text.py         wake word forms, carrier sentences, near-miss phrases, phonetic matching
  phonemes.py     ARPAbet → Cartesia / IPA / Rime phoneme syntax
  tts.py          LiveKit Inference TTS: concurrency limit, cache, retries
  voices.py       voice catalog (voices.json) and held-out voice split
  sources.py      Hugging Face noise / room / speech sources, train and test pools
  augment.py      the augmentation chain
  mix.py          signal processing: resampling, trimming, reverb, SNR mixing, filters
  dataset.py      on-disk format and schema
  loader.py       WakeWordData, Augment, PyTorch adapter
  detectors.py    Detector protocol and built-in adapters
  bench.py        benchmark runner, metrics, breakdowns, compare
  viz.py          SVG charts (LiveKit Bytes design tokens)
  report.py       HTML report
  verify.py       dataset checks
  export.py       livekit-wakeword layout, WAV + CSV
  card.py         dataset card and Hub upload
  scaffold.py     lkwb init
scripts/          development tools (not part of the package)
examples/         runnable examples
configs/          example generation configs
assets/           README charts
docs/             this documentation
```

## Development scripts

| script | does |
|---|---|
| `scripts/qa_dataset.py DATASET [--asr-n 20] [--plots DIR]` | runs `verify`, transcribes a sample of clips with Whisper to confirm they say what their labels claim, and writes a spectrogram gallery |
| `scripts/readme_figures.py RUN --dataset D --model M` | rebuilds every README chart from a saved run, including the "one clip, two rooms" example |

Both need the `dev` group; `readme_figures.py` also needs the `livekit` extra.

## Adding things

- **A detector adapter:** add a factory to `_BUILTIN` in `detectors.py`. Detectors only need `name` and `detect(audio, sample_rate) -> Prediction`.
- **An audio source:** `register(SourceSpec(...))` in `sources.py` (see [Audio sources](audio-sources.md)). Keep train and test pools disjoint.
- **A TTS voice:** add it to `voices.json` under a phoneme-capable model, then run `lkwb voices --check`.
- **A TTS provider:** it must accept inline phonemes. Add its syntax to `phonemes.py`, verify it by synthesizing words from phonemes alone, and add a test pinned to the provider's documented examples.
- **A breakdown axis:** add the metadata column to `POSITIVE_AXES` or `NEGATIVE_AXES` in `bench.py`.
