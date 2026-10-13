# CLI reference

Use this page to look up a command or flag. For a first run, follow [Benchmarking](benchmarking.md) or [Generating datasets](generating-datasets.md).

Run commands as `uv run lkwb <command>` from the repo. Add `--help` for the available options. TTS generation reads LiveKit credentials from `.env.local`; benchmarking existing audio needs no LiveKit credentials.

| command | does |
|---|---|
| [`bench`](#bench) | benchmark one or more detectors |
| [`compare`](#compare) | show what changed between two runs |
| [`report`](#report) | re-render the HTML report from saved runs |
| [`figures`](#figures) | write SVG charts for a saved run |
| [`generate`](#generate) | generate a dataset |
| [`init`](#init) | write a starter config for a wake word |
| [`pronounce`](#pronounce) | show what each TTS provider receives |
| [`knobs`](#knobs) | list every config setting |
| [`verify`](#verify) | check a dataset's audio against its labels |
| [`push`](#push) | upload a dataset to the Hugging Face Hub |
| [`export-livekit`](#export-livekit) | write livekit-wakeword's training layout |
| [`export-wav`](#export-wav) | write WAV files and a metadata CSV |
| [`voices`](#voices) | list or test the TTS voices |
| [`sources`](#sources) | download the noise, room and speech sources |

## bench

```
lkwb bench [SOURCE] -d DETECTOR [-d DETECTOR ...] [options]
```

`SOURCE` is a dataset directory or Hugging Face repo id (default: `binhpham/wakeword-bench-hey-livekit`).

| option | default | meaning |
|---|---|---|
| `-d`, `--detector` | required | detector spec, repeatable: `livekit:model.onnx`, `openwakeword:name`, `module:factory?key=value` |
| `-o`, `--out` | `results` | output directory |
| `--split` | `test` | split to run on |
| `--limit N` | all | random fixed subset of N clips |
| `-f`, `--filter` | | `key=value[,value2]` on clip metadata, repeatable |
| `--threshold` | 0.5 | operating threshold for breakdowns |
| `--score-fpph` | 0.5 | false-alarm target for `score` |
| `--debounce` | 2.0 | seconds between counted triggers |
| `--workers` | 1 | parallel processes (compute timing is only accurate with 1) |
| `--json` | | print machine-readable results only |

See [Benchmarking](benchmarking.md).

## compare

```
lkwb compare BASE_RUN NEW_RUN [--top 10] [--min-n 20] [--json]
```

Headline metric deltas, then the conditions that regressed and improved most. Conditions with fewer than `--min-n` clips are skipped.

## report

```
lkwb report RUN [RUN ...] [-o results/report.html] [--source DATASET] [--split test] [--threshold T]
```

Builds the HTML report from saved runs, side by side when given several. `--source` embeds failure audio; `--threshold` re-scores at another threshold without running the model.

## figures

```
lkwb figures RUN [-o assets] [--prefix NAME_]
```

Writes the README charts for a run as SVG, in dark and light versions.

## generate

```
lkwb generate ["WAKE WORD"] -o OUT [-c CONFIG] [--size tiny|small|default] [--plan build|ship|scale]
              [--set key=value ...] [-s SPLIT ...] [--overwrite]
```

Give a wake word, a config, or both (the wake word overrides the config's). Needs LiveKit credentials.

| option | meaning |
|---|---|
| `-o`, `--out` | output dataset directory |
| `-c`, `--config` | YAML config |
| `--size` | preset split sizes |
| `--plan` | LiveKit plan, sets TTS concurrency |
| `--set` | any knob, repeatable |
| `-s`, `--split` | only these splits |
| `--overwrite` | replace existing shards of the generated splits |

## init

```
lkwb init "WAKE WORD" [--plan build] [--size default] [-o FILE] [--force]
```

Writes `<name>.yaml` with the wake word, guessed pronunciations (marked `TODO` when unknown), phrasing, plan and split sizes. `--size` takes `tiny`, `small` or `default`.

## pronounce

```
lkwb pronounce "TEXT" [-c CONFIG]
```

Prints the ARPAbet for the text and the exact string each TTS provider receives, using the config's `pronunciations`.

## knobs

```
lkwb knobs [-c CONFIG] [--json]
```

Every config key and its value (defaults, or as resolved from the config).

## verify

```
lkwb verify DATASET [--split test] [--limit N]
```

Checks integrity, train/test leakage, levels, SNR, timing and band-limits. Prints a JSON report; `failures` lists any clips that failed a check.

## push

```
lkwb push DATASET_DIR USER/REPO [--public]
```

Uploads the dataset card, config and shards. Private unless `--public`.

## export-livekit

```
lkwb export-livekit DATASET OUT_DIR --model-name NAME
```

Writes `OUT_DIR/NAME/{positive,negative}_{train,test}/clip_NNNNNN.wav` from the train and validation splits, the layout livekit-wakeword's training steps read.

## export-wav

```
lkwb export-wav DATASET OUT_DIR [--split test]
```

Writes `OUT_DIR/<split>/<id>.wav` and `metadata.csv` with every column.

## voices

```
lkwb voices [-m MODEL ...] [--check]
```

Lists voices per model and accent. `--check` synthesizes a phoneme test phrase with each voice (uses `LKWB_TTS_CONCURRENCY`, default 5).

## sources

```
lkwb sources [NAME ...] [--cache-dir .cache/lkwb]
```

Downloads and indexes audio sources (all by default). See [Audio sources](audio-sources.md).
