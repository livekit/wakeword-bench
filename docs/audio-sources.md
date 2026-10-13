# Audio sources

Background noise, room acoustics and real conversation come from public datasets on the Hugging Face Hub. They are split into two pools that never share audio:

- the **train pool** feeds the train and validation splits and `Augment()`;
- the **test pool** feeds the test split only.

A model is never tested on the noise recordings, rooms or real speakers it trained with, so the benchmark measures how well it generalizes. MUSAN and the MIT room recordings are what livekit-wakeword trains on, so they are train-only.

## Sources

| pool | kind | name | content | license |
|---|---|---|---|---|
| train | noise | `musan_noise` | MUSAN noise (Freesound, SoundBible) | CC BY 4.0 |
| train | noise | `dns_noise` | DNS Challenge noise, AudioSet part only (2 shards) | CC BY 4.0 |
| train | room | `mit_rir` | MIT environmental impulse responses (270 rooms) | — |
| train | speech | `librispeech` | LibriSpeech test-clean, read speech | CC BY 4.0 |
| test | noise | `demand` | DEMAND: 17 scenes (kitchen, office, café, station, traffic, bus, park, …) | CC BY 4.0 |
| test | noise | `wham` | WHAM! restaurant and bar babble | CC BY-NC 4.0 |
| test | noise | `esc50` | ESC-50: 50 classes of everyday sounds (folds 1–2) | CC BY-NC |
| test | noise | `gtzan` | GTZAN: music in 10 genres | research use |
| test | room | `slr28_rir` | OpenSLR 28: real rooms (RWCP, REVERB, AIR) and simulated rooms | Apache 2.0 |
| test | room | `treble10_rir` | Treble10: 10 furnished rooms | CC BY-NC-SA 4.0 |
| test | speech | `ami` | AMI meetings, far-field microphone | CC BY 4.0 |
| test | speech | `earnings22` | Earnings-22 earnings calls, many accents | CC BY-SA 4.0 |
| test | speech | `peoples_speech` | People's Speech, mixed web audio | CC BY |

Real speech is used two ways: as `real_speech` negative clips (the main false-alarm audio) and as the competing voice in the "another voice" augmentation. Long recordings are cut into 60-second pieces, and a real speech segment continues through consecutive pieces of the same meeting or call so it sounds like natural conversation.

## Licensing

WHAM!, ESC-50, Treble10 and GTZAN are non-commercial or research-only. That is fine for benchmarking. For a dataset you plan to use commercially, restrict the sources:

```yaml
sources: [musan_noise, dns_noise, mit_rir, librispeech, demand, slr28_rir, ami, earnings22, peoples_speech]
```

The dataset card written by `lkwb generate` lists the sources each dataset used.

## Noise categories

Every noise recording gets a category of the form `<type>/<detail>`, for example `transport/bus` or `babble/restaurant`. The type is a shared vocabulary across datasets: `babble`, `music`, `domestic`, `office`, `street`, `transport`, `nature`, `public`, `animals`, `human`, `mixed`. Clips store both (`noise_type`, `noise_category`), so results can be read at either level.

## Downloading

Sources download on first use. To prefetch them, or to fetch only some:

```bash
uv run lkwb sources                    # all of them (about 4.5 GB)
uv run lkwb sources demand ami         # just these
```

Each source is downloaded once, converted to 16 kHz mono FLAC under `.cache/lkwb/sources/<name>/`, and indexed in a `manifest.jsonl` there. Delete a source's folder to rebuild it.

## Adding a source

Sources are registered in `src/livekit_wakeword_bench/sources.py`. A source has a name, a kind (`noise`, `rir` or `speech`), a pool (`train` or `test`) and a function that yields audio:

```python
register(SourceSpec(
    "my_cafe_noise", "noise", "test",
    _parquet_rows("me/cafe-noise", ["data/test-*.parquet"], "audio",
                  lambda row: (row["id"], "babble/cafe", {})),
    "My café recordings (CC BY 4.0)",
))
```

`_parquet_rows` reads Hugging Face parquet shards; `_audio_files` reads WAV/FLAC files from a repo. The row function returns `(name, category, extra)`; for speech, `extra` can hold `speaker`, `transcript` and `group` (the recording a piece belongs to). Keep the pools disjoint: a recording must never appear in both.
