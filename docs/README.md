# Documentation

Each page covers one part of LiveKit WakeWord Bench. Start with the guide that matches what you want to do.

| I want to… | Read |
|---|---|
| Measure how good a wake word model is | [Benchmarking](benchmarking.md) |
| Understand the score and every other number | [Metrics](metrics.md) |
| Make a dataset for my own wake word | [Generating datasets](generating-datasets.md) |
| Control exactly how the wake word is pronounced | [Phoneme steering](phonemes.md) |
| Understand or tune the noise, rooms and other effects | [Augmentation](augmentation.md) |
| Know where the noise, rooms and real speech come from | [Audio sources](audio-sources.md) |
| Read the dataset files with my own tools | [Dataset format](dataset-format.md) |
| Feed the data into a training loop | [Loading data](loading-data.md) |
| See every setting and how to change it | [Configuration](configuration.md) |
| Look up a command | [CLI reference](cli.md) |
| Hack on this repo | [Development](development.md) |

Coding agents should also read [AGENTS.md](../AGENTS.md), which describes the benchmark → train → compare loop.

## How the pieces fit

```
lkwb generate ──► dataset (parquet: audio + recipe per clip) ──► lkwb bench ──► results/<model>/
      ▲                       │                                       │
      │                       └──► WakeWordData / Augment ──► your training loop
      │                                                               │
      └──────────── generate more of what fails ◄── lkwb compare ◄────┘
```

1. **Generate** speech with LiveKit Inference TTS, steered with phonemes, then mix in real noise, rooms, other talkers and device effects. Every clip records how it was made.
2. **Benchmark** any detector on a dataset: the public "Hey LiveKit" set or your own.
3. **Read the breakdown** to find the conditions where recall drops or false alarms rise.
4. **Generate training data** for those conditions, train, and **compare** the new model against the old one on the same test set.
