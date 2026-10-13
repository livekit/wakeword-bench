# Documentation

Suppose your model catches a wake word in a quiet room but misses it when someone else is talking. You need to find out how often that happens, build training data that addresses it, and check whether the next model does better.

These guides follow that workflow. Start with [Benchmarking](benchmarking.md) if you already have a model. If you need data for a new wake word, start with [Generating datasets](generating-datasets.md).

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

## From a failure to the next model

1. **Run a baseline.** Benchmark your model on the public “Hey LiveKit” dataset or your own test set. Keep the run so you can compare against it later.
2. **Find a condition worth investigating.** Read the breakdown and listen to failed clips. A large drop across hundreds of clips is more useful than a bad result from two.
3. **Make training examples for it.** Increase competing speech, loud noise or another relevant effect in the training split. Keep test voices, rooms and recordings out of training.
4. **Train and compare.** Run the candidate on the same test set and compare both the score and the conditions. Check whether the fix helped elsewhere or introduced regressions.

The benchmark and generator support this loop; you supply the model and training code.
