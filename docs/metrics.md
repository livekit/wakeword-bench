# Metrics

A wake word model has two jobs: wake up on the wake word, and stay quiet otherwise. Every number below measures one of those jobs, or the cost of doing them.

## The score

```
score = recall at the lowest threshold where false alarms per hour ≤ 0.5
```

An always-listening device that wakes up by itself more than about once every two hours annoys people, so the benchmark asks: *keeping false alarms at or below 0.5 per hour, how many wake words does the model catch?* Higher is better. Change the target with `--score-fpph` (CLI) or `BenchConfigRun(score_fpph=...)`.

The score is computed from a sweep over 101 thresholds (0.00, 0.01, … 1.00). For each threshold the benchmark measures recall and false alarms per hour; the score is the recall at the lowest threshold that meets the target.

## Catching the wake word

**Recall** is the share of wake word clips (`category = positive`) where the score reaches the threshold between the start of the wake word and one second after it ends. Triggers before the wake word starts do not count as a catch.

**Latency** is the time from the end of the wake word to the first trigger, reported as the median (p50) and 90th percentile (p90) in milliseconds. A negative latency means the model fired before the word finished, which is normal for models that recognize the word early.

Both use exact wake word timing. When the wake word sits inside a sentence ("Hey LiveKit, set a timer"), the segments are synthesized separately and joined, so the start and end of the wake word are known precisely (`phrase_onset_s`, `phrase_offset_s`).

## Staying quiet

**False alarms per hour (FPPH)** counts triggers on audio that has no wake word: real conversation (`real_speech`), synthetic everyday speech (`speech`) and background noise (`noise`). Triggers closer than 2 seconds apart count once (`--debounce`), like a device that ignores repeated triggers while it is already listening.

**False accepts per clip** is the share of those negative clips with at least one trigger.

**Near-misses accepted** (`adversarial_false_accept_rate`) is the share of near-miss clips ("hey live kid", "hey lime kit") that trigger the model. These clips are deliberately hard and short, so they are reported separately and are not part of FPPH.

## Cost

**Real-time factor (RTF)** is the time spent inside `detect()` divided by the length of the audio. 0.2 means the model processes 1 second of audio in 0.2 seconds. Measure it with `--workers 1`.

## Operating points

`summary()` reports the metrics at:

- the **operating threshold** (`--threshold`, default 0.5), used for breakdowns and failure examples;
- each **false-alarm target** (1.0, 0.5 and 0.1 per hour): the lowest threshold that meets it, and the recall there.

## Conditions and confidence intervals

`breakdown()` splits the results by every recorded property of the clips, at the operating threshold.

For wake word clips (recall per value): noise level, noise type, noise category, reverb, room type, competing voice, mic EQ, band-limit, clipping, codec, augmentation combination, input level, speaking rate, phrasing, TTS provider, gender, accent, held-out voice, and individual voice.

For negative clips (false accepts per value, plus FPPH): category, noise level, noise type, noise category, reverb, codec, augmentation combination, provider, phonetic distance to the wake word, and each individual near-miss phrase.

Every value comes with `n` (number of clips) and a 95% Wilson confidence interval `ci95`. Small groups have wide intervals; treat them as anecdotes. Conditions are sorted worst first by the confidence bound rather than the raw value, so a 0% from two clips does not outrank a confidently bad condition with hundreds.

In the report and charts, a condition is colored **only when its confidence interval excludes the overall rate**: red for worse, green for better, gray for no clear difference.

## What the numbers do and do not tell you

- **Wake words are synthetic.** They come from TTS, not people. The differences between conditions are the reliable signal; absolute recall on real users will differ.
- **Rare false alarms need many hours.** The public test set has 6.1 hours of negative audio. Zero false alarms in 6.1 hours only bounds the true rate at ≤ 0.49 per hour (95% confidence). It cannot separate 0.5 from 0.1 per hour.
- **Accent labels** come from TTS provider metadata.
- **Compare like with like.** Compare runs on the same dataset, split and `--limit`.
