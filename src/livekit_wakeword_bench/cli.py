"""`lkwb` command line."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .dataset import DEFAULT_DATASET

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Generate wake word datasets and benchmark any wake word model.",
)
console = Console(soft_wrap=True)


def _env(quiet: bool = False) -> None:
    from dotenv import load_dotenv

    for f in (".env.local", ".env"):
        if Path(f).exists():
            load_dotenv(f)
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("livekit.agents", "httpx", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _require_credentials() -> None:
    import os

    missing = [
        k for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET") if not os.getenv(k)
    ]
    if missing:
        console.print(
            f"[red]Missing {', '.join(missing)}.[/] Generation uses LiveKit Inference TTS.\n"
            "Get credentials with the LiveKit CLI:\n"
            "  lk cloud auth          # link your LiveKit Cloud account (once)\n"
            "  lk app env -w          # writes .env.local from .env.example\n"
            "Benchmarking an existing dataset needs no credentials."
        )
        raise typer.Exit(1)


def _parse_filters(filters: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for f in filters:
        k, _, v = f.partition("=")
        vals = v.split(",")
        conv: list[object] = []
        for x in vals:
            conv.append({"true": True, "false": False}.get(x.lower(), x))
        out[k] = conv if len(conv) > 1 else conv[0]
    return out


@app.command(rich_help_panel="Generate data")
def init(
    wake_word: str = typer.Argument(..., help='e.g. "hey jarvis"'),
    plan: str = typer.Option("build", help="LiveKit plan: build (5) / ship (20) / scale (50)"),
    size: str = typer.Option("default", help="tiny / small / default split sizes"),
    out: Path | None = typer.Option(None, "--out", "-o", help="Config path (default: <name>.yaml)"),
    force: bool = typer.Option(False, help="Overwrite an existing file"),
) -> None:
    """Write a config file for a new wake word."""
    import re

    from .scaffold import config_yaml, guess_pronunciations, preview

    name = re.sub(r"[^a-z0-9]+", "_", wake_word.lower()).strip("_")
    path = out or Path(f"{name}.yaml")
    if path.exists() and not force:
        console.print(f"[red]{path} exists[/] (use --force)")
        raise typer.Exit(1)
    path.write_text(config_yaml(wake_word, plan, size))
    guesses, unknown = guess_pronunciations(wake_word)
    console.print(f"wrote [bold]{path}[/]")
    if guesses:
        console.print(f"guessed pronunciations (check them): {guesses}")
    if unknown:
        console.print(f"[yellow]fill in pronunciations for: {unknown}[/]")
    else:
        for prov, txt in preview(wake_word, guesses).items():
            console.print(f"  {prov:9s} {txt}")
    console.print(f"next: lkwb generate -c {path} -o out/{name}")


@app.command(rich_help_panel="Generate data")
def knobs(
    config: Path | None = typer.Option(None, "--config", "-c", help="Show values from this config"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """List every setting and its value."""
    from .config import load_config

    def flat(d: dict, prefix: str = "") -> dict:
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if (
                isinstance(v, dict)
                and v
                and k not in ("snr_buckets", "snr_weights", "pronunciations")
            ):
                out.update(flat(v, key + "."))
            else:
                out[key] = v
        return out

    values = flat(load_config(config).model_dump(mode="json"))
    if as_json:
        print(json.dumps(values, indent=2))
        return
    for k, v in values.items():
        console.print(f"{k:40s} {json.dumps(v)}")


@app.command(rich_help_panel="Generate data")
def pronounce(
    text: str = typer.Argument(..., help="Phrase to render"),
    config: Path | None = typer.Option(None, "--config", "-c", help="Use its pronunciations"),
) -> None:
    """Show how each TTS provider will pronounce a phrase."""
    from .config import load_config
    from .phonemes import arpabet_string, phrase_parts, validate_arpabet
    from .scaffold import guess_pronunciations, preview

    overrides = load_config(config).pronunciations if config else {}
    bad = {w: e for w, p in overrides.items() if (e := validate_arpabet(p))}
    if bad:
        console.print(f"[red]invalid pronunciations: {bad}[/]")
        raise typer.Exit(1)
    if phrase_parts(text, {k.lower(): v for k, v in overrides.items()}) is None:
        guesses, unknown = guess_pronunciations(text)
        console.print(
            f"[red]unknown words: {unknown or list(guesses)}[/] add them to `pronunciations`"
        )
        raise typer.Exit(1)
    console.print(f"ARPAbet   {arpabet_string(text, {k.lower(): v for k, v in overrides.items()})}")
    for prov, txt in preview(text, overrides).items():
        console.print(f"{prov:9s} {txt}")


@app.command(rich_help_panel="Generate data")
def generate(
    wake_word: str | None = typer.Argument(None, help='e.g. "hey jarvis" (or use --config)'),
    out: Path = typer.Option(..., "--out", "-o", help="Output dataset directory"),
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config"),
    size: str | None = typer.Option(None, help="tiny / small / default split sizes"),
    plan: str | None = typer.Option(None, help="LiveKit plan: build (5) / ship (20) / scale (50)"),
    knobs: list[str] = typer.Option([], "--set", help="Override any config key: a.b.c=value"),
    splits: list[str] = typer.Option([], "--split", "-s", help="Only these splits"),
    overwrite: bool = typer.Option(False, help="Replace existing shards"),
) -> None:
    """Generate a dataset for a wake word (needs LiveKit credentials)."""
    _env()
    _require_credentials()
    from .config import make_config
    from .generate import Generator

    if not wake_word and not config:
        console.print('[red]give a wake word ("hey jarvis") or --config[/]')
        raise typer.Exit(2)
    try:
        cfg = make_config(wake_word, config, size, plan, knobs)
    except (ValueError, KeyError) as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e
    stats = Generator(cfg, out).run(splits=splits or None, overwrite=overwrite)
    console.print_json(json.dumps(stats))


@app.command(rich_help_panel="Dataset tools")
def push(
    root: Path = typer.Argument(..., help="Generated dataset directory"),
    repo_id: str = typer.Argument(..., help="e.g. livekit/wakeword-bench-hey-livekit"),
    public: bool = typer.Option(False, help="Create a public repo"),
) -> None:
    """Upload a dataset to the Hugging Face Hub."""
    from .card import push_to_hub

    console.print(push_to_hub(root, repo_id, private=not public))


@app.command("media", rich_help_panel="Benchmark")
def media_cmd(
    out: Path = typer.Option(Path("out/media"), "--out", "-o", help="Output dataset directory"),
    hours: float = typer.Option(
        1.0, help="Scale every source's hours (1.0 = the ~24 h default mix)"
    ),
) -> None:
    """Build the long-form media negatives (TV, music, speeches)."""
    _env()
    from .media import DEFAULT_HOURS, build_media

    path = build_media(out, {k: v * hours for k, v in DEFAULT_HOURS.items()})
    console.print(f"media negatives: {path}")


@app.command(rich_help_panel="Benchmark")
def bench(
    models: list[str] = typer.Argument(
        ...,
        help="livekit-wakeword .onnx files, or detector specs (module:factory, openwakeword:name)",
    ),
    source: str = typer.Option(DEFAULT_DATASET, "--data", help="Dataset dir or HF repo id"),
    out: Path = typer.Option(Path("results"), "--out", "-o"),
    split: str = typer.Option("test", "--split"),
    threshold: float = typer.Option(0.5, help="Operating threshold for breakdowns"),
    score_fpph: float = typer.Option(0.5, help="`score` = recall at FPPH <= this"),
    debounce: float = typer.Option(2.0, help="Seconds between counted triggers"),
    workers: int = typer.Option(1, help="Parallel processes (RTF is only accurate with 1)"),
    limit: int | None = typer.Option(None, help="Random subset size (quick runs)"),
    filters: list[str] = typer.Option([], "--filter", "-f", help="key=value[,value2]"),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable results only"),
    media: Path | None = typer.Option(
        None, "--media", help="Long-form negatives set (`lkwb media`): false alarms per hour"
    ),
) -> None:
    """Benchmark models on a dataset."""
    _env(quiet=as_json)
    from .bench import BenchConfigRun, add_media, evaluate, run_detector
    from .detectors import load_detector
    from .loader import WakeWordData
    from .report import render_report

    data = WakeWordData(source, split=split)
    if filters:
        data = data.filter(**_parse_filters(filters))
    if limit:
        data = data.shuffle(0).take(limit)
    cfg = BenchConfigRun(debounce_s=debounce, threshold=threshold, score_fpph=score_fpph)
    results = []
    for spec in models:
        det = load_detector(spec)
        name = det.name
        if not as_json:
            console.print(f"[bold]{name}[/] on {len(data)} clips")
        run = det if workers <= 1 else spec  # worker processes load their own copy
        raw = run_detector(run, data, source=source, workers=workers, progress=not as_json)
        res = evaluate(raw, data, name, cfg)
        if media is not None:
            add_media(res, run, data, media, workers=workers, progress=not as_json)
        res.save(out / _safe(name))
        results.append(res)
        if not as_json:
            _print_summary(res)
    path = render_report(
        results, out / "report.html", data=data, title=f"Wake word benchmark · {source}"
    )
    if as_json:
        print(json.dumps([_agent_view(r, out / _safe(r.detector)) for r in results], indent=2))
    else:
        console.print(f"report: [link=file://{path.resolve()}]{path}[/link]")


@app.command(rich_help_panel="Benchmark")
def compare(
    base: Path = typer.Argument(..., help="Baseline run dir (results/<model>)"),
    new: Path = typer.Argument(..., help="Candidate run dir"),
    top: int = typer.Option(10, help="Slices to show per direction"),
    min_n: int = typer.Option(20, help="Ignore slices smaller than this"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Compare two benchmark runs, condition by condition."""
    from .bench import BenchResult
    from .bench import compare as do_compare

    c = do_compare(BenchResult.load(base), BenchResult.load(new), min_n=min_n)
    if as_json:
        print(json.dumps({**c, "slices": c["slices"][:top] + c["slices"][-top:]}, indent=2))
        return
    for k, v in c["headline"].items():
        d = v["delta"]
        console.print(
            f"{k:32s} {v['base']!s:>10.10} -> {v['new']!s:>10.10}  ({'' if d is None else f'{d:+.4f}'})"
        )
    console.print("\n[red]regressions[/]")
    for r in c["slices"][:top]:
        console.print(f"  {r['slice']:50s} {r['metric']:18s} {r['base']:.3f} -> {r['new']:.3f}")
    console.print("[green]improvements[/]")
    for r in c["slices"][::-1][:top]:
        console.print(f"  {r['slice']:50s} {r['metric']:18s} {r['base']:.3f} -> {r['new']:.3f}")


def _safe(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def _agent_view(res: object, run_dir: Path, worst: int = 5) -> dict:
    """Compact summary for agents: headline metrics + the worst slices per axis."""
    s = res.summary()  # type: ignore[attr-defined]
    s.pop("curve", None)
    bd = res.breakdown()  # type: ignore[attr-defined]
    s["worst_slices"] = {
        k: [
            {
                kk: r[kk]
                for kk in r
                if kk in ("value", "n", "recall", "false_accept_rate", "fpph", "ci95")
            }
            for r in rows[:worst]
        ]
        for k, rows in bd.items()
    }
    s["run_dir"] = str(run_dir)
    return s


@app.command(rich_help_panel="Benchmark")
def figures(
    run: Path = typer.Argument(..., help="Saved run dir (results/<model>)"),
    out: Path = typer.Option(Path("assets"), "--out", "-o"),
    prefix: str = typer.Option("", help="Filename prefix"),
) -> None:
    """Write SVG charts for a saved run."""
    from .bench import BenchResult
    from .viz import write_figures

    for p in write_figures(BenchResult.load(run), out, prefix):
        console.print(p)


@app.command(rich_help_panel="Benchmark")
def report(
    runs: list[Path] = typer.Argument(..., help="Saved run directories"),
    out: Path = typer.Option(Path("results/report.html"), "--out", "-o"),
    source: str | None = typer.Option(None, help="Dataset (for failure audio)"),
    split: str = typer.Option("test"),
    threshold: float | None = typer.Option(None, help="Re-threshold without re-running"),
) -> None:
    """Rebuild the HTML report from saved runs."""
    from .bench import BenchResult
    from .loader import WakeWordData
    from .report import render_report

    results = [BenchResult.load(r) for r in runs]
    if threshold is not None:
        for r in results:
            r.cfg.threshold = threshold
    data = WakeWordData(source, split=split) if source else None
    console.print(render_report(results, out, data=data))


def _print_summary(res: object) -> None:
    s = res.summary()  # type: ignore[attr-defined]
    a = s["at_threshold"]
    t = Table(show_header=True, header_style="bold")
    for col in ("thr", "recall", "FPPH", "FA/clip", "adv FA", "lat p50 ms", "RTF"):
        t.add_column(col, justify="right")
    t.add_row(
        f"{a['threshold']:.2f}",
        f"{a['recall']:.3f}",
        f"{a['fpph']:.2f}",
        f"{a['false_accept_rate']:.3f}",
        f"{a['adversarial_false_accept_rate']:.3f}",
        str(a["latency_ms_p50"]),
        f"{s['compute']['real_time_factor']:.4f}",
    )
    console.print(t)
    console.print(f"  FPPH measured on {s['fpph_hours']} h of ambient negatives")
    for k, v in s["at_target_fpph"].items():
        if v:
            console.print(f"  {k}: recall {v['recall']:.3f} @ thr {v['threshold']}")
    pt = s.get("party")
    if pt and pt["recall"]["at_threshold"] is not None:
        far = pt["false_accept_rate"]["at_threshold"]
        console.print(
            f"  party ({pt['n_positive']} wake words, {pt['n_negative']} negatives among 2-4 "
            f"talkers + babble): recall {pt['recall']['at_threshold']:.3f}"
            + (f", false accepts {far:.3f}" if far is not None else "")
            + f" @ thr {a['threshold']:.2f}"
        )
    m = s.get("media")
    if m:
        kinds = ", ".join(f"{k} {v['at_threshold']:.2f}" for k, v in m["by_type"].items())
        at_score = m["at_score_threshold"]
        console.print(
            f"  media ({m['hours']} h): {m['at_threshold']:.2f} false alarms/h @ thr "
            f"{a['threshold']:.2f} ({kinds})"
            + (f"; {at_score:.2f} @ the score threshold" if at_score is not None else "")
        )


@app.command("export-livekit", rich_help_panel="Dataset tools")
def export_livekit(
    source: str = typer.Argument(...),
    out: Path = typer.Argument(..., help="livekit-wakeword output_dir"),
    model_name: str = typer.Option(..., help="livekit-wakeword model_name"),
) -> None:
    """Export train/validation clips for livekit-wakeword training."""
    from .export import export_livekit_wakeword

    console.print_json(json.dumps(export_livekit_wakeword(source, out, model_name)))


@app.command("export-wav", rich_help_panel="Dataset tools")
def export_wav(
    source: str = typer.Argument(...),
    out: Path = typer.Argument(...),
    split: str = typer.Option("test"),
) -> None:
    """Export WAV files and a metadata CSV."""
    from .export import export_wav_folder

    console.print(export_wav_folder(source, out, split))


@app.command(rich_help_panel="Generate data")
def voices(
    check: bool = typer.Option(False, help="Synthesize a test phrase with every voice"),
    models: list[str] = typer.Option([], "--model", "-m"),
) -> None:
    """List or test the TTS voices."""
    _env()
    from collections import Counter

    from .voices import load_voices

    vs = load_voices(models=models or None)
    for m, n in Counter(v.model for v in vs).items():
        console.print(f"{m}: {n} voices")
    console.print("accents:", dict(Counter(v.accent for v in vs)))
    if check:
        import asyncio
        import os

        from .phonemes import provider_options, render_phrase
        from .tts import Synthesizer, TTSRequest

        async def run() -> list[object]:
            conc = int(os.getenv("LKWB_TTS_CONCURRENCY", "5"))
            async with Synthesizer(Path(".cache/lkwb"), conc, retries=2) as s:
                reqs = [
                    TTSRequest(
                        v.model,
                        v.id,
                        "Testing one two " + (render_phrase("three", v.model) or "three") + ".",
                        extra=provider_options(v.model),
                    )
                    for v in vs
                ]
                return await asyncio.gather(*(s.synthesize(r) for r in reqs))

        res = asyncio.run(run())
        bad = [v.key for v, r in zip(vs, res) if r is None]
        console.print(f"ok {len(vs) - len(bad)}, failed {len(bad)}", bad)


@app.command(rich_help_panel="Dataset tools")
def verify(
    source: str = typer.Argument(..., help="Dataset directory or HF repo id"),
    split: str = typer.Option("test", "--split"),
    limit: int | None = typer.Option(None),
) -> None:
    """Check a dataset's audio against its labels."""
    _env()
    from .verify import verify as run_verify

    rep = run_verify(source, split, limit=limit)
    console.print_json(json.dumps(rep, default=str))


@app.command(rich_help_panel="Dataset tools")
def sources(
    names: list[str] = typer.Argument(None, help="Source names (default: all)"),
    cache_dir: Path = typer.Option(Path(".cache/lkwb")),
) -> None:
    """Download the noise, room and speech sources."""
    _env()
    from .sources import SOURCES, prepare_source

    for name in names or list(SOURCES):
        items = prepare_source(name, cache_dir)
        console.print(f"{name}: {len(items)} items")


_ORDER = [
    "bench",
    "compare",
    "report",
    "figures",
    "media",
    "generate",
    "init",
    "pronounce",
    "knobs",
    "voices",
    "verify",
    "push",
    "export-livekit",
    "export-wav",
    "sources",
]
app.registered_commands.sort(
    key=lambda c: _ORDER.index(c.name or c.callback.__name__.replace("_", "-"))
)


if __name__ == "__main__":
    app()
