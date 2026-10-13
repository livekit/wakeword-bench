"""Static SVG charts in the LiveKit Bytes design language (dark canonical, light variant).

Used by the HTML report and `lkwb figures` (README images). No plotting dependencies.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from .bench import THRESHOLDS, BenchResult, wilson

SANS = "'Public Sans', ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "CommitMono, ui-monospace, 'SF Mono', 'Cascadia Mono', Menlo, monospace"
DISPLAY = f"'TWK Everett', {SANS}"

# Bytes tokens (oklch L C h), from @repo/bytes-core tailwind-colors
_TOKENS = {
    "dark": {
        "fg0": (1, 0, 0),
        "fg1": (0.845, 0, 0),
        "fg2": (0.764, 0, 0),
        "fg3": (0.683, 0, 0),
        "fg4": (0.51, 0, 0),
        "bg0": (0, 0, 0),
        "bg1": (0.129, 0, 0),
        "bg2": (0.187, 0, 0),
        "bg3": (0.239, 0, 0),
        "separator1": (0.244, 0, 0),
        "separator2": (0.309, 0.002, 106.29),
        "accent": (0.806, 0.139, 216.34),
        "chart1": (0.806, 0.139, 216.34),
        "chart2": (0.606, 0.288, 312.36),
        "chart3": (0.715, 0.18, 49.51),
        "chart4": (0.754, 0.188, 315.57),
        "chart5": (0.612, 0.162, 151.75),
        "chart8": (0.661, 0.213, 29.95),
        "success": (0.685, 0.163, 154.55),
        "moderate": (0.829, 0.142, 72.57),
    },
    "light": {
        "fg0": (0, 0, 0),
        "fg1": (0.352, 0, 0),
        "fg2": (0.42, 0, 0),
        "fg3": (0.5, 0, 0),
        "fg4": (0.545, 0, 0),
        "bg0": (1, 0, 0),
        "bg1": (0.994, 0.001, 105.51),
        "bg2": (0.951, 0.004, 106.18),
        "bg3": (0.912, 0.004, 106.2),
        "separator1": (0.891, 0.004, 106.21),
        "separator2": (0.798, 0.003, 106.11),
        "accent": (0.462, 0.283, 264.17),
        "chart1": (0.462, 0.283, 264.17),
        "chart2": (0.606, 0.288, 312.36),
        "chart3": (0.715, 0.18, 49.51),
        "chart4": (0.754, 0.188, 315.57),
        "chart5": (0.612, 0.162, 151.75),
        "chart8": (0.661, 0.213, 29.95),
        "success": (0.493, 0.128, 152.53),
        "moderate": (0.511, 0.128, 52.64),
    },
}


def _oklch_hex(lch: tuple[float, float, float]) -> str:
    L, C, h = lch
    a, b = C * math.cos(math.radians(h)), C * math.sin(math.radians(h))
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    rgb = (
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    )

    def enc(x: float) -> int:
        x = min(max(x, 0.0), 1.0)
        x = 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055
        return round(x * 255)

    return "#{:02x}{:02x}{:02x}".format(*(enc(c) for c in rgb))


PALETTE = {theme: {k: _oklch_hex(v) for k, v in t.items()} for theme, t in _TOKENS.items()}
SERIES = ("chart1", "chart2", "chart3", "chart4", "chart5", "chart8")


def _e(x: Any) -> str:
    return html.escape(str(x))


def _svg(w: int, h: int, body: str, c: dict[str, str], bg: bool = True) -> str:
    rect = f'<rect width="{w}" height="{h}" rx="8" fill="{c["bg1"]}"/>' if bg else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{_e(SANS)}">{rect}{body}</svg>'
    )


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.1f}%"


def _caps(x: str) -> str:
    return _e(x.upper())


# ---- scorecard -------------------------------------------------------------------------------
def scorecard(res: BenchResult, theme: str = "dark", w: int = 880) -> str:
    c = PALETTE[theme]
    s = res.summary()
    a = s["at_threshold"]
    tiles = [
        (
            "score",
            _pct(s["score"]),
            f"recall at ≤ {res.cfg.score_fpph:g} false alarms / h",
            c["accent"],
        ),
        ("recall", _pct(a["recall"]), f"at threshold {a['threshold']:g}", c["fg0"]),
        (
            "false alarms / h",
            f"{a['fpph']:.1f}",
            f"at threshold {a['threshold']:g}, {s['fpph_hours']:.1f} h of audio",
            c["fg0"],
        ),
        (
            "confusables accepted",
            _pct(a["adversarial_false_accept_rate"]),
            f"of {s['n_adversarial']:,} near-miss phrases",
            c["fg0"],
        ),
        (
            "latency p50",
            f"{a['latency_ms_p50']:+.0f} ms" if a["latency_ms_p50"] is not None else "–",
            "vs end of wake word",
            c["fg0"],
        ),
        ("compute", f"{s['compute']['real_time_factor']:.2f}×", "real-time factor (CPU)", c["fg0"]),
    ]
    cols, gap, pad = 3, 12, 20
    tw = (w - 2 * pad - (cols - 1) * gap) / cols
    th = 96
    rows = math.ceil(len(tiles) / cols)
    h = pad * 2 + 30 + rows * th + (rows - 1) * gap
    out = [
        f'<text x="{pad}" y="{pad + 14}" fill="{c["fg1"]}" font-size="14" font-weight="600">{_e(res.detector)}</text>',
        f'<text x="{w - pad}" y="{pad + 14}" fill="{c["fg3"]}" font-size="12" text-anchor="end">'
        f"{s['n_positive']:,} wake words · {s['n_negative']:,} negatives · held-out voices &amp; rooms</text>",
    ]
    for i, (label, value, sub, col) in enumerate(tiles):
        x = pad + (i % cols) * (tw + gap)
        y = pad + 30 + (i // cols) * (th + gap)
        out.append(
            f'<rect x="{x:.1f}" y="{y}" width="{tw:.1f}" height="{th}" rx="6" fill="{c["bg2"]}" stroke="{c["separator1"]}"/>'
        )
        out.append(
            f'<text x="{x + 16:.1f}" y="{y + 24}" fill="{c["fg3"]}" font-family="{_e(MONO)}" font-size="11" letter-spacing="0.6">{_caps(label)}</text>'
        )
        out.append(
            f'<text x="{x + 16:.1f}" y="{y + 60}" fill="{col}" font-family="{_e(MONO)}" font-size="30" font-weight="500">{_e(value)}</text>'
        )
        out.append(
            f'<text x="{x + 16:.1f}" y="{y + 82}" fill="{c["fg4"]}" font-size="11.5">{_e(sub)}</text>'
        )
    return _svg(w, int(h), "".join(out), c)


# ---- recall vs false alarms --------------------------------------------------------------------
def tradeoff(results: list[BenchResult], theme: str = "dark", w: int = 880, h: int = 380) -> str:
    c = PALETTE[theme]
    L, R, T, B = 64, 24, 48, 52
    xmin, xmax = 0.1, 1000.0

    def lx(f: float) -> float:
        span = math.log10(xmax) - math.log10(xmin)
        return L + (math.log10(min(max(f, xmin), xmax)) - math.log10(xmin)) / span * (w - L - R)

    ly = lambda r: T + (1 - r) * (h - T - B)  # noqa: E731
    out = [
        f'<text x="{L}" y="26" fill="{c["fg1"]}" font-size="14" font-weight="600">Recall vs false alarms</text>',
        f'<text x="{w - R}" y="26" fill="{c["fg3"]}" font-size="12" text-anchor="end">each step = one threshold · up and left is better</text>',
    ]
    for f in (0.1, 1, 10, 100, 1000):
        out.append(
            f'<line x1="{lx(f):.1f}" y1="{T}" x2="{lx(f):.1f}" y2="{h - B}" stroke="{c["separator1"]}"/>'
        )
        out.append(
            f'<text x="{lx(f):.1f}" y="{h - B + 18}" fill="{c["fg3"]}" font-family="{_e(MONO)}" font-size="11" text-anchor="middle">{f:g}</text>'
        )
    for r in (0, 0.25, 0.5, 0.75, 1):
        out.append(
            f'<line x1="{L}" y1="{ly(r):.1f}" x2="{w - R}" y2="{ly(r):.1f}" stroke="{c["separator1"]}"/>'
        )
        out.append(
            f'<text x="{L - 10}" y="{ly(r) + 4:.1f}" fill="{c["fg3"]}" font-family="{_e(MONO)}" font-size="11" text-anchor="end">{int(r * 100)}%</text>'
        )
    out.append(
        f'<text x="{(L + w - R) / 2}" y="{h - 12}" fill="{c["fg3"]}" font-size="12" text-anchor="middle">false alarms per hour (log)</text>'
    )
    out.append(
        f'<text x="18" y="{(T + h - B) / 2}" fill="{c["fg3"]}" font-size="12" text-anchor="middle" transform="rotate(-90 18 {(T + h - B) / 2})">recall</text>'
    )
    tgt = results[0].cfg.score_fpph if results else 0.5
    out.append(
        f'<line x1="{lx(tgt):.1f}" y1="{T}" x2="{lx(tgt):.1f}" y2="{h - B}" stroke="{c["fg4"]}" stroke-dasharray="4 4"/>'
    )
    out.append(
        f'<text x="{lx(tgt) + 6:.1f}" y="{T + 14}" fill="{c["fg3"]}" font-size="11">target ≤ {tgt:g}/h</text>'
    )
    for i, res in enumerate(results):
        col = c[SERIES[i % len(SERIES)]]
        cv = res.summary()["curve"]
        pts: list[tuple[float, float]] = []
        for f, r in reversed(list(zip(cv["fpph"], cv["recall"]))):
            if f is None or r is None:
                continue
            x, y = lx(f), ly(r)
            if pts:
                pts.append((x, pts[-1][1]))
            pts.append((x, y))
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        s = res.summary()
        a = s["at_threshold"]
        sx, sy = lx(a["fpph"]), ly(a["recall"])
        out.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4.5" fill="{c["bg1"]}" stroke="{col}" stroke-width="2"/>'
        )
        out.append(
            f'<text x="{sx + 9:.1f}" y="{sy + 16:.1f}" fill="{c["fg2"]}" font-family="{_e(MONO)}" font-size="11">thr {a["threshold"]:g}: {_pct(a["recall"])} @ {a["fpph"]:.1f}/h</text>'
        )
        tgt_op = s["at_target_fpph"].get(f"fpph<={tgt}") or s["at_target_fpph"].get(
            f"fpph<={tgt:g}"
        )
        if tgt_op:
            tx, ty = lx(max(tgt_op["fpph"], xmin)), ly(s["score"])
            out.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="5" fill="{col}"/>')
            out.append(
                f'<text x="{tx + 9:.1f}" y="{ty - 8:.1f}" fill="{col}" font-family="{_e(MONO)}" font-size="11.5" font-weight="600">score {_pct(s["score"])} (thr {tgt_op["threshold"]:g})</text>'
            )
        if len(results) > 1:
            out.append(
                f'<rect x="{L + 12}" y="{T + 10 + i * 18}" width="10" height="10" rx="2" fill="{col}"/>'
            )
            out.append(
                f'<text x="{L + 28}" y="{T + 19 + i * 18}" fill="{c["fg2"]}" font-size="11.5">{_e(res.detector)}</text>'
            )
    return _svg(w, h, "".join(out), c)


# ---- where it fails ----------------------------------------------------------------------------
FAILURE_AXES = [
    (
        "interferer",
        "Competing talker",
        {
            "": "none",
            "ami": "meeting (AMI)",
            "earnings22": "earnings call",
            "peoples_speech": "web speech",
        },
    ),
    (
        "text_form",
        "How it's said",
        {
            "isolated": "alone",
            "carrier_prefix": "“Hey LiveKit, …”",
            "carrier_suffix": "“Okay, hey LiveKit”",
        },
    ),
    ("accent", "Speaker accent", {"unknown": "unlabeled"}),
    (
        "noise_bucket",
        "Noise level",
        {
            "clean": "clean",
            "quiet": "quiet 20–30 dB",
            "moderate": "moderate 10–20",
            "noisy": "noisy 5–10",
            "very_noisy": "very noisy 0–5",
            "extreme": "extreme −5–0",
        },
    ),
    ("noise_type", "Noise type", {"": "none"}),
    ("reverb", "Room", {"False": "dry", "True": "reverberant"}),
    ("provider", "TTS provider", {}),
    ("codec", "Codec", {"none": "none"}),
]


def _slices(
    res: BenchResult, min_n: int, significant_only: bool = False
) -> tuple[float, list[tuple[str, list[dict]]]]:
    bd = res.breakdown()
    overall = res.summary()["at_threshold"]["recall"]
    groups = []
    for ax, title, names in FAILURE_AXES:
        rows = [r for r in bd.get(f"positive/{ax}", []) if r["n"] >= min_n]
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda r: -r["recall"])
        if significant_only and not any(
            r["ci95"][1] < overall or r["ci95"][0] > overall for r in rows
        ):
            continue
        for r in rows:
            r["label"] = names.get(r["value"], r["value"] or "none")
        groups.append((title, rows))
    return overall, groups


def failures(
    res: BenchResult,
    theme: str = "dark",
    w: int = 880,
    min_n: int = 50,
    significant_only: bool = False,
) -> str:
    """Recall per slice with 95% CIs. Colored only when the CI excludes the overall recall."""
    c = PALETTE[theme]
    overall, groups = _slices(res, min_n, significant_only)
    L, R, T, row, ghead, gap = 210, 120, 64, 22, 26, 12
    n_rows = sum(len(g[1]) for g in groups)
    h = T + len(groups) * (ghead + gap) + n_rows * row + 24
    x0, x1 = L, w - R
    sx = lambda v: x0 + v * (x1 - x0)  # noqa: E731
    out = [
        f'<text x="24" y="26" fill="{c["fg1"]}" font-size="14" font-weight="600">Where it fails · recall at threshold {res.cfg.threshold:g}</text>',
        f'<text x="24" y="44" fill="{c["fg3"]}" font-size="12">bars = recall, whiskers = 95% CI, colored only when the CI excludes the overall {_pct(overall)}</text>',
    ]
    for v in (0, 0.25, 0.5, 0.75, 1):
        out.append(
            f'<line x1="{sx(v):.1f}" y1="{T - 6}" x2="{sx(v):.1f}" y2="{h - 22}" stroke="{c["separator1"]}"/>'
        )
        out.append(
            f'<text x="{sx(v):.1f}" y="{h - 8}" fill="{c["fg4"]}" font-family="{_e(MONO)}" font-size="10.5" text-anchor="middle">{int(v * 100)}%</text>'
        )
    out.append(
        f'<line x1="{sx(overall):.1f}" y1="{T - 6}" x2="{sx(overall):.1f}" y2="{h - 22}" stroke="{c["fg3"]}" stroke-dasharray="3 3"/>'
    )
    y = T
    for title, rows in groups:
        out.append(
            f'<text x="24" y="{y + 15}" fill="{c["fg2"]}" font-family="{_e(MONO)}" font-size="11" letter-spacing="0.6">{_caps(title)}</text>'
        )
        out.append(
            f'<line x1="24" y1="{y + ghead - 4}" x2="{w - 24}" y2="{y + ghead - 4}" stroke="{c["separator1"]}"/>'
        )
        y += ghead
        for r in rows:
            lo, hi = r["ci95"]
            col = c["chart8"] if hi < overall else c["success"] if lo > overall else c["fg4"]
            cy = y + row / 2
            out.append(
                f'<text x="{L - 12}" y="{cy + 4:.1f}" fill="{c["fg1"]}" font-size="12" text-anchor="end">{_e(r["label"])}</text>'
            )
            out.append(
                f'<rect x="{x0}" y="{cy - 6:.1f}" width="{max(sx(r["recall"]) - x0, 1):.1f}" height="12" rx="3" fill="{col}" fill-opacity="0.85"/>'
            )
            out.append(
                f'<line x1="{sx(lo):.1f}" y1="{cy:.1f}" x2="{sx(hi):.1f}" y2="{cy:.1f}" stroke="{c["fg0"]}" stroke-width="1.2"/>'
            )
            for e in (lo, hi):
                out.append(
                    f'<line x1="{sx(e):.1f}" y1="{cy - 4:.1f}" x2="{sx(e):.1f}" y2="{cy + 4:.1f}" stroke="{c["fg0"]}" stroke-width="1.2"/>'
                )
            out.append(
                f'<text x="{x1 + 10}" y="{cy + 4:.1f}" fill="{col if col != c["fg4"] else c["fg2"]}" font-family="{_e(MONO)}" font-size="12" font-weight="600">{100 * r["recall"]:.0f}%</text>'
            )
            out.append(
                f'<text x="{x1 + 52}" y="{cy + 4:.1f}" fill="{c["fg4"]}" font-family="{_e(MONO)}" font-size="10.5">n={r["n"]}</text>'
            )
            y += row
        y += gap
    return _svg(w, int(h), "".join(out), c)


# ---- false alarms -------------------------------------------------------------------------------
def _which_word(phrase: str, wake: list[str]) -> str | None:
    w = phrase.split()
    if len(w) == len(wake):
        diff = [i for i in range(len(w)) if w[i] != wake[i]]
        if len(diff) == 1:
            return f"“{wake[diff[0]]}” swapped"
        return None
    it = iter(wake)
    if 0 < len(w) < len(wake) and all(x in it for x in w):  # true subsequence
        return "word(s) dropped"
    return None


def false_alarms(res: BenchResult, theme: str = "dark", w: int = 880, top: int = 8) -> str:
    """Left: false alarms / hour by audio source. Right: confusable phrases accepted."""
    import numpy as np

    from .text import _cmu, _split_unknown

    c = PALETTE[theme]
    df = res.frame()
    ti = int(np.argmin(np.abs(THRESHOLDS - res.cfg.threshold)))
    neg = df[(df.label == 0) & (df.category != "adversarial")].copy()
    neg["ev"] = [e[ti] for e in neg.events]
    names = {
        "speech": "TTS everyday speech",
        "real_speech": "real conversation",
        "noise": "background noise",
    }
    src = []
    for cat, g in neg.groupby("category"):
        hrs = g.duration_s.sum() / 3600
        src.append((names.get(cat, cat), g.ev.sum() / max(hrs, 1e-9), hrs))
    src.sort(key=lambda r: -r[1])
    adv = df[df.category == "adversarial"].copy()
    adv["fa"] = [e[ti] > 0 for e in adv.events]
    wake_words = [
        p for wd in str(df.wake_word.iloc[0]).lower().split() for p in _split_unknown(wd, _cmu())
    ]
    adv["kind"] = adv.phrase.map(lambda p: _which_word(p, wake_words))
    kinds = []
    for k, g in adv.dropna(subset=["kind"]).groupby("kind"):
        kinds.append((k, g.fa.mean(), wilson(float(g.fa.sum()), len(g)), len(g)))
    kinds.sort(key=lambda r: -r[1])
    phr = []
    for p, g in adv.groupby("phrase"):
        if len(g) >= 15:
            phr.append((p, g.fa.mean(), wilson(float(g.fa.sum()), len(g)), len(g)))
    phr.sort(key=lambda r: (-r[2][0], -r[1]))
    phr = phr[:top]

    half = (w - 72) / 2
    row = 22
    right_h = sum(10 + len(items) * row + 18 for items in (kinds, phr) if items)
    h = 72 + max(len(src) * 34 + 30, right_h) + 16
    out = [
        f'<text x="24" y="26" fill="{c["fg1"]}" font-size="14" font-weight="600">What makes it fire · threshold {res.cfg.threshold:g}</text>',
        f'<text x="24" y="44" fill="{c["fg3"]}" font-size="12">left: false alarms per hour by audio source · right: share of confusable clips accepted (95% CI)</text>',
    ]
    # left panel
    X, Y = 24, 72
    mx = max([r[1] for r in src] + [1e-9])
    out.append(
        f'<text x="{X}" y="{Y}" fill="{c["fg2"]}" font-family="{_e(MONO)}" font-size="11" letter-spacing="0.6">FALSE ALARMS / HOUR</text>'
    )
    for i, (name, fpph, hrs) in enumerate(src):
        y = Y + 18 + i * 34
        bw = (half - 150) * fpph / mx
        out.append(f'<text x="{X}" y="{y + 12}" fill="{c["fg1"]}" font-size="12">{_e(name)}</text>')
        out.append(
            f'<text x="{X}" y="{y + 26}" fill="{c["fg4"]}" font-size="10.5">{hrs:.1f} h</text>'
        )
        out.append(
            f'<rect x="{X + 130}" y="{y + 4}" width="{max(bw, 1):.1f}" height="14" rx="3" fill="{c["chart3"]}" fill-opacity="0.9"/>'
        )
        out.append(
            f'<text x="{X + 138 + bw:.1f}" y="{y + 15}" fill="{c["fg1"]}" font-family="{_e(MONO)}" font-size="12" font-weight="600">{fpph:.1f}</text>'
        )
    # right panel
    X = 48 + half
    x0, x1 = X + 150, X + half - 56
    sx = lambda v: x0 + v * (x1 - x0)  # noqa: E731
    y = Y
    for title, items in (("WHICH PART IS IGNORED", kinds), ("WORST PHRASES (n ≥ 15)", phr)):
        if not items:
            continue
        out.append(
            f'<text x="{X}" y="{y}" fill="{c["fg2"]}" font-family="{_e(MONO)}" font-size="11" letter-spacing="0.6">{title}</text>'
        )
        y += 10
        for label, rate, (lo, hi), n in items:
            cy = y + row / 2
            out.append(
                f'<text x="{x0 - 10}" y="{cy + 4:.1f}" fill="{c["fg1"]}" font-size="12" text-anchor="end">{_e(label)}</text>'
            )
            out.append(
                f'<rect x="{x0}" y="{cy - 6:.1f}" width="{max(sx(rate) - x0, 1):.1f}" height="12" rx="3" fill="{c["chart8"]}" fill-opacity="0.85"/>'
            )
            out.append(
                f'<line x1="{sx(lo):.1f}" y1="{cy:.1f}" x2="{sx(hi):.1f}" y2="{cy:.1f}" stroke="{c["fg0"]}" stroke-width="1.2"/>'
            )
            out.append(
                f'<text x="{x1 + 8}" y="{cy + 4:.1f}" fill="{c["fg1"]}" font-family="{_e(MONO)}" font-size="12" font-weight="600">{100 * rate:.0f}%</text>'
            )
            y += row
        y += 18
    return _svg(w, int(h), "".join(out), c)


def write_figures(res: BenchResult, out_dir: str | Path, prefix: str = "") -> list[Path]:
    """scorecard / tradeoff / failures / false_alarms as <name>_{dark,light}.svg."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for theme in ("dark", "light"):
        for name, svg in (
            ("scorecard", scorecard(res, theme)),
            ("tradeoff", tradeoff([res], theme)),
            ("failures", failures(res, theme, significant_only=True)),
            ("false_alarms", false_alarms(res, theme)),
        ):
            p = out / f"{prefix}{name}_{theme}.svg"
            p.write_text(svg)
            paths.append(p)
    return paths
