"""Static SVG charts in the LiveKit Bytes design language (dark canonical, light variant).

Charts carry almost no text: labels, values, axes. Explanations belong in captions.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import numpy as np

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


def _svg(w: int, h: int, body: str, c: dict[str, str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{_e(SANS)}"><rect width="{w}" height="{h}" fill="{c["bg1"]}"/>{body}</svg>'
    )


def _text(
    x: float,
    y: float,
    s: str,
    c: str,
    size: float = 12,
    anchor: str = "start",
    mono: bool = False,
    weight: int = 400,
    caps: bool = False,
) -> str:
    fam = f' font-family="{_e(MONO)}"' if mono or caps else ""
    ls = ' letter-spacing="0.7"' if caps else ""
    s = s.upper() if caps else s
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{c}" font-size="{size}" text-anchor="{anchor}"'
        f' font-weight="{weight}"{fam}{ls}>{_e(s)}</text>'
    )


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def _ci_bar(
    x0: float, sx: Any, cy: float, v: float, lo: float, hi: float, col: str, c: dict
) -> str:
    out = f'<rect x="{x0}" y="{cy - 7:.1f}" width="{max(sx(v) - x0, 1):.1f}" height="14" fill="{col}"/>'
    out += f'<line x1="{sx(lo):.1f}" y1="{cy:.1f}" x2="{sx(hi):.1f}" y2="{cy:.1f}" stroke="{c["fg0"]}" stroke-width="1.3"/>'
    for e in (lo, hi):
        out += f'<line x1="{sx(e):.1f}" y1="{cy - 4:.1f}" x2="{sx(e):.1f}" y2="{cy + 4:.1f}" stroke="{c["fg0"]}" stroke-width="1.3"/>'
    return out


def _tone(lo: float, hi: float, ref: float, c: dict, good_high: bool = True) -> str:
    if hi < ref:
        return c["chart8"] if good_high else c["success"]
    if lo > ref:
        return c["success"] if good_high else c["chart8"]
    return c["fg4"]


# ---- teaser: one clip, with and without a second voice ----------------------------------------
def traces(examples: list[dict[str, Any]], thr: float, theme: str = "dark", w: int = 520) -> str:
    """examples: [{"label", "audio", "sr", "times", "scores"}]; waveform + score per row."""
    c = PALETTE[theme]
    row_h, top, L, R = 120, 16, 16, 16
    h = top + len(examples) * row_h + 26
    out = []
    T = max(len(e["audio"]) / e["sr"] for e in examples)
    sx = lambda t: L + t / T * (w - L - R)  # noqa: E731
    for i, ex in enumerate(examples):
        y0 = top + i * row_h
        hit = bool(np.any(ex["scores"] >= thr))
        col = c["success"] if hit else c["chart8"]
        out.append(_text(L, y0 + 12, ex["label"], c["fg1"], 13, weight=600))
        out.append(
            _text(w - R, y0 + 12, "detected" if hit else "missed", col, 12, "end", caps=True)
        )
        a = ex["audio"]
        n = 240
        seg = np.array_split(np.abs(a), n)
        peak = max(float(np.abs(a).max()), 1e-6)
        mid, amp = y0 + 52, 18
        for j, s in enumerate(seg):
            v = float(s.max()) / peak * amp if len(s) else 0
            x = sx(j / n * T)
            out.append(
                f'<line x1="{x:.1f}" y1="{mid - v:.1f}" x2="{x:.1f}" y2="{mid + v:.1f}" stroke="{c["fg4"]}" stroke-width="1"/>'
            )
        sy = lambda s, y0=y0: y0 + 106 - s * 36  # noqa: E731
        out.append(
            f'<line x1="{L}" y1="{sy(thr):.1f}" x2="{w - R}" y2="{sy(thr):.1f}" stroke="{c["fg3"]}" stroke-dasharray="3 3"/>'
        )
        pts = " ".join(
            f"{sx(float(t)):.1f},{sy(float(s)):.1f}" for t, s in zip(ex["times"], ex["scores"])
        )
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"/>')
        k = int(np.argmax(ex["scores"]))
        out.append(
            f'<circle cx="{sx(float(ex["times"][k])):.1f}" cy="{sy(float(ex["scores"][k])):.1f}" r="3.5" fill="{col}"/>'
        )
    out.append(
        _text(w - R, h - 8, f"score  ···  threshold {thr:g}", c["fg3"], 11, "end", mono=True)
    )
    return _svg(w, h, "".join(out), c)


def recall_by_talker(res: BenchResult, theme: str = "dark", w: int = 340) -> str:
    """Recall vs competing-talker loudness (SIR bins)."""
    c = PALETTE[theme]
    df = res.frame()
    ti = int(np.argmin(np.abs(THRESHOLDS - res.cfg.threshold)))
    pos = df[df.label == 1].copy()
    pos["hit"] = [d[ti] for d in pos.detected]
    groups = [("none", pos[pos.interferer == ""])]
    tk = pos[pos.interferer != ""]
    for lo, hi in ((15, 20), (10, 15), (5, 10), (0, 5)):
        groups.append((f"{lo}–{hi}", tk[(tk.sir_db >= lo) & (tk.sir_db < hi)]))
    L, R, T, B = 40, 12, 24, 44
    h = 282
    bw = (w - L - R) / len(groups)
    sy = lambda v: T + (1 - v) * (h - T - B)  # noqa: E731
    out = []
    for v in (0, 0.5, 1):
        out.append(
            f'<line x1="{L}" y1="{sy(v):.1f}" x2="{w - R}" y2="{sy(v):.1f}" stroke="{c["separator1"]}"/>'
        )
        out.append(_text(L - 6, sy(v) + 4, f"{int(v * 100)}%", c["fg3"], 10.5, "end", mono=True))
    for i, (name, g) in enumerate(groups):
        if not len(g):
            continue
        r = float(g.hit.mean())
        lo, hi = wilson(float(g.hit.sum()), len(g))
        x = L + i * bw + bw * 0.18
        col = c["accent"] if i == 0 else c["chart8"]
        out.append(
            f'<rect x="{x:.1f}" y="{sy(r):.1f}" width="{bw * 0.64:.1f}" height="{sy(0) - sy(r):.1f}" fill="{col}"/>'
        )
        cx = x + bw * 0.32
        out.append(
            f'<line x1="{cx:.1f}" y1="{sy(lo):.1f}" x2="{cx:.1f}" y2="{sy(hi):.1f}" stroke="{c["fg0"]}" stroke-width="1.3"/>'
        )
        out.append(_text(cx, sy(hi) - 6, _pct(r), c["fg0"], 12, "middle", mono=True, weight=600))
        out.append(_text(cx, h - B + 16, name, c["fg2"], 11, "middle", mono=True))
    out.append(
        _text(
            (L + w - R) / 2, h - 8, "second voice: dB quieter than speaker", c["fg3"], 11, "middle"
        )
    )
    return _svg(w, h, "".join(out), c)


# ---- headline numbers ------------------------------------------------------------------------
def scorecard(res: BenchResult, theme: str = "dark", w: int = 880) -> str:
    c = PALETTE[theme]
    s = res.summary()
    a = s["at_threshold"]
    tiles = [
        ("score", f"{100 * s['score']:.1f}%", c["accent"]),
        ("recall", f"{100 * a['recall']:.1f}%", c["fg0"]),
        ("false alarms / h", f"{a['fpph']:.1f}", c["fg0"]),
        ("near-misses accepted", f"{100 * a['adversarial_false_accept_rate']:.0f}%", c["fg0"]),
        (
            "latency p50",
            f"{a['latency_ms_p50']:+.0f} ms" if a["latency_ms_p50"] is not None else "–",
            c["fg0"],
        ),
        ("compute", f"{s['compute']['real_time_factor']:.2f}× RT", c["fg0"]),
    ]
    cols, gap = 3, 1
    tw = (w - (cols - 1) * gap) / cols
    th = 84
    rows = math.ceil(len(tiles) / cols)
    h = rows * th + (rows - 1) * gap
    out = [f'<rect width="{w}" height="{h}" fill="{c["separator1"]}"/>']
    for i, (label, value, col) in enumerate(tiles):
        x = (i % cols) * (tw + gap)
        y = (i // cols) * (th + gap)
        out.append(f'<rect x="{x:.1f}" y="{y}" width="{tw:.1f}" height="{th}" fill="{c["bg1"]}"/>')
        out.append(_text(x + 18, y + 26, label, c["fg3"], 10.5, caps=True))
        out.append(_text(x + 18, y + 64, value, col, 28, mono=True, weight=500))
    return _svg(w, int(h), "".join(out), c)


# ---- recall vs false alarms ------------------------------------------------------------------
def tradeoff(results: list[BenchResult], theme: str = "dark", w: int = 880, h: int = 340) -> str:
    c = PALETTE[theme]
    L, R, T, B = 56, 24, 24, 48
    xmin, xmax = 0.1, 1000.0

    def lx(f: float) -> float:
        span = math.log10(xmax) - math.log10(xmin)
        return L + (math.log10(min(max(f, xmin), xmax)) - math.log10(xmin)) / span * (w - L - R)

    def ly(r: float) -> float:
        return T + (1 - r) * (h - T - B)

    out = []
    for f in (0.1, 1, 10, 100, 1000):
        out.append(
            f'<line x1="{lx(f):.1f}" y1="{T}" x2="{lx(f):.1f}" y2="{h - B}" stroke="{c["separator1"]}"/>'
        )
        out.append(_text(lx(f), h - B + 18, f"{f:g}", c["fg3"], 11, "middle", mono=True))
    for r in (0, 0.5, 1):
        out.append(
            f'<line x1="{L}" y1="{ly(r):.1f}" x2="{w - R}" y2="{ly(r):.1f}" stroke="{c["separator1"]}"/>'
        )
        out.append(_text(L - 8, ly(r) + 4, f"{int(r * 100)}%", c["fg3"], 11, "end", mono=True))
    out.append(_text((L + w - R) / 2, h - 10, "false alarms per hour", c["fg3"], 12, "middle"))
    ym = (T + h - B) / 2
    out.append(
        f'<text transform="rotate(-90 16 {ym})" x="16" y="{ym}" fill="{c["fg3"]}" font-size="12" text-anchor="middle">recall</text>'
    )
    tgt = results[0].cfg.score_fpph if results else 0.5
    out.append(
        f'<rect x="{L}" y="{T}" width="{lx(tgt) - L:.1f}" height="{h - T - B}" fill="{c["accent"]}" fill-opacity="0.07"/>'
    )
    out.append(
        f'<line x1="{lx(tgt):.1f}" y1="{T}" x2="{lx(tgt):.1f}" y2="{h - B}" stroke="{c["fg3"]}" stroke-dasharray="3 3"/>'
    )
    out.append(_text(lx(tgt) + 6, T + 16, f"≤ {tgt:g}/h", c["fg3"], 11, mono=True))
    for i, res in enumerate(results):
        col = c[SERIES[i % len(SERIES)]]
        s = res.summary()
        cv = s["curve"]
        pts: list[tuple[float, float]] = []
        for f, r in reversed(list(zip(cv["fpph"], cv["recall"]))):
            if f is None or r is None:
                continue
            x, y = lx(f), ly(r)
            if pts:
                pts.append((x, pts[-1][1]))
            pts.append((x, y))
        out.append(
            f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{col}" stroke-width="2"/>'
        )
        a = s["at_threshold"]
        out.append(
            f'<circle cx="{lx(a["fpph"]):.1f}" cy="{ly(a["recall"]):.1f}" r="4" fill="{c["bg1"]}" stroke="{col}" stroke-width="2"/>'
        )
        out.append(
            _text(
                lx(a["fpph"]) + 8,
                ly(a["recall"]) + 16,
                f"{_pct(a['recall'])} at {a['fpph']:.0f}/h",
                c["fg2"],
                11.5,
                mono=True,
            )
        )
        out.append(f'<circle cx="{lx(xmin):.1f}" cy="{ly(s["score"]):.1f}" r="5" fill="{col}"/>')
        out.append(
            _text(
                lx(xmin) + 10,
                ly(s["score"]) - 8,
                f"{100 * s['score']:.1f}%",
                col,
                13,
                mono=True,
                weight=600,
            )
        )
        if len(results) > 1:
            out.append(
                f'<rect x="{w - R - 180}" y="{h - B - 18 - i * 18}" width="10" height="10" fill="{col}"/>'
            )
            out.append(_text(w - R - 164, h - B - 9 - i * 18, res.detector, c["fg2"], 11.5))
    return _svg(w, h, "".join(out), c)


# ---- where it fails ------------------------------------------------------------------------------
FAILURE_AXES = [
    (
        "interferer",
        "Second voice",
        {"": "none", "ami": "meeting", "earnings22": "phone call", "peoples_speech": "web audio"},
    ),
    (
        "text_form",
        "Phrasing",
        {
            "isolated": "“Hey LiveKit”",
            "carrier_prefix": "“Hey LiveKit, …”",
            "carrier_suffix": "“Okay, hey LiveKit”",
        },
    ),
    ("accent", "Accent", {"unknown": "unlabeled"}),
    (
        "noise_bucket",
        "Noise",
        {
            "clean": "none",
            "quiet": "20–30 dB",
            "moderate": "10–20 dB",
            "noisy": "5–10 dB",
            "very_noisy": "0–5 dB",
            "extreme": "−5–0 dB",
        },
    ),
    ("noise_type", "Noise type", {"": "none"}),
    ("reverb", "Room", {"False": "dry", "True": "reverberant"}),
    ("provider", "Voice provider", {}),
    ("codec", "Codec", {"none": "none"}),
]


def _slices(
    res: BenchResult, min_n: int, axes: list[str] | None, significant_only: bool
) -> tuple[float, list]:
    bd = res.breakdown()
    overall = res.summary()["at_threshold"]["recall"]
    groups = []
    for ax, title, names in FAILURE_AXES:
        if axes is not None and ax not in axes:
            continue
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
    axes: list[str] | None = None,
    significant_only: bool = False,
) -> str:
    """Recall per slice, 95% CI whiskers; colored only when the CI excludes the overall recall."""
    c = PALETTE[theme]
    overall, groups = _slices(res, min_n, axes, significant_only)
    L, R, T, row, ghead, gap = 190, 64, 16, 24, 28, 10
    h = T + len(groups) * (ghead + gap) + sum(len(g[1]) for g in groups) * row + 26
    x0, x1 = L, w - R
    sx = lambda v: x0 + v * (x1 - x0)  # noqa: E731
    out = []
    for v in (0, 0.5, 1):
        out.append(
            f'<line x1="{sx(v):.1f}" y1="{T}" x2="{sx(v):.1f}" y2="{h - 24}" stroke="{c["separator1"]}"/>'
        )
        out.append(_text(sx(v), h - 8, f"{int(v * 100)}%", c["fg4"], 10.5, "middle", mono=True))
    out.append(
        f'<line x1="{sx(overall):.1f}" y1="{T}" x2="{sx(overall):.1f}" y2="{h - 24}" stroke="{c["fg2"]}" stroke-dasharray="3 3"/>'
    )
    out.append(
        _text(sx(overall), h - 8, f"all {_pct(overall)}", c["fg2"], 10.5, "middle", mono=True)
    )
    y = T
    for title, rows in groups:
        out.append(_text(24, y + 16, title, c["fg2"], 11, caps=True))
        y += ghead
        for r in rows:
            lo, hi = r["ci95"]
            col = _tone(lo, hi, overall, c)
            cy = y + row / 2
            out.append(_text(L - 12, cy + 4, r["label"], c["fg1"], 12.5, "end"))
            out.append(_ci_bar(x0, sx, cy, r["recall"], lo, hi, col, c))
            out.append(
                _text(
                    x1 + 12,
                    cy + 4,
                    _pct(r["recall"]),
                    c["fg1"] if col == c["fg4"] else col,
                    12.5,
                    mono=True,
                    weight=600,
                )
            )
            y += row
        y += gap
    return _svg(w, int(h), "".join(out), c)


# ---- what makes it fire --------------------------------------------------------------------------
def _which_word(phrase: str, wake: list[str]) -> str | None:
    w = phrase.split()
    if len(w) == len(wake):
        diff = [i for i in range(len(w)) if w[i] != wake[i]]
        return f"“{wake[diff[0]]}” swapped" if len(diff) == 1 else None
    it = iter(wake)
    if 0 < len(w) < len(wake) and all(x in it for x in w):
        return "word dropped"
    return None


def near_misses(res: BenchResult, theme: str = "dark", w: int = 880, top: int = 6) -> str:
    """Share of near-miss phrases accepted: by swapped word, then the worst phrases."""
    from .text import _cmu, _split_unknown

    c = PALETTE[theme]
    df = res.frame()
    ti = int(np.argmin(np.abs(THRESHOLDS - res.cfg.threshold)))
    adv = df[df.category == "adversarial"].copy()
    adv["fa"] = [e[ti] > 0 for e in adv.events]
    wake = [
        p for wd in str(df.wake_word.iloc[0]).lower().split() for p in _split_unknown(wd, _cmu())
    ]
    adv["kind"] = adv.phrase.map(lambda p: _which_word(p, wake))
    kinds = [
        (k, g.fa.mean(), wilson(float(g.fa.sum()), len(g)))
        for k, g in adv.dropna(subset=["kind"]).groupby("kind")
        if len(g) >= 30
    ]
    kinds.sort(key=lambda r: -r[1])
    phr = [
        (f"“{p}”", g.fa.mean(), wilson(float(g.fa.sum()), len(g)))
        for p, g in adv.groupby("phrase")
        if len(g) >= 15
    ]
    phr.sort(key=lambda r: (-r[2][0], -r[1]))
    phr = phr[:top]
    overall = float(adv.fa.mean()) if len(adv) else 0.0
    L, R, row, ghead, gap = 190, 64, 24, 28, 10
    groups = [g for g in (("Which word changed", kinds), ("Worst phrases", phr)) if g[1]]
    h = 16 + len(groups) * (ghead + gap) + sum(len(g[1]) for g in groups) * row + 26
    x0, x1 = L, w - R
    sx = lambda v: x0 + v * (x1 - x0)  # noqa: E731
    out = []
    for v in (0, 0.5, 1):
        out.append(
            f'<line x1="{sx(v):.1f}" y1="16" x2="{sx(v):.1f}" y2="{h - 24}" stroke="{c["separator1"]}"/>'
        )
        out.append(_text(sx(v), h - 8, f"{int(v * 100)}%", c["fg4"], 10.5, "middle", mono=True))
    out.append(
        f'<line x1="{sx(overall):.1f}" y1="16" x2="{sx(overall):.1f}" y2="{h - 24}" stroke="{c["fg2"]}" stroke-dasharray="3 3"/>'
    )
    out.append(
        _text(sx(overall), h - 8, f"all {_pct(overall)}", c["fg2"], 10.5, "middle", mono=True)
    )
    y = 16
    for title, items in groups:
        out.append(_text(24, y + 16, title, c["fg2"], 11, caps=True))
        y += ghead
        for label, rate, (lo, hi) in items:
            cy = y + row / 2
            out.append(_text(L - 12, cy + 4, label, c["fg1"], 12.5, "end"))
            out.append(_ci_bar(x0, sx, cy, rate, lo, hi, c["chart8"], c))
            out.append(_text(x1 + 12, cy + 4, _pct(rate), c["chart8"], 12.5, mono=True, weight=600))
            y += row
        y += gap
    return _svg(w, int(h), "".join(out), c)


def alarm_sources(res: BenchResult, theme: str = "dark", w: int = 880) -> str:
    """False alarms per hour by negative audio source."""
    c = PALETTE[theme]
    df = res.frame()
    ti = int(np.argmin(np.abs(THRESHOLDS - res.cfg.threshold)))
    neg = df[(df.label == 0) & (df.category != "adversarial")].copy()
    neg["ev"] = [e[ti] for e in neg.events]
    names = {
        "speech": "synthetic speech",
        "real_speech": "real conversation",
        "noise": "background noise",
    }
    src = [
        (names.get(k, k), g.ev.sum() / max(g.duration_s.sum() / 3600, 1e-9))
        for k, g in neg.groupby("category")
    ]
    src.sort(key=lambda r: -r[1])
    L, R, row = 190, 64, 30
    h = 16 + len(src) * row + 10
    mx = max([v for _, v in src] + [1e-9])
    out = []
    for i, (name, v) in enumerate(src):
        cy = 16 + i * row + row / 2
        bw = (w - L - R - 60) * v / mx
        out.append(_text(L - 12, cy + 4, name, c["fg1"], 12.5, "end"))
        out.append(
            f'<rect x="{L}" y="{cy - 8:.1f}" width="{max(bw, 1):.1f}" height="16" fill="{c["chart3"]}"/>'
        )
        out.append(_text(L + bw + 10, cy + 5, f"{v:.0f}/h", c["fg1"], 12.5, mono=True, weight=600))
    return _svg(w, int(h), "".join(out), c)


def hstack(svgs: list[str], theme: str = "dark", gap: int = 1) -> str:
    """Place SVGs side by side (separated by a hairline)."""
    import re

    c = PALETTE[theme]
    sizes = [
        tuple(int(v) for v in re.search(r'width="(\d+)" height="(\d+)"', s).groups()) for s in svgs
    ]
    w = sum(sz[0] for sz in sizes) + gap * (len(svgs) - 1)
    h = max(sz[1] for sz in sizes)
    body, x = [f'<rect width="{w}" height="{h}" fill="{c["separator1"]}"/>'], 0
    for s, (sw, sh) in zip(svgs, sizes):
        inner = re.sub(r"^<svg[^>]*>", "", s)[: -len("</svg>")]
        body.append(f'<rect x="{x}" width="{sw}" height="{h}" fill="{c["bg1"]}"/>')
        body.append(
            f'<svg x="{x}" y="0" width="{sw}" height="{sh}" viewBox="0 0 {sw} {sh}">{inner}</svg>'
        )
        x += sw + gap
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{_e(SANS)}">{"".join(body)}</svg>'
    )


def write_figures(res: BenchResult, out_dir: str | Path, prefix: str = "") -> list[Path]:
    """README charts as <name>_{dark,light}.svg."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    readme_axes = ["text_form", "accent", "noise_bucket", "noise_type", "reverb"]
    for theme in ("dark", "light"):
        for name, svg in (
            ("scorecard", scorecard(res, theme)),
            ("tradeoff", tradeoff([res], theme)),
            ("failures", failures(res, theme, axes=readme_axes, significant_only=True)),
            ("talker", recall_by_talker(res, theme)),
            ("near_misses", near_misses(res, theme)),
            ("alarm_sources", alarm_sources(res, theme)),
        ):
            p = out / f"{prefix}{name}_{theme}.svg"
            p.write_text(svg)
            paths.append(p)
    return paths
