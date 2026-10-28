"""Static SVG charts: one soft light theme, rounded marks, few words.

Charts carry almost no text: labels, values, axes. Explanations belong in captions.
Colors are colorblind-safe: blue means caught or better, coral means missed or worse.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import numpy as np

from .bench import THRESHOLDS, BenchResult, wilson

SANS = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, 'SF Mono', 'Cascadia Mono', Menlo, monospace"
DISPLAY = SANS

PALETTE = {
    "fg0": "#1d1d1f",  # titles, values
    "fg1": "#3d3c39",  # labels
    "fg2": "#5f5e5a",  # secondary labels
    "fg3": "#8d8c87",  # axes, notes
    "fg4": "#b4b3ad",  # faint
    "bg0": "#f6f6f3",  # page
    "bg1": "#ffffff",  # card
    "bg2": "#f7f7f4",  # tile, hover
    "bg3": "#efeee9",  # bar track
    "separator1": "#ebeae5",
    "separator2": "#dddcd6",
    "accent": "#3b82e6",
    "better": "#3b82e6",
    "worse": "#e8655c",
    "neutral": "#c9c8c1",
    "wave": "#d3d2cc",
}
# multi-model series, in fixed order (validated adjacent pairs, light surface)
SERIES = ("#3b82e6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#eda100")
# coral, light to dark: a second voice from very quiet to loud
CORAL = ("#f19a92", "#e8655c", "#d24c43", "#b0392f")
RADIUS = 14  # card corners
NUM = 'style="font-variant-numeric:tabular-nums"'


def _e(x: Any) -> str:
    return html.escape(str(x))


def _svg(w: int, h: int, body: str, card: bool = True) -> str:
    bg = (
        f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="{RADIUS}" '
        f'fill="{PALETTE["bg1"]}" stroke="{PALETTE["separator1"]}"/>'
        if card
        else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'font-family="{_e(SANS)}">{bg}{body}</svg>'
    )


def _text(
    x: float,
    y: float,
    s: str,
    c: str,
    size: float = 12,
    anchor: str = "start",
    num: bool = False,
    weight: int = 400,
) -> str:
    style = f" {NUM}" if num else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" fill="{c}" font-size="{size}" text-anchor="{anchor}"'
        f' font-weight="{weight}"{style}>{_e(s)}</text>'
    )


PAD = 36  # outer padding on every chart


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def _frame(title: str, legend: list[tuple[str, str]] | None = None, w: int = 880) -> list[str]:
    """Title (top-left) and an optional color legend (top-right)."""
    c = PALETTE
    out = [_text(PAD, PAD + 14, title, c["fg0"], 16, weight=600)]
    x = w - PAD
    for label, col in reversed(legend or []):
        out.append(_text(x, PAD + 13, label, c["fg2"], 12, "end"))
        x -= len(label) * 6.2 + 10
        out.append(f'<circle cx="{x - 4:.1f}" cy="{PAD + 9}" r="4" fill="{col}"/>')
        x -= 26
    return out


def _bar_row(sx: Any, x0: float, cy: float, v: float, lo: float, hi: float, col: str) -> str:
    """A rounded bar on a full-width track, with its 95% interval as a thin line."""
    c, hh = PALETTE, 6
    out = (
        f'<rect x="{x0}" y="{cy - hh:.1f}" width="{sx(1) - x0:.1f}" height="{2 * hh}" '
        f'rx="{hh}" fill="{c["bg3"]}"/>'
    )
    out += (
        f'<rect x="{x0}" y="{cy - hh:.1f}" width="{max(sx(v) - x0, 2 * hh):.1f}" height="{2 * hh}" '
        f'rx="{hh}" fill="{col}"/>'
    )
    out += (
        f'<line x1="{sx(lo):.1f}" y1="{cy:.1f}" x2="{sx(hi):.1f}" y2="{cy:.1f}" '
        f'stroke="{c["fg0"]}" stroke-opacity="0.35" stroke-width="1.5" stroke-linecap="round"/>'
    )
    return out


def _tone(lo: float, hi: float, ref: float, good_high: bool = True) -> str:
    if hi < ref:
        return PALETTE["worse"] if good_high else PALETTE["better"]
    if lo > ref:
        return PALETTE["better"] if good_high else PALETTE["worse"]
    return PALETTE["neutral"]


def _pill(x: float, y: float, label: str, col: str, anchor: str = "start") -> str:
    """A small tinted label, e.g. caught / missed."""
    w = len(label) * 6.6 + 18
    x0 = x if anchor == "start" else x - w
    return (
        f'<rect x="{x0:.1f}" y="{y - 14:.1f}" width="{w:.1f}" height="20" rx="10" '
        f'fill="{col}" fill-opacity="0.12"/>'
        + _text(x0 + w / 2, y + 0.5, label, col, 11.5, "middle", weight=600)
    )


def _vline_label(x: float, y0: float, y1: float, label: str) -> list[str]:
    c = PALETTE
    return [
        f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}" stroke="{c["fg3"]}" '
        'stroke-dasharray="3 4" stroke-linecap="round"/>',
        _text(x, y0 - 8, label, c["fg2"], 11.5, "middle", num=True, weight=500),
    ]


# ---- teaser: one clip, with and without a second voice ----------------------------------------
def traces(examples: list[dict[str, Any]], thr: float, w: int = 560) -> str:
    """examples: [{"label", "audio", "sr", "times", "scores"}]."""
    c = PALETTE
    out = _frame("One clip, two rooms", w=w)
    top, row_h = PAD + 50, 140
    h = top + len(examples) * row_h + PAD - 16
    L, R = PAD, w - PAD - 104
    T = max(len(e["audio"]) / e["sr"] for e in examples)
    sx = lambda t: L + t / T * (R - L)  # noqa: E731
    out.append(
        "<defs>"
        + "".join(
            f'<linearGradient id="fill-{name}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{c[name]}" stop-opacity="0.22"/>'
            f'<stop offset="1" stop-color="{c[name]}" stop-opacity="0"/></linearGradient>'
            for name in ("better", "worse")
        )
        + "</defs>"
    )
    for i, ex in enumerate(examples):
        y0 = top + i * row_h
        best = float(np.max(ex["scores"]))
        hit = best >= thr
        tone = "better" if hit else "worse"
        col = c[tone]
        out.append(_text(L, y0 + 4, ex["label"], c["fg1"], 13, weight=500))
        a = ex["audio"]
        n = 110
        peak = max(float(np.abs(a).max()), 1e-6)
        mid, amp = y0 + 34, 16
        for j, seg in enumerate(np.array_split(np.abs(a), n)):
            v = max(float(seg.max()) / peak * amp if len(seg) else 0, 1)
            x = sx((j + 0.5) / n * T)
            out.append(
                f'<line x1="{x:.1f}" y1="{mid - v:.1f}" x2="{x:.1f}" y2="{mid + v:.1f}" '
                f'stroke="{c["wave"]}" stroke-width="2" stroke-linecap="round"/>'
            )
        base, top_s = y0 + 112, y0 + 64
        sy = lambda v, b=base, t=top_s: b - v * (b - t)  # noqa: E731
        out.append(
            f'<line x1="{L}" y1="{sy(thr):.1f}" x2="{R}" y2="{sy(thr):.1f}" stroke="{c["fg4"]}" '
            'stroke-dasharray="3 4" stroke-linecap="round"/>'
        )
        xy = [(sx(float(t)), sy(float(v))) for t, v in zip(ex["times"], ex["scores"])]
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
        area = f"{xy[0][0]:.1f},{base} {pts} {xy[-1][0]:.1f},{base}"
        out.append(f'<polygon points="{area}" fill="url(#fill-{tone})"/>')
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        out.append(_text(R + 24, base - 30, f"{best:.2f}", c["fg0"], 26, num=True, weight=600))
        out.append(_pill(R + 24, base - 2, "caught" if hit else "missed", col))
        if i == 0:
            out.append(_text(L, sy(thr) - 7, "trigger level", c["fg3"], 11))
    return _svg(w, h, "".join(out))


def recall_by_talker(res: BenchResult, w: int = 380, h: int = 380) -> str:
    """Recall vs how loud a second talker is."""
    c = PALETTE
    df = res.frame()
    ti = int(np.argmin(np.abs(THRESHOLDS - res.cfg.threshold)))
    pos = df[df.label == 1].copy()
    pos["hit"] = [d[ti] for d in pos.detected]
    groups = [("nobody", pos[pos.interferer == ""])]
    tk = pos[pos.interferer != ""]
    for lo, hi, name in (
        (15, 20, "very quiet"),
        (10, 15, "quiet"),
        (5, 10, "medium"),
        (0, 5, "loud"),
    ):
        groups.append((name, tk[(tk.sir_db >= lo) & (tk.sir_db < hi)]))
    out = _frame("Another voice talks louder", w=w)
    L, R, T, B = PAD + 30, w - PAD, PAD + 60, 92
    bw = (R - L) / len(groups)
    cw = min(bw * 0.62, 36)
    sy = lambda v: T + (1 - v) * (h - T - B)  # noqa: E731
    for v in (0, 0.5, 1):
        out.append(
            f'<line x1="{L}" y1="{sy(v):.1f}" x2="{R}" y2="{sy(v):.1f}" stroke="{c["separator1"]}"/>'
        )
        out.append(_text(L - 10, sy(v) + 4, f"{int(v * 100)}%", c["fg3"], 11, "end", num=True))
    for i, (name, g) in enumerate(groups):
        if not len(g):
            continue
        r = float(g.hit.mean())
        lo, hi = wilson(float(g.hit.sum()), len(g))
        cx = L + (i + 0.5) * bw
        col = c["better"] if i == 0 else CORAL[min(i - 1, len(CORAL) - 1)]
        y, b, rr = sy(r), sy(0), min(6, (sy(0) - sy(r)) / 2)
        x0, x1 = cx - cw / 2, cx + cw / 2
        out.append(
            f'<path d="M{x0:.1f},{b:.1f} V{y + rr:.1f} Q{x0:.1f},{y:.1f} {x0 + rr:.1f},{y:.1f} '
            f'H{x1 - rr:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + rr:.1f} V{b:.1f} Z" fill="{col}"/>'
        )
        out.append(
            f'<line x1="{cx:.1f}" y1="{sy(lo):.1f}" x2="{cx:.1f}" y2="{sy(hi):.1f}" '
            f'stroke="{c["fg0"]}" stroke-opacity="0.35" stroke-width="1.5" stroke-linecap="round"/>'
        )
        out.append(_text(cx, sy(hi) - 9, _pct(r), c["fg0"], 13, "middle", num=True, weight=600))
        out.append(_text(cx, h - B + 22, name, c["fg2"], 11.5, "middle"))
    ax = h - B + 52
    out.append(
        f'<line x1="{L}" y1="{ax}" x2="{R - 6}" y2="{ax}" stroke="{c["fg4"]}" stroke-linecap="round"/>'
    )
    out.append(
        f'<path d="M{R - 9},{ax - 4} L{R - 3},{ax} L{R - 9},{ax + 4}" fill="none" '
        f'stroke="{c["fg4"]}" stroke-linecap="round" stroke-linejoin="round"/>'
    )
    out.append(_text((L + R) / 2, ax + 20, "other voice gets louder", c["fg3"], 11.5, "middle"))
    return _svg(w, h, "".join(out))


# ---- headline numbers ------------------------------------------------------------------------
def scorecard(res: BenchResult, w: int = 880) -> str:
    c = PALETTE
    s = res.summary()
    a = s["at_threshold"]
    tiles = [
        ("Score", f"{100 * s['score']:.1f}%", c["accent"]),
        ("Recall", f"{100 * a['recall']:.1f}%", c["fg0"]),
        ("False alarms / h", f"{a['fpph']:.1f}", c["fg0"]),
        ("Near-misses accepted", f"{100 * a['adversarial_false_accept_rate']:.0f}%", c["fg0"]),
        (
            "Latency p50",
            f"{a['latency_ms_p50']:+.0f} ms" if a["latency_ms_p50"] is not None else "–",
            c["fg0"],
        ),
        ("Compute", f"{s['compute']['real_time_factor']:.3f}× RT", c["fg0"]),
    ]
    cols, gap = 3, 12
    tw = (w - 2 * PAD - (cols - 1) * gap) / cols
    th = 96
    rows = math.ceil(len(tiles) / cols)
    h = 2 * PAD + rows * th + (rows - 1) * gap
    out = []
    for i, (label, value, col) in enumerate(tiles):
        x = PAD + (i % cols) * (tw + gap)
        y = PAD + (i // cols) * (th + gap)
        out.append(
            f'<rect x="{x:.1f}" y="{y}" width="{tw:.1f}" height="{th}" rx="10" fill="{c["bg2"]}"/>'
        )
        out.append(_text(x + 20, y + 32, label, c["fg3"], 12, weight=500))
        out.append(_text(x + 20, y + 72, value, col, 28, num=True, weight=600))
    return _svg(w, int(h), "".join(out))


# ---- recall vs false alarms ------------------------------------------------------------------
def tradeoff(results: list[BenchResult], w: int = 880, h: int = 420) -> str:
    """Recall vs false alarms per hour, with the default threshold and the target called out."""
    c = PALETTE
    legend = [(r.detector, SERIES[i % len(SERIES)]) for i, r in enumerate(results)]
    out = _frame("Fewer false alarms cost recall", legend if len(results) > 1 else None, w)
    L, R, T, B = PAD + 44, w - PAD, PAD + 56, 76
    xmin, xmax = 0.1, 1000.0

    def lx(f: float) -> float:
        span = math.log10(xmax) - math.log10(xmin)
        return L + (math.log10(min(max(f, xmin), xmax)) - math.log10(xmin)) / span * (R - L)

    def ly(r: float) -> float:
        return T + (1 - r) * (h - T - B)

    tgt = results[0].cfg.score_fpph if results else 0.5
    out.append(
        f'<rect x="{L}" y="{T}" width="{lx(tgt) - L:.1f}" height="{h - T - B}" rx="8" '
        f'fill="{c["better"]}" fill-opacity="0.07"/>'
    )
    out.append(_text(L + 10, h - B - 12, "target", c["better"], 11.5, weight=600))
    for f, lab in ((0.1, "0.1"), (1, "1"), (10, "10"), (100, "100"), (1000, "1000")):
        out.append(
            f'<line x1="{lx(f):.1f}" y1="{T}" x2="{lx(f):.1f}" y2="{h - B}" stroke="{c["separator1"]}"/>'
        )
        out.append(_text(lx(f), h - B + 22, lab, c["fg3"], 11, "middle", num=True))
    for r in (0, 0.5, 1):
        out.append(
            f'<line x1="{L}" y1="{ly(r):.1f}" x2="{R}" y2="{ly(r):.1f}" stroke="{c["separator1"]}"/>'
        )
        out.append(_text(L - 10, ly(r) + 4, f"{int(r * 100)}%", c["fg3"], 11, "end", num=True))
    out.append(_text((L + R) / 2, h - B + 50, "false alarms per hour", c["fg3"], 12, "middle"))
    ym = (T + h - B) / 2
    out.append(
        f'<text transform="rotate(-90 {PAD + 2} {ym})" x="{PAD + 2}" y="{ym}" fill="{c["fg3"]}" '
        'font-size="12" text-anchor="middle">wake words caught</text>'
    )
    for i, res in enumerate(results):
        col = SERIES[i % len(SERIES)]
        s = res.summary()
        cv = s["curve"]
        curve = [
            (f, r) for f, r in zip(cv["fpph"], cv["recall"]) if f is not None and r is not None
        ]
        # everything at or below the left edge collapses to its best recall (no drop to 0%)
        r0 = max([r for f, r in curve if f <= xmin], default=None)
        curve = [(f, r) for f, r in curve if f > xmin] + ([(xmin, r0)] if r0 is not None else [])
        pts: list[tuple[float, float]] = []
        for f, r in reversed(curve):
            x, y = lx(f), ly(r)
            if pts:
                pts.append((x, pts[-1][1]))
            pts.append((x, y))
        out.append(
            f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" '
            f'stroke="{col}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        a = s["at_threshold"]
        px, py = lx(a["fpph"]), ly(a["recall"])
        out.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{c["bg1"]}" stroke="{col}" stroke-width="2"/>'
        )
        if len(results) == 1:
            out.append(
                _text(
                    px + 12,
                    py + 24,
                    f"default threshold: {_pct(a['recall'])} caught, {a['fpph']:.1f} false alarms/h",
                    c["fg2"],
                    12,
                )
            )
            qx, qy = lx(xmin), ly(s["score"])
            out.append(
                f'<circle cx="{qx:.1f}" cy="{qy:.1f}" r="6" fill="{col}" stroke="{c["bg1"]}" stroke-width="2"/>'
            )
            out.append(
                _text(
                    qx + 14,
                    qy - 10,
                    f"at target: {100 * s['score']:.1f}% caught",
                    c["fg0"],
                    12.5,
                    weight=600,
                )
            )
    return _svg(w, h, "".join(out))


# ---- where it fails ------------------------------------------------------------------------------
FAILURE_AXES = [
    (
        "interferer",
        "Another voice",
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
        "Background noise",
        {
            "clean": "none",
            "quiet": "quiet",
            "moderate": "moderate",
            "noisy": "noisy",
            "very_noisy": "very noisy",
            "extreme": "louder than speech",
        },
    ),
    ("noise_type", "Noise type", {"": "none"}),
    ("reverb", "Room", {"False": "dry", "True": "echoey"}),
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
        if significant_only:
            # keep the conditions that clearly differ from the average, plus the best for reference
            sig = [r for r in rows if r["ci95"][1] < overall or r["ci95"][0] > overall]
            if not sig:
                continue
            rows = [rows[0]] + [r for r in sig if r is not rows[0]]
        for r in rows:
            r["label"] = names.get(r["value"], r["value"] or "none")
        groups.append((title, rows))
    return overall, groups


def _legend() -> list[tuple[str, str]]:
    c = PALETTE
    return [("worse", c["worse"]), ("better", c["better"]), ("no clear difference", c["neutral"])]


def failures(
    res: BenchResult,
    w: int = 880,
    min_n: int = 50,
    axes: list[str] | None = None,
    significant_only: bool = False,
) -> str:
    """Recall per condition vs the average; color only when the 95% CI excludes the average."""
    c = PALETTE
    overall, groups = _slices(res, min_n, axes, significant_only)
    out = _frame("Wake words caught, by condition", _legend(), w)
    L, R, row, ghead, gap = PAD + 170, w - PAD - 56, 28, 32, 14
    T = PAD + 70
    h = T + len(groups) * (ghead + gap) + sum(len(g[1]) for g in groups) * row + PAD - gap
    sx = lambda v: L + v * (R - L)  # noqa: E731
    out += _vline_label(sx(overall), T, h - PAD, f"average {_pct(overall)}")[1:]
    y = T
    for title, rows in groups:
        out.append(_text(PAD, y + 20, title, c["fg0"], 12.5, weight=600))
        y += ghead
        for r in rows:
            lo, hi = r["ci95"]
            cy = y + row / 2
            out.append(_text(L - 16, cy + 4, r["label"], c["fg2"], 12.5, "end"))
            out.append(_bar_row(sx, L, cy, r["recall"], lo, hi, _tone(lo, hi, overall)))
            out.append(
                _text(R + 16, cy + 4.5, _pct(r["recall"]), c["fg0"], 13, num=True, weight=600)
            )
            y += row
        y += gap
    # draw the average line on top of the bars
    out += _vline_label(sx(overall), T, h - PAD, f"average {_pct(overall)}")[:1]
    return _svg(w, int(h), "".join(out))


# ---- what makes it fire --------------------------------------------------------------------------
def _which_word(phrase: str, wake: list[str]) -> str | None:
    w = phrase.split()
    if len(w) == len(wake):
        diff = [i for i in range(len(w)) if w[i] != wake[i]]
        return f"“{wake[diff[0]]}” replaced" if len(diff) == 1 else None
    it = iter(wake)
    if 0 < len(w) < len(wake) and all(x in it for x in w):
        return "a word left out"
    return None


def near_misses(res: BenchResult, w: int = 880, top: int = 6) -> str:
    """How often near-miss phrases trigger the model."""
    from .text import _cmu, _split_unknown

    c = PALETTE
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
    phr.sort(key=lambda r: (-r[2][0], -r[1]))  # pick by CI lower bound (robust)...
    phr = sorted(phr[:top], key=lambda r: -r[1])  # ...display by the value shown
    out = _frame("Similar-sounding phrases that wake the model", w=w)
    L, R, row, ghead, gap = PAD + 170, w - PAD - 56, 28, 32, 14
    T = PAD + 44
    groups = [g for g in (("By change", kinds), ("Worst phrases", phr)) if g[1]]
    h = T + len(groups) * (ghead + gap) + sum(len(g[1]) for g in groups) * row + PAD - gap
    sx = lambda v: L + v * (R - L)  # noqa: E731
    y = T
    for title, items in groups:
        out.append(_text(PAD, y + 20, title, c["fg0"], 12.5, weight=600))
        y += ghead
        for label, rate, (lo, hi) in items:
            cy = y + row / 2
            out.append(_text(L - 16, cy + 4, label, c["fg2"], 12.5, "end"))
            out.append(_bar_row(sx, L, cy, rate, lo, hi, c["worse"]))
            out.append(_text(R + 16, cy + 4.5, _pct(rate), c["fg0"], 13, num=True, weight=600))
            y += row
        y += gap
    return _svg(w, int(h), "".join(out))


def alarm_sources(res: BenchResult, w: int = 880) -> str:
    """False alarms per hour by negative audio source."""
    c = PALETTE
    df = res.frame()
    ti = int(np.argmin(np.abs(THRESHOLDS - res.cfg.threshold)))
    neg = df[(df.label == 0) & (df.category != "adversarial")].copy()
    neg["ev"] = [e[ti] for e in neg.events]
    names = {
        "speech": "people talking (synthetic)",
        "real_speech": "people talking (real)",
        "noise": "background noise",
        "media": "TV, music, speeches",
    }
    src = [
        (names.get(k, k), g.ev.sum() / max(g.duration_s.sum() / 3600, 1e-9))
        for k, g in neg.groupby("category")
    ]
    src.sort(key=lambda r: -r[1])
    out = _frame("False alarms per hour, by what the model hears", w=w)
    L, R, row = PAD + 190, w - PAD - 64, 36
    T = PAD + 44
    h = T + len(src) * row + PAD - 8
    mx = max([v for _, v in src] + [1e-9])
    for i, (name, v) in enumerate(src):
        cy = T + i * row + row / 2
        bw = (R - L) * v / mx
        out.append(_text(L - 16, cy + 4, name, c["fg2"], 12.5, "end"))
        out.append(
            f'<rect x="{L}" y="{cy - 6:.1f}" width="{R - L:.1f}" height="12" rx="6" fill="{c["bg3"]}"/>'
        )
        if v > 0:
            out.append(
                f'<rect x="{L}" y="{cy - 6:.1f}" width="{max(bw, 12):.1f}" height="12" rx="6" '
                f'fill="{c["worse"]}"/>'
            )
        out.append(_text(R + 16, cy + 4.5, f"{v:.1f}", c["fg0"], 13, num=True, weight=600))
    return _svg(w, int(h), "".join(out))


def hstack(svgs: list[str], gap: int = 16) -> str:
    """Place chart cards side by side."""
    import re

    sizes = [
        tuple(int(v) for v in re.search(r'width="(\d+)" height="(\d+)"', s).groups()) for s in svgs
    ]
    w = sum(sz[0] for sz in sizes) + gap * (len(svgs) - 1)
    h = max(sz[1] for sz in sizes)
    body, x = [], 0
    for s, (sw, sh) in zip(svgs, sizes):
        inner = re.sub(r"^<svg[^>]*>", "", s)[: -len("</svg>")]
        # stretch each card to the tallest one
        inner = inner.replace(
            f'height="{sh - 1}" rx="{RADIUS}"', f'height="{h - 1}" rx="{RADIUS}"', 1
        )
        body.append(
            f'<svg x="{x}" y="0" width="{sw}" height="{h}" viewBox="0 0 {sw} {h}">{inner}</svg>'
        )
        x += sw + gap
    return _svg(w, h, "".join(body), card=False)


README_AXES = ["interferer", "noise_bucket", "noise_type", "reverb", "accent", "provider"]


def write_figures(res: BenchResult, out_dir: str | Path, prefix: str = "") -> list[Path]:
    """Charts as <name>.svg."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, svg in (
        ("tradeoff", tradeoff([res])),
        ("failures", failures(res, axes=README_AXES, significant_only=True)),
        ("near_misses", near_misses(res)),
        ("alarm_sources", alarm_sources(res)),
    ):
        p = out / f"{prefix}{name}.svg"
        p.write_text(svg, encoding="utf-8")
        paths.append(p)
    return paths
