"""Self-contained HTML report: summary, recall/FPPH curve, per-axis breakdowns, failure gallery."""

from __future__ import annotations

import base64
import html
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .bench import BenchResult

PALETTE = ["#2563eb", "#e4572e", "#17a589", "#a855f7", "#d4a017", "#64748b"]


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _fmt(v: Any, pct: bool = False, nd: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if pct:
        return f"{100 * v:.1f}%"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return _esc(v)


def _wav_b64(audio: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def _curve_svg(results: list[BenchResult], w: int = 640, h: int = 340) -> str:
    """Recall (y) vs false positives per hour (log x)."""
    pad_l, pad_b, pad_t, pad_r = 52, 40, 14, 14
    xmin, xmax = 0.01, 100.0

    def X(f: float) -> float:
        f = min(max(f, xmin), xmax)
        return pad_l + (math.log10(f) - math.log10(xmin)) / (
            math.log10(xmax) - math.log10(xmin)
        ) * (w - pad_l - pad_r)

    def Y(r: float) -> float:
        return pad_t + (1 - r) * (h - pad_t - pad_b)

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Recall vs FPPH">']
    for f in [0.01, 0.1, 1, 10, 100]:
        parts.append(
            f'<line x1="{X(f):.1f}" y1="{pad_t}" x2="{X(f):.1f}" y2="{h - pad_b}" class="grid"/>'
        )
        parts.append(
            f'<text x="{X(f):.1f}" y="{h - pad_b + 16}" class="tick" text-anchor="middle">{f:g}</text>'
        )
    for r in [0, 0.25, 0.5, 0.75, 1.0]:
        parts.append(
            f'<line x1="{pad_l}" y1="{Y(r):.1f}" x2="{w - pad_r}" y2="{Y(r):.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{Y(r) + 4:.1f}" class="tick" text-anchor="end">{int(r * 100)}%</text>'
        )
    parts.append(
        f'<text x="{(w + pad_l) / 2}" y="{h - 6}" class="axis" text-anchor="middle">false positives per hour (log)</text>'
    )
    parts.append(
        f'<text x="14" y="{(h - pad_b) / 2}" class="axis" text-anchor="middle" transform="rotate(-90 14 {(h - pad_b) / 2})">recall</text>'
    )
    for i, res in enumerate(results):
        c = res.summary()["curve"]
        # thresholds high -> low trace a monotone step curve
        pts: list[tuple[float, float]] = []
        for f, r in reversed(list(zip(c["fpph"], c["recall"]))):
            if f is None or r is None:
                continue
            x, y = X(f), Y(r)
            if pts:
                pts.append((x, pts[-1][1]))
            pts.append((x, y))
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        col = PALETTE[i % len(PALETTE)]
        parts.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        op = res.summary()["at_threshold"]
        parts.append(
            f'<circle cx="{X(op["fpph"]):.1f}" cy="{Y(op["recall"]):.1f}" r="4" fill="{col}">'
            f"<title>{_esc(res.detector)} @ {op['threshold']}: recall {op['recall']:.3f}, FPPH {op['fpph']:.2f}</title></circle>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _trace_svg(
    t: np.ndarray,
    s: np.ndarray,
    thr: float,
    region: tuple[float, float] | None,
    w: int = 360,
    h: int = 70,
) -> str:
    if len(t) == 0:
        return ""
    T = float(max(t.max(), 1e-3))

    def X(x: float) -> float:
        return 4 + x / T * (w - 8)

    def Y(y: float) -> float:
        return 4 + (1 - y) * (h - 8)

    parts = [f'<svg viewBox="0 0 {w} {h}" class="trace">']
    if region and region[0] == region[0]:
        parts.append(
            f'<rect x="{X(region[0]):.1f}" y="2" width="{max(X(region[1]) - X(region[0]), 1):.1f}" height="{h - 4}" class="region"/>'
        )
    parts.append(f'<line x1="4" x2="{w - 4}" y1="{Y(thr):.1f}" y2="{Y(thr):.1f}" class="thr"/>')
    d = " ".join(f"{X(float(a)):.1f},{Y(float(b)):.1f}" for a, b in zip(t, s))
    parts.append(f'<polyline points="{d}" class="line"/></svg>')
    return "".join(parts)


def _bar(v: float | None, good_high: bool = True, ci: Any = None) -> str:
    if v is None or v != v:
        return '<td class="num">–</td>'
    q = v if good_high else 1 - min(v, 1)
    hue = int(120 * q)  # red..green
    ci_txt = ""
    if ci and ci[0] == ci[0]:
        ci_txt = f"<small class='ci' title='95% Wilson interval'> {100 * ci[0]:.0f}–{100 * ci[1]:.0f}</small>"
    return (
        f'<td class="num"><span class="bar" style="--w:{100 * min(v, 1):.0f}%;--c:hsl({hue} 65% 45%)"></span>'
        f"{100 * v:.1f}%{ci_txt}</td>"
    )


def _summary_table(results: list[BenchResult]) -> str:
    rows = []
    for i, r in enumerate(results):
        s = r.summary()
        a = s["at_threshold"]
        tg = s["at_target_fpph"]
        tcells = "".join(
            f"<td class='num'>{_fmt(v['recall'], True) if v else '–'}<small> @{v['threshold'] if v else ''}</small></td>"
            for v in tg.values()
        )
        rows.append(
            f"<tr><td><span class='dot' style='background:{PALETTE[i % len(PALETTE)]}'></span>{_esc(r.detector)}</td>"
            f"<td class='num'>{_fmt(a['recall'], True)}</td><td class='num'>{_fmt(a['fpph'])}</td>"
            f"<td class='num'>{_fmt(a['false_accept_rate'], True)}</td>"
            f"<td class='num'>{_fmt(a['adversarial_false_accept_rate'], True)}</td>"
            f"<td class='num'>{_fmt(a['latency_ms_p50'], nd=0)} / {_fmt(a['latency_ms_p90'], nd=0)}</td>"
            f"{tcells}<td class='num'>{_fmt(s['compute']['real_time_factor'], nd=4)}</td></tr>"
        )
    s0 = results[0].summary()
    heads = "".join(f"<th>recall @ {k.replace('<=', '≤ ')}</th>" for k in s0["at_target_fpph"])
    return (
        f"<table class='summary'><thead><tr><th>model</th><th>recall</th><th>FPPH</th>"
        f"<th>false-accept / clip</th><th>adversarial false-accept</th><th>latency p50 / p90 (ms)</th>{heads}<th>RTF</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"<p class='note'>recall / FPPH / false-accept / latency at threshold {results[0].cfg.threshold}. "
        f"Positives: {s0['n_positive']}. Negatives: {s0['n_negative']} clips; FPPH is measured on "
        f"{s0['fpph_hours']} h of ambient audio ({', '.join(results[0].cfg.fpph_categories)}); "
        f"the {s0['n_adversarial']} adversarial clips are reported as a false-accept rate. "
        f"Latency = first trigger − end of wake word (negative = fired before the word ended). "
        f"RTF = detect() wall time ÷ audio duration.</p>"
    )


def _breakdowns(results: list[BenchResult]) -> str:
    bds = [r.breakdown() for r in results]
    out = []
    for key in bds[0]:
        is_pos = key.startswith("positive/")
        metric = "recall" if is_pos else "false_accept_rate"
        values: dict[str, dict[int, dict]] = {}
        for i, bd in enumerate(bds):
            for row in bd.get(key, []):
                values.setdefault(row["value"], {})[i] = row
        order = list(values)
        heads = "".join(
            f"<th>{_esc(r.detector)}<br><small>{'recall' if is_pos else 'false-accept'}</small></th>"
            + ("" if is_pos else "<th><small>FPPH</small></th>")
            for r in results
        )
        body = []
        for v in order:
            first = next(iter(values[v].values()))
            cells = ""
            for i in range(len(results)):
                row = values[v].get(i)
                cells += _bar(
                    row[metric] if row else None,
                    good_high=is_pos,
                    ci=row.get("ci95") if row else None,
                )
                if not is_pos:
                    cells += f"<td class='num'>{_fmt(row.get('fpph')) if row and 'fpph' in row else '–'}</td>"
            body.append(f"<tr><td>{_esc(v)}</td><td class='num'>{first['n']}</td>{cells}</tr>")
        title = key.replace("/", " · ").replace("_", " ")
        out.append(
            f"<details {'open' if key in ('positive/noise_bucket', 'negative/category') else ''}>"
            f"<summary>{_esc(title)}</summary><table><thead><tr><th>value</th><th>n</th>{heads}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></details>"
        )
    return "".join(out)


def _failure_cards(res: BenchResult, data: Any, k: int) -> str:
    f = res.failures(k=k)
    thr = res.cfg.threshold
    ids = {r["id"] for rows in f.values() for r in rows}
    index = {sid: i for i, sid in enumerate(data.ds["id"])} if data is not None else {}
    out = []
    for kind, rows in f.items():
        cards = []
        for r in rows:
            t, s = res.traces.get(r["id"], (np.zeros(0), np.zeros(0)))
            region = (r.get("phrase_onset_s", math.nan), r.get("phrase_offset_s", math.nan))
            if region[0] != region[0]:
                region = (r.get("onset_s", math.nan), r.get("offset_s", math.nan))
            audio_html = ""
            if r["id"] in index:
                smp = data[index[r["id"]]]
                a, sr = smp.audio, smp.sample_rate
                if kind == "false_alarms" and len(a) / sr > 8:
                    c = r.get("max_score_t", 0.0)
                    lo = max(0.0, c - 4.0)
                    a = a[int(lo * sr) : int((c + 1.0) * sr)]
                audio_html = f'<audio controls preload="none" src="data:audio/wav;base64,{_wav_b64(a, sr)}"></audio>'
            meta = " · ".join(
                _esc(x)
                for x in [
                    r.get("category"),
                    r.get("noise_bucket"),
                    f"{r['snr_db']:.1f} dB" if r.get("snr_db") == r.get("snr_db") else None,
                    r.get("noise_category") or None,
                    "reverb" if r.get("reverb") else None,
                    r.get("bandlimit") if r.get("bandlimit") != "none" else None,
                    f"{r.get('provider')}/{r.get('voice_name')}"
                    if r.get("voice_name")
                    else r.get("provider"),
                    r.get("accent") if r.get("accent") not in ("", "unknown") else None,
                ]
                if x
            )
            cards.append(
                f"<div class='card'><div class='cardhead'><code>{_esc(r['id'])}</code>"
                f"<b>max {r['max_score']:.3f}</b></div>"
                f"<div class='text'>“{_esc(r.get('text', ''))}”</div><div class='meta'>{meta}</div>"
                f"{_trace_svg(t, s, thr, region if kind == 'missed_positives' else None)}{audio_html}</div>"
            )
        label = (
            "Missed wake words (lowest scores)"
            if kind == "missed_positives"
            else "False alarms (highest scores)"
        )
        out.append(
            f"<h3>{label} <small>{len(rows)}</small></h3><div class='cards'>{''.join(cards) or '<p>none 🎉</p>'}</div>"
        )
    _ = ids
    return "".join(out)


CSS = """
:root{--bg:#fff;--fg:#1f2328;--muted:#656d76;--line:#d8dee4;--card:#f6f8fa;--accent:#2563eb}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0d1117;--fg:#e6edf3;--muted:#8d96a0;--line:#30363d;--card:#161b22;--accent:#58a6ff}}
:root[data-theme=dark]{--bg:#0d1117;--fg:#e6edf3;--muted:#8d96a0;--line:#30363d;--card:#161b22;--accent:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 80px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:15px;margin:22px 0 8px}small,.note{color:var(--muted)}table{border-collapse:collapse;width:100%;margin:6px 0 10px;font-variant-numeric:tabular-nums}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-weight:600;font-size:12px}
td.num{text-align:right;white-space:nowrap;position:relative}.bar{display:inline-block;width:46px;height:8px;margin-right:6px;background:linear-gradient(90deg,var(--c) var(--w),var(--line) var(--w));border-radius:4px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}.wrap{overflow-x:auto}
details{border:1px solid var(--line);border-radius:8px;padding:6px 12px;margin:8px 0;background:var(--card)}summary{cursor:pointer;font-weight:600;text-transform:capitalize}
.chart{width:100%;max-width:720px;height:auto}.chart .grid{stroke:var(--line)}.chart .tick,.chart .axis{fill:var(--muted);font-size:11px}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:10px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px}
.cardhead{display:flex;justify-content:space-between;gap:8px;font-size:12px}.text{margin:4px 0;font-style:italic}.meta{font-size:12px;color:var(--muted)}
.trace{width:100%;height:70px;margin-top:6px}.trace .line{fill:none;stroke:var(--accent);stroke-width:1.5}.trace .thr{stroke:#e4572e;stroke-dasharray:4 3}.trace .region{fill:var(--accent);opacity:.15}
.ci{color:var(--muted);font-size:10px}audio{width:100%;height:32px;margin-top:6px}code{font-size:11px}
"""


def render_report(
    results: list[BenchResult],
    out_path: str | Path,
    data: Any = None,
    title: str = "Wake word benchmark",
    failures_per_model: int = 24,
) -> Path:
    body = [
        f"<h1>{_esc(title)}</h1><p class='note'>{len(results)} model(s)</p>",
        "<h2>Summary</h2><div class='wrap'>",
        _summary_table(results),
        "</div>",
        "<h2>Recall vs false positives per hour</h2>",
        _curve_svg(results),
        "<p class='note'>Dots mark the primary threshold. Up-and-left is better.</p>",
        "<h2>Where it fails: breakdowns</h2>",
        "<p class='note'>Positives: recall per slice. Negatives: share of clips with ≥1 false trigger and false triggers per hour. Sorted worst first. Small numbers are 95% confidence intervals; treat wide ones as anecdotes.</p>",
        "<div class='wrap'>",
        _breakdowns(results),
        "</div>",
    ]
    for r in results:
        body.append(f"<h2>Failures: {_esc(r.detector)}</h2>")
        body.append(_failure_cards(r, data, failures_per_model))
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_esc(title)}</title><style>{CSS}</style></head><body><main>"
        + "".join(body)
        + f"<script type='application/json' id='summary'>{json.dumps([r.summary() for r in results])}</script>"
        + "</main></body></html>"
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out
