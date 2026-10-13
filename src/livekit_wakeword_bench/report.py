"""Self-contained HTML report in the LiveKit Bytes design language (dark first)."""

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

from . import viz
from .bench import BenchResult


def _esc(x: Any) -> str:
    return html.escape(str(x))


def _pct(v: float | None) -> str:
    return "–" if v is None or v != v else f"{100 * v:.1f}%"


def _wav_b64(audio: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def _themed(fn: Any, *args: Any, **kw: Any) -> str:
    """The same chart in both themes; CSS shows the one matching the viewer's OS."""
    return (
        f"<div class='chart dark-only'>{fn(*args, theme='dark', **kw)}</div>"
        f"<div class='chart light-only'>{fn(*args, theme='light', **kw)}</div>"
    )


def _trace_svg(t: np.ndarray, s: np.ndarray, thr: float, region: tuple[float, float] | None) -> str:
    w, h = 360, 64
    if len(t) == 0:
        return ""
    T = float(max(t.max(), 1e-3))

    def X(x: float) -> float:
        return 4 + x / T * (w - 8)

    def Y(y: float) -> float:
        return 4 + (1 - y) * (h - 8)

    parts = [f'<svg viewBox="0 0 {w} {h}" class="trace">']
    if region and region[0] == region[0]:
        width = max(X(region[1]) - X(region[0]), 1)
        parts.append(
            f'<rect x="{X(region[0]):.1f}" y="2" width="{width:.1f}" height="{h - 4}" class="region"/>'
        )
    parts.append(f'<line x1="4" x2="{w - 4}" y1="{Y(thr):.1f}" y2="{Y(thr):.1f}" class="thr"/>')
    d = " ".join(f"{X(float(a)):.1f},{Y(float(b)):.1f}" for a, b in zip(t, s))
    parts.append(f'<polyline points="{d}" class="line"/></svg>')
    return "".join(parts)


def _rate_cell(v: float | None, ci: Any, overall: float | None, good_high: bool) -> str:
    """Value + CI with a meter colored only when the CI excludes `overall`."""
    if v is None or v != v:
        return "<td class='num'>–</td>"
    tone = "muted"
    if ci and overall is not None and ci[0] == ci[0]:
        below, above = ci[1] < overall, ci[0] > overall
        if below or above:
            tone = "bad" if below == good_high else "good"
    ci_txt = (
        f"<span class='ci'>{100 * ci[0]:.0f}–{100 * ci[1]:.0f}</span>"
        if ci and ci[0] == ci[0]
        else ""
    )
    return (
        f"<td class='num'><span class='meter {tone}'><i style='width:{100 * min(v, 1):.0f}%'></i></span>"
        f"<span class='val {tone}'>{100 * v:.1f}%</span>{ci_txt}</td>"
    )


def _summary_table(results: list[BenchResult]) -> str:
    s0 = results[0].summary()
    heads = "".join(f"<th>recall @ fpph ≤ {k.split('<=')[1]}</th>" for k in s0["at_target_fpph"])
    rows = []
    for i, r in enumerate(results):
        s = r.summary()
        a = s["at_threshold"]
        col = viz.SERIES[i % len(viz.SERIES)]
        tcells = "".join(
            f"<td class='num'>{_pct(v['recall']) if v else '–'}"
            f"<span class='ci'>{'thr ' + format(v['threshold'], 'g') if v else ''}</span></td>"
            for v in s["at_target_fpph"].values()
        )
        lat = (
            f"{a['latency_ms_p50']} / {a['latency_ms_p90']}"
            if a["latency_ms_p50"] is not None
            else "–"
        )
        rows.append(
            f"<tr><td><span class='deco' style='background:var(--{col})'></span>{_esc(r.detector)}</td>"
            f"<td class='num strong accent'>{_pct(s['score'])}</td>"
            f"<td class='num'>{_pct(a['recall'])}</td><td class='num'>{a['fpph']:.2f}</td>"
            f"<td class='num'>{_pct(a['false_accept_rate'])}</td>"
            f"<td class='num'>{_pct(a['adversarial_false_accept_rate'])}</td>"
            f"<td class='num'>{lat}</td>{tcells}"
            f"<td class='num'>{s['compute']['real_time_factor']:.3f}</td></tr>"
        )
    cfg = results[0].cfg
    return (
        "<div class='table-wrap'><table><thead><tr><th>model</th><th>score</th><th>recall</th>"
        "<th>fpph</th><th>false accept / clip</th><th>confusables accepted</th>"
        f"<th>latency p50 / p90 ms</th>{heads}<th>rtf</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        f"<p class='note'><b>score</b> = recall at FPPH ≤ {cfg.score_fpph:g}. Other columns at threshold "
        f"{cfg.threshold:g}. {s0['n_positive']:,} wake words; FPPH measured on {s0['fpph_hours']} h of "
        f"{', '.join(cfg.fpph_categories)}; {s0['n_adversarial']:,} confusable clips scored as a "
        "false-accept rate. Latency = first trigger − end of wake word. RTF = detect() time ÷ audio time.</p>"
    )


def _breakdowns(results: list[BenchResult]) -> str:
    bds = [r.breakdown() for r in results]
    overall_pos = [r.summary()["at_threshold"]["recall"] for r in results]
    out = []
    for key in bds[0]:
        is_pos = key.startswith("positive/")
        metric = "recall" if is_pos else "false_accept_rate"
        values: dict[str, dict[int, dict]] = {}
        for i, bd in enumerate(bds):
            for row in bd.get(key, []):
                values.setdefault(row["value"], {})[i] = row
        heads = "".join(
            f"<th>{_esc(r.detector)} · {'recall' if is_pos else 'false accept'}</th>"
            + ("" if is_pos else "<th>fpph</th>")
            for r in results
        )
        body = []
        for v, per in values.items():
            first = next(iter(per.values()))
            cells = ""
            for i in range(len(results)):
                row = per.get(i)
                ref = overall_pos[i] if is_pos else None
                cells += _rate_cell(
                    row[metric] if row else None, row.get("ci95") if row else None, ref, is_pos
                )
                if not is_pos:
                    fp = f"{row['fpph']:.1f}" if row and "fpph" in row else "–"
                    cells += f"<td class='num'>{fp}</td>"
            body.append(
                f"<tr><td>{_esc(v or 'none')}</td><td class='num dim'>{first['n']}</td>{cells}</tr>"
            )
        title = key.replace("/", " · ").replace("_", " ")
        opened = (
            "open"
            if key in ("positive/interferer", "positive/noise_bucket", "negative/category")
            else ""
        )
        out.append(
            f"<details {opened}><summary><span class='caps'>{_esc(title)}</span></summary>"
            f"<div class='table-wrap'><table><thead><tr><th>value</th><th>n</th>{heads}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div></details>"
        )
    return "".join(out)


def _failure_cards(res: BenchResult, data: Any, k: int) -> str:
    f = res.failures(k=k)
    thr = res.cfg.threshold
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
            tags = [
                r.get("category"),
                r.get("noise_bucket"),
                f"{r['snr_db']:.0f} dB" if r.get("snr_db") == r.get("snr_db") else None,
                r.get("noise_category") or None,
                "reverb" if r.get("reverb") else None,
                f"talker:{r['interferer']}" if r.get("interferer") else None,
                r.get("codec") if r.get("codec") not in (None, "none") else None,
                f"{r.get('provider')}/{r.get('voice_name')}"
                if r.get("voice_name")
                else r.get("provider"),
                r.get("accent") if r.get("accent") not in ("", "unknown") else None,
            ]
            badges = "".join(f"<span class='badge'>{_esc(x)}</span>" for x in tags if x)
            cards.append(
                f"<div class='card'><div class='cardhead'><code>{_esc(r['id'])}</code>"
                f"<span class='score'>max {r['max_score']:.2f}</span></div>"
                f"<div class='text'>“{_esc(r.get('text', ''))}”</div><div class='badges'>{badges}</div>"
                f"{_trace_svg(t, s, thr, region if kind == 'missed_positives' else None)}{audio_html}</div>"
            )
        label = "Missed wake words" if kind == "missed_positives" else "False alarms"
        sub = "lowest scores" if kind == "missed_positives" else "highest scores"
        out.append(
            f"<h3>{label} <span class='caps dim'>{sub} · {len(rows)}</span></h3>"
            f"<div class='cards'>{''.join(cards) or '<p class=note>none</p>'}</div>"
        )
    return "".join(out)


def _vars(theme: str) -> str:
    return ";".join(f"--{k}:{v}" for k, v in viz.PALETTE[theme].items())


CSS = f"""
:root{{{_vars("dark")};color-scheme:dark}}
@media (prefers-color-scheme:light){{:root{{{_vars("light")};color-scheme:light}}}}
.light-only{{display:none}}
@media (prefers-color-scheme:light){{.light-only{{display:block}}.dark-only{{display:none}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg0);color:var(--fg1);font:14px/1.55 {viz.SANS}}}
main{{max-width:960px;margin:0 auto;padding:40px 20px 96px}}
h1{{font:400 34px/1.15 {viz.DISPLAY};color:var(--fg0);margin:0 0 6px;letter-spacing:-.01em}}
h2{{font-size:17px;font-weight:600;color:var(--fg0);margin:44px 0 12px;padding-bottom:10px;border-bottom:1px solid var(--separator1)}}
h3{{font-size:14px;font-weight:600;color:var(--fg0);margin:28px 0 10px}}
.caps{{font:500 11px/1 {viz.MONO};letter-spacing:.06em;text-transform:uppercase;color:var(--fg3)}}
.dim{{color:var(--fg4)}} .note{{color:var(--fg3);font-size:12.5px}} .accent{{color:var(--accent)}}
.eyebrow{{font:500 11px {viz.MONO};letter-spacing:.08em;text-transform:uppercase;color:var(--accent);margin-bottom:10px}}
.chart svg{{width:100%;height:auto;display:block;margin:8px 0}}
.table-wrap{{overflow-x:auto;border:1px solid var(--separator1);border-radius:0;background:var(--bg1)}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th{{font:500 10.5px/1.2 {viz.MONO};letter-spacing:.06em;text-transform:uppercase;color:var(--fg3);text-align:left;padding:10px 12px;border-bottom:1px solid var(--separator1);white-space:nowrap}}
td{{padding:7px 12px;border-bottom:1px solid var(--separator1);color:var(--fg2);font-size:13px}}
td:first-child{{white-space:nowrap}} tr:last-child td{{border-bottom:0}} tbody tr:hover td{{background:var(--bg2)}}
td.num{{text-align:right;white-space:nowrap;font-family:{viz.MONO};font-size:12.5px}}
td.strong{{font-weight:600}}
.ci{{color:var(--fg4);font-size:10.5px;margin-left:6px}}
.deco{{display:inline-block;width:1.5px;height:12px;border-radius:0;margin-right:8px;vertical-align:-1px}}
.meter{{display:inline-block;width:56px;height:6px;border-radius:0;background:var(--bg3);margin-right:8px;vertical-align:1px;overflow:hidden}}
.meter i{{display:block;height:100%;background:var(--fg4)}}
.meter.bad i{{background:var(--chart8)}} .meter.good i{{background:var(--success)}}
.val.bad{{color:var(--chart8)}} .val.good{{color:var(--success)}}
details{{border:1px solid var(--separator1);border-radius:0;margin:8px 0;background:var(--bg1)}}
details>summary{{cursor:pointer;padding:12px 14px;list-style:none}}
details>summary::before{{content:'›';display:inline-block;margin-right:10px;color:var(--fg3);transition:transform .15s}}
details[open]>summary::before{{transform:rotate(90deg)}}
details .table-wrap{{border:0;border-top:1px solid var(--separator1);border-radius:0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}}
.card{{background:var(--bg1);border:1px solid var(--separator1);border-radius:0;padding:12px}}
.cardhead{{display:flex;justify-content:space-between;align-items:center;gap:8px}}
.cardhead code{{font:11px {viz.MONO};color:var(--fg3)}}
.score{{font:600 12px {viz.MONO};color:var(--fg0)}}
.text{{margin:6px 0 8px;color:var(--fg1)}}
.badges{{display:flex;flex-wrap:wrap;gap:4px}}
.badge{{font:500 10px/1 {viz.MONO};letter-spacing:.04em;text-transform:uppercase;color:var(--fg3);border:1px solid var(--separator2);border-radius:0;padding:4px 6px}}
.trace{{width:100%;height:64px;margin-top:8px}} .trace .line{{fill:none;stroke:var(--accent);stroke-width:1.5}}
.trace .thr{{stroke:var(--chart8);stroke-dasharray:4 3}} .trace .region{{fill:var(--accent);opacity:.14}}
audio{{width:100%;height:32px;margin-top:8px}}
"""


def render_report(
    results: list[BenchResult],
    out_path: str | Path,
    data: Any = None,
    title: str = "Wake word benchmark",
    failures_per_model: int = 24,
) -> Path:
    body = [
        "<div class='eyebrow'>livekit wakeword bench</div>",
        f"<h1>{_esc(title)}</h1>",
        f"<p class='note'>{len(results)} model(s). Colored values differ significantly from the model's "
        "overall rate (95% CI excludes it); gray means no detectable difference.</p>",
        "<h2>Summary</h2>",
        _summary_table(results),
        "<h2>Recall vs false alarms</h2>",
        _themed(viz.tradeoff, results),
    ]
    for r in results:
        body += [
            f"<h2>{_esc(r.detector)}</h2>",
            _themed(viz.scorecard, r),
            "<h3>Recall by condition</h3>",
            _themed(viz.failures, r),
            "<h3>Recall vs second-voice loudness</h3>",
            _themed(viz.recall_by_talker, r),
            "<h3>Near-miss phrases accepted</h3>",
            _themed(viz.near_misses, r),
            "<h3>False alarms by source</h3>",
            _themed(viz.alarm_sources, r),
        ]
    body += [
        "<h2>All slices</h2>",
        "<p class='note'>Positives: recall per slice. Negatives: share of clips with ≥1 false trigger, "
        "and false triggers per hour. Worst first by CI bound; small numbers are 95% CIs.</p>",
        _breakdowns(results),
    ]
    for r in results:
        body.append(f"<h2>Listen: {_esc(r.detector)}</h2>")
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
