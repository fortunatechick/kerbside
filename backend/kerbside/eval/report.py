"""The report: one button, one self-contained HTML file.

Self-contained means self-contained. Every figure is an inline data URI and
every style is inline, so the file can be emailed, opened offline, or printed
without dragging a directory of assets behind it. A report that only renders on
the machine that produced it is not a report.

The parameters and the calibration used are embedded verbatim, because a result
without the settings that produced it cannot be reproduced or argued with -
which is the whole purpose of the document.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from kerbside.api.session import Session

__all__ = ["build_report", "render_report_html"]


def _data_uri(frame: np.ndarray, quality: int = 78) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def build_report(session: Session) -> dict[str, Any]:
    """Gather everything the report needs, as plain data."""
    from kerbside.eval.analytics import build_analytics

    analytics = build_analytics(session)
    stages = session.pipeline.describe()

    figures: list[dict[str, str]] = []
    if session.frames:
        middle = len(session.frames) // 2
        try:
            ctx = session.context(middle)
            from kerbside.api.overlay import OverlayOptions, draw_overlay

            figures.append(
                {
                    "caption": "Annotated frame: boxes, ground points and counting lines.",
                    "uri": _data_uri(
                        draw_overlay(ctx.bgr, ctx, OverlayOptions(), session.calibration)
                    ),
                }
            )
            if ctx.foreground is not None:
                figures.append(
                    {
                        "caption": "Foreground mask after S4, before shadow suppression.",
                        "uri": _data_uri(
                            cv2.cvtColor(
                                (ctx.foreground.astype(np.uint8)) * 255, cv2.COLOR_GRAY2BGR
                            )
                        ),
                    }
                )
            if ctx.shadow is not None and ctx.shadow.any():
                tinted = ctx.bgr.copy()
                tinted[ctx.shadow] = (250, 139, 167)
                figures.append(
                    {
                        "caption": "Pixels S5 classified as shadow, which is what "
                        "splits vehicles a shadow had bridged.",
                        "uri": _data_uri(tinted),
                    }
                )
        except Exception:  # noqa: BLE001 - a missing figure must not lose the report
            pass

    calibration = None
    if session.calibration is not None:
        from kerbside.geometry.calibration import assess_calibration

        report = assess_calibration(session.calibration)
        calibration = {
            "homography": np.asarray(session.calibration.h_world_to_img).tolist(),
            "verdict": report.verdict,
            "rmse_px": report.reprojection_rmse_px,
            "camera_height_m": report.camera.height_m,
            "camera_tilt_deg": report.camera.tilt_deg,
            "focal_px": report.camera.focal_px,
            "warnings": list(report.warnings),
        }

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": session.state.to_dict(),
        "analytics": analytics,
        "stages": stages,
        "calibration": calibration,
        "timings": session.pipeline.timings.summary(),
        "figures": figures,
    }


_CSS = """
:root{--bg:#0B0F14;--surface:#121821;--raised:#1A2230;--line:#2A3644;--text:#E8EEF4;
--muted:#98A6B5;--dim:#63717F;--accent:#3FA7D6;--ok:#2FBF71;--warn:#E8B931;--stop:#E5484D;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);padding:0 0 80px;
font:400 15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:960px;margin:0 auto;padding:0 20px}
header{border-bottom:1px solid var(--line);padding:48px 0 32px;margin-bottom:8px;
background:linear-gradient(180deg,var(--surface),var(--bg))}
h1{font-size:34px;margin:0 0 10px;letter-spacing:-.02em}
h2{font-size:22px;margin:44px 0 12px}
h3{font-size:16px;margin:24px 0 8px}
.eyebrow{font:600 11px/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;
color:var(--accent);margin-bottom:14px}
table{border-collapse:collapse;width:100%;font-size:14px;margin:14px 0}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid #1E2733;vertical-align:top}
th{background:var(--raised);font:600 11px/1.4 var(--mono);letter-spacing:.07em;
text-transform:uppercase;color:var(--muted)}
td.num,th.num{text-align:right;font-family:var(--mono);font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto;border:1px solid #1E2733;border-radius:8px;margin:14px 0}
code,.mono{font-family:var(--mono);font-variant-numeric:tabular-nums}
pre{background:var(--raised);border:1px solid #1E2733;border-radius:8px;padding:12px 14px;
overflow-x:auto;font:400 12px/1.6 var(--mono);color:var(--muted)}
.panel{background:var(--surface);border:1px solid #1E2733;border-radius:10px;padding:16px 18px;margin:16px 0}
.panel.warn{border-color:#4a2b2e;background:#171214}
.grid{display:grid;gap:12px;margin:16px 0}
@media(min-width:700px){.grid.three{grid-template-columns:repeat(3,1fr)}}
.stat{background:var(--surface);border:1px solid #1E2733;border-radius:8px;padding:12px 14px}
.stat .k{font:600 10px/1.3 var(--mono);letter-spacing:.08em;text-transform:uppercase;
color:var(--dim);margin-bottom:5px}
.stat .v{font:600 24px/1.1 var(--mono);font-variant-numeric:tabular-nums}
figure{margin:18px 0}figure img{width:100%;border:1px solid #1E2733;border-radius:8px;display:block}
figcaption{font-size:13px;color:var(--dim);margin-top:7px}
footer{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);color:var(--dim);font-size:13px}
@media print{body{background:#fff;color:#000}header{background:none}
.panel,.stat,table,figure img{border-color:#ccc}th{background:#eee;color:#333}}
"""


def render_report_html(report: dict[str, Any]) -> str:
    """Render the gathered data as one self-contained HTML document."""
    a = report["analytics"]
    source = report["source"]
    esc = html.escape

    def rows(items: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
        head = "".join(f'<th class="num">{esc(label)}</th>' for _key, label in columns)
        body = ""
        for item in items:
            cells = "".join(
                f'<td class="num">{esc(str(item.get(key, "")))}</td>' for key, _l in columns
            )
            body += f"<tr>{cells}</tr>"
        return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

    figures = "".join(
        f'<figure><img src="{f["uri"]}" alt="{esc(f["caption"])}">'
        f"<figcaption>{esc(f['caption'])}</figcaption></figure>"
        for f in report["figures"]
        if f["uri"]
    )

    stage_rows = "".join(
        "<tr><td><strong>{name}</strong><br><span style='color:var(--dim);font-size:13px'>{rat}</span></td>"
        "<td class='mono'>{units}</td><td><pre style='margin:0'>{params}</pre></td></tr>".format(
            name=esc(s["name"]),
            rat=esc(s["rationale"]),
            units=", ".join(str(u) for u in s["units"]),
            params=esc(json.dumps(s["values"], indent=1)),
        )
        for s in report["stages"]
    )

    calibration_block = "<p>No calibration was loaded, so no real-world quantity in this report can be trusted.</p>"
    if report["calibration"]:
        c = report["calibration"]
        calibration_block = (
            f"<p>{esc(c['verdict'])}</p>"
            f"<p class='mono'>implied camera: {c['camera_height_m']:.2f} m high, "
            f"{c['camera_tilt_deg']:.2f}&deg; below horizontal, focal {c['focal_px']:.0f} px</p>"
            f"<pre>{esc(json.dumps(c['homography'], indent=1))}</pre>"
        )

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kerbside report &mdash; {esc(source['source_name'])}</title>
<style>{_CSS}</style></head><body>
<header><div class="wrap">
<div class="eyebrow">Kerbside &middot; classical DIP traffic analytics</div>
<h1>Analysis report</h1>
<p style="color:var(--muted);margin:0">{esc(source['source_name'])} &mdash;
{source['width_px']}&times;{source['height_px']} at {source['measured_fps']:.3f} fps measured,
{source['container_fps']:.3f} fps declared &middot; generated {esc(report['generated_utc'])}</p>
</div></header>
<div class="wrap">

<div class="panel warn"><strong>Not an enforcement device.</strong>
{esc(a['disclaimer'])}</div>

<h2>Summary</h2>
<div class="grid three">
<div class="stat"><div class="k">Vehicles counted</div><div class="v">{a['total_count']}</div></div>
<div class="stat"><div class="k">85th percentile speed</div><div class="v">{a['speed']['p85_kmph']:.0f}</div></div>
<div class="stat"><div class="k">Mean speed</div><div class="v">{a['speed']['mean_kmph']:.0f}</div></div>
</div>
<p>Counts by lane: <span class="mono">{esc(json.dumps(a['counts_by_lane']))}</span><br>
Counts by class: <span class="mono">{esc(json.dumps(a['counts_by_class']))}</span></p>

<h2>Calibration used</h2>{calibration_block}

<h2>Figures</h2>{figures or "<p>No figures were available for this run.</p>"}

<h2>Vehicles</h2>
{rows(a['vehicles'], [('track_id','Track'),('lane_id','Lane'),('direction','Dir'),
('time_s','t (s)'),('speed_kmph','km/h'),('vehicle_class','Class'),('colour','Colour'),
('length_m','L (m)'),('width_m','W (m)')])}

<h2>Flagged for review</h2>
<p>Above the configured limit of {a['speed']['limit_kmph']:.0f} km/h. Flagged for review only &mdash;
never a violation, and carrying no evidentiary standing.</p>
{rows(a['flagged'], [('track_id','Track'),('lane_id','Lane'),('time_s','t (s)'),('speed_kmph','km/h')])
 if a['flagged'] else '<p class="mono" style="color:var(--dim)">none</p>'}

<h2>Flow over time</h2>
{rows(a['intervals'], [('t_s','t (s)'),('count','Count'),('flow_veh_per_hour','veh/hr'),
('mean_speed_kmph','Mean km/h')])}

<h2>Pipeline configuration</h2>
<p>Every parameter that produced the numbers above, so the run can be reproduced
or disagreed with.</p>
<div class="scroll"><table><thead><tr><th>Stage</th><th>Units</th><th>Parameters</th></tr></thead>
<tbody>{stage_rows}</tbody></table></div>

<h2>Timing</h2>
<pre>{esc(json.dumps({k: round(v, 2) for k, v in report['timings'].items()}, indent=1))}</pre>

<footer>Generated by Kerbside. Every figure is reproducible from the source and the
parameters recorded above. Synthetic and real-footage results are never conflated.</footer>
</div></body></html>"""
