from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from wai_r0.core.reproducibility import atomic_write_text
from wai_r0.reporting.schema import ResearchReport


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "—"
    return str(value)


def render_markdown(report: ResearchReport) -> str:
    report.validate()
    lines = [
        f"# WAI-R0 Research Report — {report.identity.run_id}",
        "",
        f"**Decision:** `{report.decision}`  ",
        f"**Evidence class:** `{report.evidence_class}`  ",
        f"**Created:** {report.identity.created_at_utc}  ",
        f"**WAI-R0:** `{report.identity.wai_r0_version}`",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in report.limitations],
        "",
        "## Decision gates",
        "",
        "| Gate | Status | Metric | Observed | Threshold | Explanation |",
        "|---|---|---|---:|---:|---|",
    ]
    for gate in report.gates:
        lines.append(
            "| "
            + " | ".join(
                (
                    gate.name,
                    gate.status,
                    gate.metric or "—",
                    _format_value(gate.observed),
                    _format_value(gate.threshold),
                    gate.explanation.replace("|", "\\|"),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Failures",
            "",
            *([f"- {item}" for item in report.failures] or ["- None recorded."]),
            "",
            "## Provenance",
            "",
            f"- Config hash: `{report.identity.config_hash}`",
            f"- Experiment hash: `{report.identity.experiment_hash or 'n/a'}`",
            f"- Git commit: `{report.identity.git_commit or 'unavailable'}`",
            f"- Dirty tree: `{report.identity.git_dirty}`",
            f"- Command: `{' '.join(report.identity.command)}`",
            "",
            "## Metrics",
            "",
            "```json",
            json.dumps(
                report.metrics, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            ),
            "```",
            "",
            "## Resolved configuration",
            "",
            "```json",
            json.dumps(
                report.resolved_config,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            "```",
            "",
            "## Environment",
            "",
            "```json",
            json.dumps(
                {"hardware": report.hardware, "software": report.software},
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(report: ResearchReport) -> str:
    report.validate()
    gate_rows = "".join(
        "<tr>"
        f"<td>{html.escape(gate.name)}</td>"
        f"<td><span class='status {html.escape(gate.status)}'>{html.escape(gate.status)}</span></td>"
        f"<td>{html.escape(gate.metric or '—')}</td>"
        f"<td>{html.escape(_format_value(gate.observed))}</td>"
        f"<td>{html.escape(_format_value(gate.threshold))}</td>"
        f"<td>{html.escape(gate.explanation)}</td>"
        "</tr>"
        for gate in report.gates
    )
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in report.limitations)
    failures = (
        "".join(f"<li>{html.escape(item)}</li>" for item in report.failures)
        or "<li>None recorded.</li>"
    )
    metrics = html.escape(
        json.dumps(report.metrics, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )
    config = html.escape(
        json.dumps(
            report.resolved_config, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
    )
    environment = html.escape(
        json.dumps(
            {"hardware": report.hardware, "software": report.software},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    provenance = html.escape(
        json.dumps(report.provenance, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WAI-R0 report {html.escape(report.identity.run_id)}</title>
<style>
:root {{ color-scheme: light dark; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
body {{ margin: 0; background: #101010; color: #f1f1f1; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 48px 24px 96px; }}
h1 {{ font-size: clamp(1.7rem, 4vw, 3rem); letter-spacing: -.05em; margin: 0 0 12px; }}
h2 {{ margin-top: 48px; border-top: 1px solid #383838; padding-top: 20px; }}
.meta {{ color: #aaa; line-height: 1.7; }}
.decision {{ display: inline-block; border: 1px solid #777; padding: 5px 9px; text-transform: uppercase; }}
table {{ border-collapse: collapse; width: 100%; overflow-x: auto; display: block; }}
th, td {{ border: 1px solid #383838; padding: 10px; text-align: left; vertical-align: top; }}
th {{ background: #1c1c1c; }}
.status {{ text-transform: uppercase; }}
.status.pass {{ color: #9fe6a0; }} .status.fail {{ color: #ff9a9a; }} .status.not_run {{ color: #aaa; }} .status.inconclusive {{ color: #ddd28d; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #181818; border: 1px solid #383838; padding: 16px; line-height: 1.45; }}
.warning {{ border-left: 4px solid #ddd28d; padding: 2px 18px; background: #1c1b14; }}
</style>
</head>
<body><main>
<h1>WAI-R0 Research Report</h1>
<p class="decision">{html.escape(report.decision)}</p>
<p class="meta">run {html.escape(report.identity.run_id)} · {html.escape(report.evidence_class)}<br>
{html.escape(report.identity.created_at_utc)} · WAI-R0 {html.escape(report.identity.wai_r0_version)}</p>
<section class="warning"><h2>Limitations</h2><ul>{limitations}</ul></section>
<h2>Decision gates</h2>
<table><thead><tr><th>Gate</th><th>Status</th><th>Metric</th><th>Observed</th><th>Threshold</th><th>Explanation</th></tr></thead><tbody>{gate_rows}</tbody></table>
<h2>Failures</h2><ul>{failures}</ul>
<h2>Provenance</h2><pre>{provenance}</pre>
<h2>Metrics</h2><pre>{metrics}</pre>
<h2>Resolved configuration</h2><pre>{config}</pre>
<h2>Environment</h2><pre>{environment}</pre>
</main></body></html>"""


def write_rendered_report(
    path: str | Path,
    report: ResearchReport,
    *,
    format: str | None = None,
) -> Path:
    destination = Path(path)
    resolved = (format or destination.suffix.lstrip(".")).lower()
    if resolved in {"md", "markdown"}:
        content = render_markdown(report)
    elif resolved in {"html", "htm"}:
        content = render_html(report)
    else:
        raise ValueError("report format must be markdown or html")
    return atomic_write_text(destination, content)


__all__ = ["render_html", "render_markdown", "write_rendered_report"]
