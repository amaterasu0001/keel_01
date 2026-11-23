"""Tables and charts for the thesis, written from the JSON the experiments produce.

No plotting library is required: charts are plain SVG, tables are CSV and Markdown. Every figure
can be rebuilt with `keel eval thesis ...` from the records alone.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PALETTE = ["#1d5b80", "#b0342b", "#1f7a4d", "#9a6a12", "#6b4fa0", "#4a6572", "#c06a2b", "#2b8a8a"]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _md_table(rows: list[dict], fmt=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)) -> str:
    if not rows:
        return "_no rows_\n"
    fields = list(rows[0].keys())
    out = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    out += ["| " + " | ".join(fmt(r[f]) if r[f] is not None else "–" for f in fields) + " |" for r in rows]
    return "\n".join(out) + "\n"


def method_rows(methods: dict, metric_keys=("fdr", "viol_fdr", "merge")) -> list[dict]:
    rows = []
    for name, rs in methods.items():
        for r in rs:
            rows.append({"method": name, "alpha": r["alpha"], **{k: r[k] for k in metric_keys}})
    return rows


def line_chart(series: dict[str, list[tuple[float, float]]], title: str, xlabel: str, ylabel: str,
               diagonal: bool = False, width: int = 640, height: int = 400, ymax: float | None = None) -> str:
    """A minimal SVG line chart. diagonal=True draws y = x (the target line for FMR vs alpha)."""
    pts = [p for s in series.values() for p in s]
    if not pts:
        return "<svg xmlns='http://www.w3.org/2000/svg'/>"
    x0, x1 = 0.0, max(p[0] for p in pts) * 1.05
    y1 = ymax if ymax is not None else max(max(p[1] for p in pts), x1 if diagonal else 0) * 1.1 or 1.0
    ml, mr, mt, mb = 60, 160, 40, 50
    W, H = width - ml - mr, height - mt - mb

    def X(v):
        return ml + (v - x0) / (x1 - x0 or 1) * W

    def Y(v):
        return mt + H - v / (y1 or 1) * H

    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
             f"font-family='sans-serif' font-size='12'>",
             f"<rect width='{width}' height='{height}' fill='white'/>",
             f"<text x='{ml}' y='22' font-size='14' font-weight='600'>{title}</text>"]
    for i in range(6):
        yv = y1 * i / 5
        parts.append(f"<line x1='{ml}' x2='{ml + W}' y1='{Y(yv):.1f}' y2='{Y(yv):.1f}' stroke='#e4e9ec'/>")
        parts.append(f"<text x='{ml - 6}' y='{Y(yv) + 4:.1f}' text-anchor='end' fill='#556572'>{yv:.2f}</text>")
        xv = x0 + (x1 - x0) * i / 5
        parts.append(f"<text x='{X(xv):.1f}' y='{mt + H + 18}' text-anchor='middle' fill='#556572'>{xv:.2f}</text>")
    parts.append(f"<line x1='{ml}' x2='{ml + W}' y1='{mt + H}' y2='{mt + H}' stroke='#556572'/>")
    parts.append(f"<text x='{ml + W / 2}' y='{height - 10}' text-anchor='middle'>{xlabel}</text>")
    parts.append(f"<text x='16' y='{mt + H / 2}' transform='rotate(-90 16 {mt + H / 2})' "
                 f"text-anchor='middle'>{ylabel}</text>")
    if diagonal:
        parts.append(f"<line x1='{X(0):.1f}' y1='{Y(0):.1f}' x2='{X(min(x1, y1)):.1f}' y2='{Y(min(x1, y1)):.1f}' "
                     "stroke='#9aabb7' stroke-dasharray='5 4'/>")
    for i, (name, s) in enumerate(series.items()):
        col = PALETTE[i % len(PALETTE)]
        s = sorted(s)
        d = " ".join(f"{'M' if j == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for j, (x, y) in enumerate(s))
        parts.append(f"<path d='{d}' fill='none' stroke='{col}' stroke-width='2'/>")
        for x, y in s:
            parts.append(f"<circle cx='{X(x):.1f}' cy='{Y(y):.1f}' r='3' fill='{col}'/>")
        ly = mt + 14 + i * 18
        parts.append(f"<rect x='{ml + W + 14}' y='{ly - 9}' width='12' height='3' fill='{col}'/>")
        parts.append(f"<text x='{ml + W + 32}' y='{ly - 4}'>{name}</text>")
    parts.append("</svg>")
    return "\n".join(parts)


def write_thesis_outputs(results: dict, out_dir: str | Path) -> list[Path]:
    """results: {"inflation": ..., "correction": ..., "budget": [...]} as built by `keel eval thesis`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = [out / "results.json"]
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    md = ["# Thesis results", "", f"Records: {results.get('n_bugs')} bugs. "
          f"Evaluation oracle: `{results.get('eval_label')}`. Splits: {results.get('n_splits')}.", ""]

    inf = results.get("inflation")
    if inf:
        rows = []
        series = {}
        for oracle, rs in inf["by_oracle"].items():
            series[f"judged by {oracle}"] = [(r["alpha"], r["fdr"]) for r in rs]
            rows += [{"oracle": oracle, "alpha": r["alpha"], "fdr": r["fdr"], "viol_fdr": r["viol_fdr"],
                      "merge": r["merge"]} for r in rs]
        _write_csv(out / "rq1_inflation.csv", rows)
        (out / "rq1_inflation.svg").write_text(line_chart(series, "RQ1: plain LTT, calibrated on held-out tests",
                                                          "target α", "realised false-merge rate", True),
                                               encoding="utf-8")
        written += [out / "rq1_inflation.csv", out / "rq1_inflation.svg"]
        md += ["## RQ1: inflation", "", _md_table(rows), "![RQ1](rq1_inflation.svg)", ""]

    cor = results.get("correction")
    if cor:
        rows = method_rows(cor["methods"])
        _write_csv(out / "rq3_correction.csv", rows)
        keep = [m for m in ("plain", "plain_abs", "A", "B", "uncalibrated_score") if m in cor["methods"]]
        fmr = {m: [(r["alpha"], r["fdr"]) for r in cor["methods"][m]] for m in keep}
        viol = {m: [(r["alpha"], r["viol_fdr"]) for r in cor["methods"][m]] for m in keep}
        merge = {m: [(r["alpha"], r["merge"]) for r in cor["methods"][m]] for m in keep}
        lab = cor.get("eval_label", "")
        for name, ser, yl, diag, ymax in (("rq3_fmr", fmr, f"false-merge rate ({lab})", True, None),
                                          ("rq3_violation", viol, f"P(FMR > α) ({lab})", False, 1.0),
                                          ("rq3_merge", merge, "share auto-merged", False, 1.0)):
            (out / f"{name}.svg").write_text(line_chart(ser, f"RQ3: {yl}", "target α", yl, diag, ymax=ymax),
                                             encoding="utf-8")
            written.append(out / f"{name}.svg")
        written.append(out / "rq3_correction.csv")
        md += ["## RQ3: correction", "", f"Audit pool: {cor.get('n_audit_pool')} bugs; δ = {cor.get('delta')}.", "",
               _md_table(rows), "![FMR](rq3_fmr.svg) ![violation](rq3_violation.svg) ![merge](rq3_merge.svg)", ""]

    bud = results.get("budget")
    if bud:
        _write_csv(out / "rq4_budget.csv", bud)
        ser = {}
        for v in ("A", "B"):
            if any(f"{v}_merge" in r for r in bud):
                ser[f"variant {v}"] = [(r["budget"], r[f"{v}_merge"]) for r in bud if f"{v}_merge" in r]
        (out / "rq4_budget.svg").write_text(line_chart(ser, "RQ4: audit budget vs automation", "audited bugs",
                                                       "share auto-merged", ymax=1.0), encoding="utf-8")
        written += [out / "rq4_budget.csv", out / "rq4_budget.svg"]
        md += ["## RQ4: audit budget", "", _md_table(bud), "![RQ4](rq4_budget.svg)", ""]

    (out / "README.md").write_text("\n".join(md), encoding="utf-8")
    written.append(out / "README.md")
    return written
