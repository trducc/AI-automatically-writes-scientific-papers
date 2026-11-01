from __future__ import annotations

import re
from typing import Any


_EVIDENCE_LABELS = {
    "measured": "Measured",
    "literature baseline / planned": "Literature Baseline / Planned",
    "controlled simulation": "Controlled Simulation",
    "planned / unverified": "Planned / Unverified",
    "estimated": "Estimated",
    "proposed": "Proposed",
    "unresolved": "Unresolved",
}


def _normalize_evidence(value: Any, role: str) -> str:
    normalized = str(value or "").strip().lower()
    for key, label in _EVIDENCE_LABELS.items():
        if normalized == key:
            return label
    return "Measured" if role == "Measured" else "Controlled Simulation"


def audit_manuscript(paper: dict[str, Any], research: dict[str, Any]) -> dict[str, Any]:
    """Audit formulas, metric relationships, and evidence labels without inventing data."""
    issues: list[dict[str, str]] = []
    formulas = paper.get("mathematical_formulas", [])
    for index, formula in enumerate(formulas, 1):
        value = str(formula)
        if value.count("{") != value.count("}") or value.count("[") != value.count("]"):
            issues.append({"type": "latex", "location": f"formula_{index}", "message": "Unbalanced delimiters."})
        if not re.search(r"[A-Za-z]", value):
            issues.append({"type": "latex", "location": f"formula_{index}", "message": "Formula has no identifiable symbols."})

    rows = research.get("benchmark_table", [])
    normalized_rows = []
    for row in rows:
        role = str(row.get("role", "Baseline"))
        row["evidence_type"] = _normalize_evidence(row.get("evidence_type"), role)
        normalized_rows.append(row)
    proposed = [row for row in normalized_rows if row.get("role") == "Proposed"]
    baselines = [row for row in normalized_rows if row.get("role") == "Baseline"]
    if proposed and baselines:
        proposed_row = proposed[0]
        best_baseline_latency = min(float(row.get("latency_ms", 0)) for row in baselines)
        if float(proposed_row.get("latency_ms", 0)) > best_baseline_latency * 1.5:
            issues.append({
                "type": "benchmark",
                "location": "benchmark_table",
                "message": "Proposed latency is materially higher than the best baseline; efficiency claims need review.",
            })
        best_baseline_flops = min(float(row.get("flops_g", 0)) for row in baselines)
        if float(proposed_row.get("flops_g", 0)) > best_baseline_flops * 1.5:
            issues.append({
                "type": "benchmark",
                "location": "benchmark_table",
                "message": "Proposed FLOPs are materially higher than the best baseline; do not claim compute reduction.",
            })

    return {
        "status": "passed" if not issues else "needs_review",
        "issues": issues,
        "formula_count": len(formulas),
        "row_count": len(normalized_rows),
        "evidence_labels": sorted({row["evidence_type"] for row in normalized_rows}),
    }
