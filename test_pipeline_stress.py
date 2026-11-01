from __future__ import annotations

import sys
from pathlib import Path

from backend.main import _run_pipeline_request
from backend.schemas import PipelineRequest
from app import pdf_report, persist_artifacts


TOPICS = [
    "Multimodal LLM Pruning",
    "Quantization for Edge Devices",
    "Self-Correction in Code Agents",
    "Sparse Attention Mechanisms",
    "Video Diffusion Compression",
]


def main() -> int:
    failures: list[str] = []
    for topic in TOPICS:
        try:
            result = _run_pipeline_request(PipelineRequest(topic=topic))
            pdf = pdf_report(result.paper.model_dump(), result.research.model_dump())
            archive, directory = persist_artifacts(
                result.paper.model_dump(), result.research.model_dump(), pdf
            )
            required = ("paper.pdf", "paper.tex", "benchmark_matrix.json", "artifacts.zip")
            missing = [name for name in required if not (directory / name).exists()]
            if not archive or missing:
                failures.append(f"{topic}: missing {missing or 'ZIP bytes'}")
                continue
            print(f"PASS: {topic} -> {directory}")
        except Exception as exc:
            failures.append(f"{topic}: {exc}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
