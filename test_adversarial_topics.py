from __future__ import annotations

import app
from academic_sanitizer import sanitize_academic_topic
from backend.main import _run_pipeline_request
from backend.schemas import PipelineRequest


ADVERSARIAL_TOPICS = [
    "Tối ưu hóa Transformer trên thiết bị nhúng % $ # & _",
    "Pruning",
    "Quantum Federated Learning for Micro-Grid Energy Systems in 2026",
    "Graph Neural Networks with 99.99% Compression",
    "Mô hình phát hiện gian lận tài chính dùng Deep Learning",
    "<<< XML-like topic >>>",
    "Attention & Memory_Tradeoffs #Edge $Systems",
    "A",
    "Graph Neural Networks {with} \\ unsafe commands",
    "Video diffusion: latency < 20ms & memory > accuracy",
]


def main() -> int:
    failures: list[str] = []
    for index, raw_topic in enumerate(ADVERSARIAL_TOPICS, 1):
        try:
            sanitized = sanitize_academic_topic(raw_topic)
            result = _run_pipeline_request(PipelineRequest(topic=raw_topic))
            pdf = app.pdf_report(result.paper.model_dump(), result.research.model_dump())
            if not sanitized or len(pdf) < 1000:
                raise RuntimeError("empty sanitized topic or PDF")
            print(f"[TEST {index}/{len(ADVERSARIAL_TOPICS)}] PASS: {sanitized} ({len(pdf)} bytes)")
        except Exception as error:
            failures.append(f"[TEST {index}] {raw_topic}: {error}")
            print(f"[TEST {index}/{len(ADVERSARIAL_TOPICS)}] FAIL: {error}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
