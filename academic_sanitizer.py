from __future__ import annotations

import re
import unicodedata


_VIETNAMESE_REPHRASES = {
    "tối ưu hóa": "optimization",
    "tối ưu": "optimization",
    "trên": "on",
    "thiết bị nhúng": "embedded devices",
    "edge": "edge",
    "mô hình": "model",
    "phát hiện": "detection",
    "gian lận": "fraud",
    "tài chính": "financial",
    "dùng": "using",
    "sử dụng": "using",
    "học sâu": "deep learning",
    "mạng nơ-ron đồ thị": "graph neural networks",
    "đồ thị": "graph",
    "năng lượng": "energy",
    "hệ thống": "systems",
    "cho": "for",
}

_SHORT_TOPIC_EXPANSIONS = {
    "pruning": "Dynamic Pruning Methods for Resource-Constrained Inference",
    "quantization": "Adaptive Quantization Methods for Efficient Neural Inference",
    "transformer": "Efficient Transformer Architectures for Resource-Constrained Inference",
    "attention": "Adaptive Attention Mechanisms for Efficient Neural Inference",
    "classification": "Robust Neural Classification under Resource Constraints",
    "fraud": "Deep Learning Methods for Financial Fraud Detection",
}


def sanitize_academic_topic(topic: str) -> str:
    """Return a safe, readable topic while preserving meaning and Unicode."""
    value = unicodedata.normalize("NFKC", str(topic or ""))
    value = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    value = value.replace("<", " ").replace(">", " ")
    value = re.sub(r"[`$#%&_{}\\]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" .,:;|")
    lowered = value.casefold()
    for source, target in sorted(_VIETNAMESE_REPHRASES.items(), key=lambda item: -len(item[0])):
        value = re.sub(re.escape(source), target, value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value.split()) < 3:
        value = _SHORT_TOPIC_EXPANSIONS.get(
            lowered,
            f"{value.title()} Methods for Resource-Constrained Research and Deployment",
        )
    return value[:240].strip(" .,:;")
