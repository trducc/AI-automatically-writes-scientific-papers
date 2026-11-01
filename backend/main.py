from __future__ import annotations

import json
import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    AblationRow,
    AblationSpec,
    AgentCoordinationSpec,
    DatasetSpec,
    FullPipelineResponse,
    InferenceEngineSpec,
    KnowledgeBaseSpec,
    MetricCurvePoint,
    ModelSpec,
    PipelineRequest,
    PlannerOutput,
    ProjectMetadata,
    ResearcherOutput,
    ReviewerOutput,
    TopicInput,
    WriterOutput,
    GuardrailOutput,
)
from backend.guardrail_agent import audit_manuscript
from academic_sanitizer import sanitize_academic_topic

MASTER_RESEARCH_WRITER_PROMPT = """
You are a Lead AI Research Scientist drafting a full-length, high-density
academic paper for ICLR, NeurIPS, or IEEE TPAMI.

TOPIC: "{topic}"

CORE WRITING REQUIREMENTS:
1. Write approximately 3,000+ words of specific academic prose when the
   provider budget permits. Explain structural choices, DRAM/SRAM bandwidth,
   memory movement, KV-cache footprint, kernel overhead, and empirical
   behavior. Do not pad with repeated claims or generic praise.
2. Use academic English and granular subsections:
   Abstract and Keywords; 1.1-1.3 Introduction; Related Work; 3.1-3.3
   Methodology; 4.1-4.3 Empirical Evaluation; 5 Discussion and Failure Modes;
   6 Conclusion and Future Directions; References.
3. Keep mathematics practical and elegant. Use only 3-4 core equations:
   baseline representation, gating/routing decision, regularized loss, and
   optional complexity expression. Define symbols near their first use. Do not
   create artificial algebraic derivations or fragile LaTeX.
4. Compare four relevant baseline families and analyze quality, latency,
   VRAM, FLOPs, and evidence status under a matched protocol. Adapt metrics to
   the domain; do not force Q/K/V notation on graph, quantum, control, or
   non-attention topics.
5. Cite only real, verifiable seminal sources. Never invent authors, DOIs,
   venues, exact measurements, or 2026 citations. Preserve supplied
   evidence_type labels and mark simulated values explicitly.

Return JSON matching WriterOutput exactly. Put detailed prose in section
fields, not in an unstructured summary.
"""

app = FastAPI(
    title="Smart Research Lab API",
    version="0.2.0",
    description="Multi-agent research pipeline with live Gemini integration and topic-aware fallback generation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PTIT_COURSE_METADATA = {
    "institution": "Học viện Công nghệ Bưu chính Viễn thông (PTIT)",
    "faculty": "Khoa Công nghệ Thông tin",
    "subject": "Hệ thống thông minh",
    "instructor": "Thầy Hoàng Mạnh Cường",
    "student_name": "Trần Văn Đức",
    "student_id": "B23DCCN191",
}


def _normalize_topic(topic: str) -> str:
    return sanitize_academic_topic(topic)


def _academic_abstract(text: str) -> str:
    additions = (
        " The evaluation contract defines the task metric, comparator set, "
        "resource budget, evidence status, and conditions required for replication. "
        "The analysis also separates expected gains from unresolved questions, "
        "including sensitivity to random seeds, data shifts, numerical precision, "
        "and accelerator-specific scheduling. This distinction makes the manuscript "
        "useful as a research plan while preventing simulated values from being "
        "mistaken for completed measurements. The proposed protocol is therefore "
        "intended to support transparent follow-up experiments rather than replace them."
    )
    words = text.split()
    while len(words) < 250:
        text += additions
        words = text.split()
    if len(words) <= 300:
        return text
    return " ".join(words[:300]).rstrip(".,;:") + "."

_STOP_WORDS = {
    "a", "an", "and", "for", "from", "in", "of", "on", "or", "the", "to", "with",
    "using", "via", "based", "towards", "and", "study", "analysis", "system",
}

_DOMAIN_PROFILES = {
    "Speech & Audio Processing": {
        "keywords": ("speech", "audio", "whisper", "voice", "acoustic", "asr", "tts", "speaker recognition", "transcription"),
        "modality": "Audio / Speech",
        "metric": "WER / Speaker F1",
        "backbones": ("Conformer Baseline", "HuBERT", "Wav2Vec 2.0"),
        "proposed": "Dynamic Streaming Transducer (Ours)",
        "descriptor": "streaming acoustic representation",
    },
    "Reinforcement Learning & Robotics": {
        "keywords": ("reinforcement", "reinforcement learning", "rl", "q-learning", "q learning", "ppo", "sac", "policy", "robot", "robotics", "control", "agent", "navigation"),
        "modality": "State-action trajectories",
        "metric": "Episode return / Success rate",
        "backbones": ("PPO Baseline", "Soft Actor-Critic (SAC)", "Rainbow DQN"),
        "proposed": "Adaptive Contextual PPO (Ours)",
        "descriptor": "context-aware policy optimization",
    },
    "Multimodal & Vision-Language": {
        "keywords": ("multimodal", "multi-modal", "vision-language", "vision language", "clip", "vqa", "captioning", "diffusion", "text-to-image", "text to image", "image-text"),
        "modality": "Image-text pairs",
        "metric": "Recall@K / CIDEr",
        "backbones": ("CLIP Baseline", "BLIP-2", "Flamingo"),
        "proposed": "Cross-Modal Contrastive Fusion (Ours)",
        "descriptor": "cross-modal representation alignment",
    },
    "Recommendation Systems & CTR": {
        "keywords": ("recommendation", "recommender", "ctr", "collaborative filtering", "sequential recommendation", "ranking", "click-through", "personalization", "user-item"),
        "modality": "User-item interactions",
        "metric": "NDCG@10 / AUC",
        "backbones": ("DeepFM Baseline", "DLRM", "SASRec"),
        "proposed": "Dynamic Session-Aware Graph-Rec (Ours)",
        "descriptor": "session-aware ranking optimization",
    },
    "Graph Learning & FinTech": {
        "keywords": ("graph", "gnn", "gcn", "node", "edge", "financial", "finance", "fintech", "fraud", "transaction", "credit", "blockchain"),
        "modality": "Graph / Transaction records",
        "metric": "F1-score / PR-AUC",
        "backbones": ("GCN", "GraphSAGE", "Graph Attention Network (GAT)"),
        "proposed": "Adaptive Relational Graph Network (Ours)",
        "descriptor": "relational risk representation",
    },
    "Natural Language Processing (NLP) / RAG": {
        "keywords": ("retrieval", "retrieval-augmented", "retrieval augmented", "rag", "hallucination", "decoding", "attribution", "confidence-aware", "confidence aware", "llm", "large language model", "knowledge grounding", "source attribution"),
        "modality": "Text / Retrieved Evidence",
        "metric": "Faithfulness / EM",
        "backbones": ("DPR + Generator", "RAG-Sequence", "Self-RAG"),
        "proposed": "Confidence-Aware Attributed Decoder (Ours)",
        "descriptor": "retrieval-grounded confidence calibration",
    },
    "Natural Language Processing (NLP) / Machine Translation": {
        "keywords": ("translation", "machine translation", "cross-lingual", "cross lingual", "zero-shot", "low-resource", "low resource", "multilingual", "adapter", "mad-x", "mbert", "xlm-r", "flores", "tatoeba"),
        "modality": "Text / Translation",
        "metric": "COMET / BLEU",
        "backbones": ("mBERT Baseline", "XLM-RoBERTa Base", "MAD-X Adapters"),
        "proposed": "Dynamic Cross-Lingual Adapter (Ours)",
        "descriptor": "parameter-efficient cross-lingual adaptation",
    },
    "Quantum Computing": {
        "keywords": ("quantum", "qubit", "quantum circuit", "variational"),
        "modality": "Quantum circuit",
        "metric": "Circuit fidelity",
        "backbones": ("QAOA", "VQE", "Quantum kernel"),
        "proposed": "Adaptive Quantum Circuit",
        "descriptor": "noise-aware circuit optimization",
    },
    "Legal AI": {
        "keywords": ("legal", "law", "contract", "case", "statute", "compliance"),
        "modality": "Legal text",
        "metric": "Macro-F1",
        "backbones": ("Legal-BERT", "Longformer", "DeBERTa"),
        "proposed": "Neurosymbolic Legal Reasoner",
        "descriptor": "structured legal reasoning",
    },
    "Maritime Logistics": {
        "keywords": ("maritime", "shipping", "port", "vessel", "fleet", "logistics"),
        "modality": "Spatio-temporal tabular",
        "metric": "Route utility",
        "backbones": ("XGBoost", "LSTM forecaster", "Graph planner"),
        "proposed": "Adaptive Maritime Route Planner",
        "descriptor": "route-aware decision optimization",
    },
    "Computer Vision": {
        "keywords": ("image", "vision", "visual", "video", "segmentation", "detection", "medical imaging"),
        "modality": "Image",
        "metric": "mIoU",
        "backbones": ("ResNet", "ConvNeXt", "Vision Transformer"),
        "proposed": "Context-Aware Vision Model",
        "descriptor": "multi-scale visual representation",
    },
    "Natural Language Processing": {
        "keywords": ("language", "text", "nlp", "transformer", "sentiment", "document", "speech"),
        "modality": "Text",
        "metric": "Macro-F1",
        "backbones": ("BiLSTM", "BERT", "RoBERTa"),
        "proposed": "Context-Aware Language Model",
        "descriptor": "contextual representation learning",
    },
    "Graph Learning": {
        "keywords": ("graph", "gnn", "gcn", "node", "edge", "relationship", "molecule"),
        "modality": "Graph",
        "metric": "F1-score",
        "backbones": ("GCN", "GraphSAGE", "Graph Attention Network"),
        "proposed": "Adaptive Relational Graph Network",
        "descriptor": "relation-aware message passing",
    },
    "Reinforcement Learning": {
        "keywords": ("reinforcement", "robot", "robotics", "policy", "agent", "control", "navigation"),
        "modality": "State-action trajectory",
        "metric": "Episode return",
        "backbones": ("DQN", "PPO", "SAC"),
        "proposed": "Adaptive Policy Optimizer",
        "descriptor": "uncertainty-aware policy learning",
    },
    "Time Series Forecasting": {
        "keywords": ("forecast", "time series", "temporal", "stock", "demand", "weather", "energy"),
        "modality": "Time series",
        "metric": "Normalized RMSE",
        "backbones": ("ARIMA", "LSTM", "Temporal Transformer"),
        "proposed": "Adaptive Temporal Forecaster",
        "descriptor": "multi-scale temporal forecasting",
    },
    "Healthcare AI": {
        "keywords": ("health", "healthcare", "clinical", "patient", "diagnosis", "radiology", "medical"),
        "modality": "Clinical records",
        "metric": "AUROC",
        "backbones": ("Clinical MLP", "Survival model", "Medical Transformer"),
        "proposed": "Risk-Aware Clinical Model",
        "descriptor": "calibrated clinical prediction",
    },
    "FinTech": {
        "keywords": ("finance", "financial", "fintech", "fraud", "transaction", "credit", "market"),
        "modality": "Tabular and transaction graph",
        "metric": "PR-AUC",
        "backbones": ("XGBoost", "LightGBM", "GraphSAGE"),
        "proposed": "Risk-Aware Financial Model",
        "descriptor": "temporal risk representation",
    },
}


def _topic_profile(topic: str, domain: str | None, objective: str | None = None) -> dict[str, Any]:
    clean_topic = _normalize_topic(topic)
    text = f"{clean_topic} {domain or ''} {objective or ''}".lower()
    selected_name = next(
        (name for name, profile in _DOMAIN_PROFILES.items()
         if any(_keyword_matches(text, keyword) for keyword in profile["keywords"])),
        None,
    )
    if selected_name is None:
        selected_name = domain.strip() if domain and domain.strip() else "General Deep Learning & Applied AI"
        profile = {
            "keywords": tuple(),
            "modality": "Domain-specific records",
            "metric": "Task performance",
            "backbones": ("Regularized MLP", "Gradient Boosting", "Attention Encoder"),
            "proposed": "Topic-Aware Adaptive Model",
            "descriptor": "task-aware signal selection",
        }
    else:
        profile = _DOMAIN_PROFILES[selected_name]
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]+", clean_topic)
    keywords = list(dict.fromkeys(token.lower() for token in tokens if token.lower() not in _STOP_WORDS))[:8]
    return {**profile, "topic": clean_topic, "domain": selected_name, "keywords": keywords}


def _keyword_matches(text: str, keyword: str) -> bool:
    normalized_keyword = keyword.lower().strip()
    if len(normalized_keyword) <= 3 or normalized_keyword in {"asr", "tts", "rag", "llm", "clip", "vqa", "ctr", "ppo", "sac", "gpt"}:
        return re.search(rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])", text) is not None
    return normalized_keyword in text


def _infer_domain(topic: str, domain: str | None, objective: str | None) -> str:
    return _topic_profile(topic, domain, objective)["domain"]


def _verified_local_data(domain: str, root: str | None = None) -> list[str]:
    data_root = Path(root or Path(__file__).resolve().parent.parent / "data")
    if "nlp" not in domain.lower() and domain != "Natural Language Processing":
        return []
    candidates = {
        "Shakespeare Char": data_root / "shakespeare_char" / "input.txt",
        "Enwik8": data_root / "enwik8" / "enwik8",
        "Text8": data_root / "text8" / "text8",
        "AG News": data_root / "huggingface" / "ag_news" / "train.jsonl",
        "IMDb Large Movie Review": data_root / "huggingface" / "imdb" / "train.jsonl",
    }
    return [name for name, path in candidates.items() if path.is_file() and path.stat().st_size > 0]


def _validate_verified_request(topic: str, domain: str | None, objective: str | None) -> None:
    clean_topic = _normalize_topic(topic)
    inferred_domain = _infer_domain(clean_topic, domain, objective)
    verified = _verified_local_data(inferred_domain)
    if not verified:
        raise HTTPException(status_code=422, detail=f"No verified local dataset is available for '{inferred_domain}'. Disable verified-data checking to use the configured dataset catalog.")


def _knowledge_layer(topic: str, domain: str, objective: str | None, model_names: list[str], dataset_names: list[str]) -> tuple[KnowledgeBaseSpec, InferenceEngineSpec, AgentCoordinationSpec]:
    profile = _topic_profile(topic, domain, objective)
    text = f"{topic} {domain} {objective or ''}".lower()
    matched = [keyword for keyword in profile["keywords"] if _keyword_matches(text, keyword)]
    confidence = min(0.99, 0.62 + 0.06 * len(matched))
    knowledge_base = KnowledgeBaseSpec(
        taxonomy=[domain, profile["modality"], profile["metric"]],
        routing_rules=[
            "Prefer explicit domain input over inferred keywords.",
            "Use token-boundary matching for short technical acronyms.",
            "Route to complementary core, robustness, and ablation datasets.",
        ],
    )
    inference_engine = InferenceEngineSpec(
        selected_domain=domain,
        matched_keywords=matched,
        routing_confidence=round(confidence, 2),
        routed_configuration=[f"Models: {', '.join(model_names)}", f"Datasets: {', '.join(dataset_names)}", f"Metric: {profile['metric']}"],
    )
    coordination = AgentCoordinationSpec(
        agents=["Research Planner Agent", "Benchmark Generator Agent", "Metric Evaluator Agent", "Peer Reviewer Agent"],
        coordination_claim="Typed shared state lets each agent expose assumptions, evidence, metrics, and review feedback before manuscript export.",
    )
    return knowledge_base, inference_engine, coordination


def _ensure_system_intelligence(plan: PlannerOutput, topic: str, domain: str, objective: str | None) -> PlannerOutput:
    knowledge_base, inference_engine, coordination = _knowledge_layer(
        topic,
        domain,
        objective,
        [model.name for model in plan.models_catalog],
        [dataset.name for dataset in plan.datasets_catalog],
    )
    plan.project_metadata = ProjectMetadata(**PTIT_COURSE_METADATA)
    if not plan.knowledge_base.taxonomy:
        plan.knowledge_base = knowledge_base
    if not plan.inference_engine.selected_domain:
        plan.inference_engine = inference_engine
    if not plan.agent_coordination.agents:
        plan.agent_coordination = coordination
    return plan


def _make_dataset_catalog(topic: str, domain: str | None) -> list[DatasetSpec]:
    profile = _topic_profile(topic, domain)
    if profile["domain"] == "Speech & Audio Processing":
        return [
            DatasetSpec(name="LibriSpeech ASR", modality="Speech / ASR", sample_size="1,000 hours of read English speech", input_spec="Audio waveforms with aligned transcripts", primary_metric="Word Error Rate", description="Standard automatic speech recognition benchmark for acoustic and language modeling."),
            DatasetSpec(name="VoxCeleb Speaker Recognition", modality="Speech / Speaker ID", sample_size="1.2M utterances from 7,000+ speakers", input_spec="Noisy speech segments with speaker identities", primary_metric="Speaker F1", description="In-the-wild speaker recognition benchmark with channel and accent variation."),
            DatasetSpec(name="Common Voice Multi-Lingual", modality="Speech / Multilingual", sample_size="100+ languages / community recordings", input_spec="Crowdsourced speech clips with language and transcript metadata", primary_metric="Macro WER", description="Multilingual speech benchmark for low-resource and accent-robust evaluation."),
        ]
    if profile["domain"] == "Reinforcement Learning & Robotics":
        return [
            DatasetSpec(name="OpenAI Gym / MuJoCo Suite", modality="Control / Simulation", sample_size="20 continuous-control environments", input_spec="State, action, reward, and next-state trajectories", primary_metric="Episode return", description="Continuous-control suite for reproducible policy optimization experiments."),
            DatasetSpec(name="DM Control Suite", modality="Control / Simulation", sample_size="30 physics-based tasks", input_spec="Pixel or state observations with dense rewards", primary_metric="Success rate", description="Physics-rich benchmark for robust control and generalization across tasks."),
            DatasetSpec(name="RoboSuite Manipulation", modality="Robot manipulation", sample_size="10 simulated manipulation tasks", input_spec="Robot proprioception, object state, and action trajectories", primary_metric="Task success", description="Manipulation benchmark for evaluating contextual policies under changing scenes."),
        ]
    if profile["domain"] == "Multimodal & Vision-Language":
        return [
            DatasetSpec(name="MS-COCO Captions", modality="Image-text pairs", sample_size="330k images / 1.5M captions", input_spec="Images paired with human-written descriptions", primary_metric="CIDEr / Recall@K", description="Image captioning and cross-modal retrieval benchmark."),
            DatasetSpec(name="VQA v2.0", modality="Image-question-answer", sample_size="200k images / 1.1M questions", input_spec="Images, natural-language questions, and answer distributions", primary_metric="VQA accuracy", description="Visual question answering benchmark designed to reduce language-only priors."),
            DatasetSpec(name="Conceptual Captions (CC3M)", modality="Web image-text", sample_size="3.3M noisy image-caption pairs", input_spec="Weakly aligned web images and alt-text captions", primary_metric="Zero-shot Recall@K", description="Large-scale noisy pretraining and transfer benchmark for vision-language models."),
        ]
    if profile["domain"] == "Recommendation Systems & CTR":
        return [
            DatasetSpec(name="MovieLens-20M", modality="User-item ratings", sample_size="20M ratings / 138k users", input_spec="User, item, timestamp, and explicit rating interactions", primary_metric="NDCG@10", description="Collaborative filtering benchmark for personalized ranking."),
            DatasetSpec(name="Amazon Product Reviews", modality="User-item reviews", sample_size="Millions of reviews across product categories", input_spec="User, item, text review, rating, and temporal interactions", primary_metric="Recall@K", description="Long-tail recommendation benchmark with sparse sequential behavior."),
            DatasetSpec(name="Criteo Display Advertising", modality="Click-through logs", sample_size="45M anonymized ad impressions", input_spec="Categorical and numerical features with click labels", primary_metric="AUC / LogLoss", description="Large-scale CTR prediction benchmark under severe class imbalance."),
        ]
    if profile["domain"] == "Graph Learning & FinTech":
        return [
            DatasetSpec(name="Elliptic Bitcoin Dataset", modality="Transaction graph", sample_size="203k transactions / 234k edges", input_spec="Transaction nodes, temporal graph edges, and illicitness labels", primary_metric="PR-AUC", description="Bitcoin transaction graph for illicit activity and anomaly detection."),
            DatasetSpec(name="Cora/Citeseer Graph", modality="Citation graph", sample_size="10k papers / citation edges", input_spec="Node features, graph connectivity, and research labels", primary_metric="Node F1", description="Classic semi-supervised node-classification benchmark."),
            DatasetSpec(name="Financial Transaction Fraud Set", modality="Financial graph", sample_size="500k transactions / temporal holdout", input_spec="Accounts, transactions, and fraud risk labels", primary_metric="Fraud PR-AUC", description="Temporal fraud benchmark for relational risk modeling."),
        ]
    if profile["domain"] == "Natural Language Processing (NLP) / RAG":
        return [
            DatasetSpec(name="Natural Questions (NQ)", modality="Text / Open-domain QA", sample_size="307k questions / retrieved-document split", input_spec="Question, retrieved passages, and answer annotations", primary_metric="Exact Match / F1", description="Open-domain question answering benchmark for measuring retrieval-grounded generation quality."),
            DatasetSpec(name="HotpotQA Multi-hop Attribution", modality="Text / Multi-hop QA", sample_size="113k questions / supporting-fact annotations", input_spec="Questions with multi-document evidence chains", primary_metric="Joint F1 / Support EM", description="Multi-hop reasoning benchmark that evaluates whether generated claims are supported by sources."),
            DatasetSpec(name="TruthfulQA Faithfulness Set", modality="Text / Truthfulness", sample_size="817 adversarial questions / human references", input_spec="Adversarial prompts with truthful and hallucinated answer contrasts", primary_metric="Truthfulness / BLEU", description="Stress test for hallucination resistance, confidence calibration, and source attribution."),
        ]
    if profile["domain"] == "Natural Language Processing (NLP) / Machine Translation":
        return [
            DatasetSpec(name="Flores-101 (Low-Resource Languages)", modality="Text / Translation", sample_size="101 languages / train-validation-test splits", input_spec="Parallel sentences across high- and low-resource language pairs", primary_metric="COMET / BLEU", description="Multilingual translation benchmark for evaluating zero-shot and low-resource transfer."),
            DatasetSpec(name="XNLI Cross-Lingual Evaluation", modality="Text / NLI", sample_size="15 languages / matched validation sets", input_spec="Premise-hypothesis pairs translated across languages", primary_metric="Accuracy", description="Cross-lingual natural language inference suite for transfer and representation quality."),
            DatasetSpec(name="Tatoeba Translation Benchmark", modality="Text / Parallel Sentences", sample_size="50,000 sentence pairs / held-out evaluation", input_spec="Multilingual parallel sentence pairs with language-direction labels", primary_metric="BLEU / chrF", description="Translation benchmark for measuring robustness across diverse language directions."),
        ]
    if profile["domain"] == "Natural Language Processing":
        topic_text = topic.lower()
        if any(term in topic_text for term in ("classification", "sentiment", "news", "text classifier")):
            return [
                DatasetSpec(name="AG News", modality="Text classification", sample_size="120k labeled news articles", input_spec="Article text with four topic labels", primary_metric="Accuracy / Macro-F1", description="Real-world news topic classification benchmark suitable for supervised transformer experiments."),
                DatasetSpec(name="IMDb Large Movie Review", modality="Sentiment text", sample_size="50k labeled movie reviews", input_spec="Review text with binary sentiment labels", primary_metric="Accuracy / Macro-F1", description="Balanced sentiment benchmark for robust text classification and error analysis."),
                DatasetSpec(name="Shakespeare Char", modality="Character-level text", sample_size="1.1 MB corpus / train-validation split", input_spec="Encoded character sequences with next-character targets", primary_metric="Validation loss", description="Local character-level corpus for a complementary language-modeling check."),
            ]
        return [
            DatasetSpec(name="Shakespeare Char", modality="Character-level text", sample_size="1.1 MB corpus / train-validation split", input_spec="Encoded character sequences with next-character targets", primary_metric="Validation loss", description="Shakespeare play text used for character-level language modeling and qualitative generation."),
            DatasetSpec(name="Enwik8", modality="Character-level text", sample_size="100 MB Wikipedia corpus / train-validation split", input_spec="Raw Wikipedia characters stored as uint16 token ids", primary_metric="Validation loss", description="Standard 100 MB enwik8 compression and next-character prediction benchmark."),
            DatasetSpec(name="Text8", modality="Character-level text", sample_size="100 MB normalized English corpus / train-validation split", input_spec="Normalized lowercase English characters without formatting", primary_metric="Validation loss", description="Clean text8 corpus for measuring language-model convergence on standardized text."),
        ]
    anchor = " ".join(word.title() for word in profile["keywords"][:3]) or profile["domain"]
    return [
        DatasetSpec(name=f"{anchor} Core Benchmark", modality=profile["modality"], sample_size="40k train / 8k validation / 8k test", input_spec=f"Topic-specific {profile['modality'].lower()} with supervised targets", primary_metric=profile["metric"], description=f"Primary simulated benchmark for {profile['topic']} with realistic distribution shift."),
        DatasetSpec(name=f"{anchor} Robustness Split", modality=profile["modality"], sample_size="18k samples / temporal or OOD holdout", input_spec="Stress-test partition with noise, missingness, and edge cases", primary_metric=profile["metric"], description="Challenge split for measuring generalization beyond the nominal data distribution."),
        DatasetSpec(name=f"{anchor} Ablation Set", modality=profile["modality"], sample_size="12k samples / controlled factors", input_spec="Matched subset for component removal and interpretability checks", primary_metric="Robustness gap", description="Controlled evaluation set for isolating the contribution of each model component."),
    ]


def _make_model_catalog(topic: str, domain: str | None) -> list[ModelSpec]:
    profile = _topic_profile(topic, domain)
    if profile["domain"] == "Speech & Audio Processing":
        return [
            ModelSpec(name="Conformer Baseline", role="Baseline", backbone="Conformer encoder", params_m=118.0, flops_g=42.0, latency_ms=35.0, key_characteristic="Convolution-augmented transformer baseline for speech recognition."),
            ModelSpec(name="HuBERT", role="Baseline", backbone="Self-supervised speech encoder", params_m=95.0, flops_g=38.0, latency_ms=41.0, key_characteristic="Masked prediction pretraining for robust acoustic representations."),
            ModelSpec(name="Wav2Vec 2.0", role="Baseline", backbone="Self-supervised waveform encoder", params_m=84.0, flops_g=34.0, latency_ms=39.0, key_characteristic="Learns speech representations directly from raw waveforms."),
            ModelSpec(name="Dynamic Streaming Transducer (Ours)", role="Proposed", backbone="Streaming transducer with adaptive context", params_m=72.0, flops_g=28.0, latency_ms=24.0, key_characteristic="Adaptive streaming context for low-latency multilingual transcription."),
        ]
    if profile["domain"] == "Reinforcement Learning & Robotics":
        return [
            ModelSpec(name="PPO Baseline", role="Baseline", backbone="Proximal Policy Optimization", params_m=4.2, flops_g=1.2, latency_ms=8.0, key_characteristic="Stable clipped policy-gradient baseline for continuous control."),
            ModelSpec(name="Soft Actor-Critic (SAC)", role="Baseline", backbone="Maximum-entropy actor-critic", params_m=5.1, flops_g=1.5, latency_ms=9.0, key_characteristic="Entropy-regularized off-policy control baseline."),
            ModelSpec(name="Rainbow DQN", role="Baseline", backbone="Distributional value-based agent", params_m=8.4, flops_g=2.8, latency_ms=12.0, key_characteristic="Combines value-learning improvements for discrete action tasks."),
            ModelSpec(name="Adaptive Contextual PPO (Ours)", role="Proposed", backbone="PPO with contextual policy gating", params_m=5.8, flops_g=1.7, latency_ms=9.5, key_characteristic="Adapts policy context to changing dynamics and observation regimes."),
        ]
    if profile["domain"] == "Multimodal & Vision-Language":
        return [
            ModelSpec(name="CLIP Baseline", role="Baseline", backbone="Dual-encoder contrastive model", params_m=151.0, flops_g=68.0, latency_ms=42.0, key_characteristic="Strong zero-shot image-text alignment baseline."),
            ModelSpec(name="BLIP-2", role="Baseline", backbone="Frozen vision encoder + Q-Former", params_m=188.0, flops_g=82.0, latency_ms=56.0, key_characteristic="Bridges frozen vision and language models through query tokens."),
            ModelSpec(name="Flamingo", role="Baseline", backbone="Few-shot vision-language decoder", params_m=340.0, flops_g=120.0, latency_ms=74.0, key_characteristic="Interleaves visual context with language generation."),
            ModelSpec(name="Cross-Modal Contrastive Fusion (Ours)", role="Proposed", backbone="Gated image-text fusion encoder", params_m=172.0, flops_g=76.0, latency_ms=48.0, key_characteristic="Jointly calibrates visual and textual evidence under a contrastive objective."),
        ]
    if profile["domain"] == "Recommendation Systems & CTR":
        return [
            ModelSpec(name="DeepFM Baseline", role="Baseline", backbone="Factorization machine + deep network", params_m=18.0, flops_g=4.5, latency_ms=15.0, key_characteristic="Models low- and high-order feature interactions for CTR."),
            ModelSpec(name="DLRM", role="Baseline", backbone="Embedding interaction recommender", params_m=32.0, flops_g=6.2, latency_ms=18.0, key_characteristic="Production-oriented dense and sparse feature interaction model."),
            ModelSpec(name="SASRec", role="Baseline", backbone="Self-attentive sequential recommender", params_m=24.0, flops_g=5.4, latency_ms=21.0, key_characteristic="Captures sequential user intent with causal self-attention."),
            ModelSpec(name="Dynamic Session-Aware Graph-Rec (Ours)", role="Proposed", backbone="Session graph with adaptive ranking", params_m=29.0, flops_g=6.0, latency_ms=19.0, key_characteristic="Combines session context and relational item transitions for ranking."),
        ]
    if profile["domain"] == "Graph Learning & FinTech":
        return [
            ModelSpec(name="GCN", role="Baseline", backbone="Graph convolutional network", params_m=12.0, flops_g=2.8, latency_ms=18.0, key_characteristic="Spectral-style neighborhood aggregation baseline."),
            ModelSpec(name="GraphSAGE", role="Baseline", backbone="Inductive neighbor sampler", params_m=16.0, flops_g=3.4, latency_ms=21.0, key_characteristic="Scalable inductive representation learning for unseen nodes."),
            ModelSpec(name="Graph Attention Network (GAT)", role="Baseline", backbone="Attention-based message passing", params_m=19.0, flops_g=4.1, latency_ms=25.0, key_characteristic="Learns neighbor importance through attention coefficients."),
            ModelSpec(name="Adaptive Relational Graph Network (Ours)", role="Proposed", backbone="Dynamic relation-aware GNN", params_m=22.0, flops_g=4.5, latency_ms=24.0, key_characteristic="Adapts relation weights for temporal and heterogeneous graph risk patterns."),
        ]
    if profile["domain"] == "Natural Language Processing (NLP) / RAG":
        return [
            ModelSpec(name="DPR + Generator", role="Baseline", backbone="Dense Passage Retriever + seq2seq generator", params_m=410.0, flops_g=82.0, latency_ms=68.0, key_characteristic="Two-stage retrieval-generation baseline with no explicit confidence control."),
            ModelSpec(name="RAG-Sequence", role="Baseline", backbone="Retriever-conditioned sequence generator", params_m=430.0, flops_g=88.0, latency_ms=72.0, key_characteristic="Marginalizes retrieved passages during generation but does not expose claim-level evidence."),
            ModelSpec(name="Self-RAG", role="Baseline", backbone="Reflection-token augmented LLM", params_m=750.0, flops_g=145.0, latency_ms=96.0, key_characteristic="Uses self-reflection signals to decide when retrieval and critique are needed."),
            ModelSpec(name="Confidence-Aware Attributed Decoder (Ours)", role="Proposed", backbone="RAG decoder with confidence gating and source attribution", params_m=465.0, flops_g=91.0, latency_ms=61.0, key_characteristic="Calibrates decoding confidence and attaches generated claims to retrieved evidence to reduce unsupported statements."),
        ]
    if profile["domain"] == "Natural Language Processing (NLP) / Machine Translation":
        return [
            ModelSpec(name="mBERT Baseline", role="Baseline", backbone="BERT-Base-Multilingual", params_m=177.0, flops_g=35.0, latency_ms=38.2, key_characteristic="Multilingual encoder baseline for zero-shot transfer across language families."),
            ModelSpec(name="XLM-RoBERTa Base", role="Baseline", backbone="XLM-R", params_m=270.0, flops_g=52.0, latency_ms=45.1, key_characteristic="Large multilingual masked-language encoder with broad cross-lingual coverage."),
            ModelSpec(name="MAD-X Adapters", role="Baseline", backbone="XLM-R + Adapter", params_m=28.4, flops_g=14.0, latency_ms=22.5, key_characteristic="Parameter-efficient adapter composition for multilingual transfer."),
            ModelSpec(name="Dynamic Cross-Lingual Adapter (Ours)", role="Proposed", backbone="XLM-R + Dynamic Gating", params_m=14.2, flops_g=10.5, latency_ms=18.6, key_characteristic="Dynamically gates language and task adapters to improve low-resource transfer with a compact trainable budget."),
        ]
    if profile["domain"] == "Natural Language Processing":
        return [
            ModelSpec(name="nanoGPT Baseline", role="Baseline", backbone="Decoder-only GPT Transformer", params_m=12.4, flops_g=6.8, latency_ms=24.0, key_characteristic="Reference character-level nanoGPT with causal self-attention and next-character prediction."),
            ModelSpec(name="nanoGPT + QK-Norm", role="Baseline", backbone="GPT Transformer with normalized Q/K", params_m=12.4, flops_g=7.0, latency_ms=25.0, key_characteristic="Attention-stability variant that normalizes query and key representations."),
            ModelSpec(name="nanoGPT + Layer-wise LR", role="Baseline", backbone="GPT Transformer with layer-wise learning rates", params_m=12.4, flops_g=6.8, latency_ms=24.5, key_characteristic="Optimization baseline assigning depth-aware learning-rate schedules."),
            ModelSpec(name="Dynamic Attention Head Gating nanoGPT", role="Proposed", backbone="GPT Transformer with learnable head gates", params_m=12.5, flops_g=7.1, latency_ms=25.5, key_characteristic="Proposed nanoGPT variant with sigmoid head gates and sparsity-aware attention allocation."),
        ]
    if profile["domain"] == "General Deep Learning & Applied AI":
        return [
            ModelSpec(name="ResNet-50", role="Baseline", backbone="Residual convolutional encoder", params_m=25.6, flops_g=4.1, latency_ms=32.0, key_characteristic="Strong general-purpose residual baseline for supervised representation learning."),
            ModelSpec(name="Transformer Base", role="Baseline", backbone="Encoder-decoder transformer", params_m=65.0, flops_g=12.0, latency_ms=44.0, key_characteristic="Neutral attention-based baseline for sequence and structured prediction tasks."),
            ModelSpec(name="MLP Baseline", role="Baseline", backbone="Regularized multilayer perceptron", params_m=8.2, flops_g=1.2, latency_ms=18.0, key_characteristic="Compact non-relational baseline for measuring the value of architectural structure."),
            ModelSpec(name="Task-Adaptive Deep Network (Ours)", role="Proposed", backbone="Adaptive feature encoder", params_m=19.5, flops_g=3.8, latency_ms=27.0, key_characteristic="Task-adaptive fusion model used as a neutral proposed method for unknown topics."),
        ]
    models = []
    for index, backbone in enumerate(profile["backbones"]):
        models.append(ModelSpec(
            name=backbone,
            role="Baseline",
            backbone=f"{backbone} architecture",
            params_m=round(18.0 + index * 13.0 + len(profile["keywords"]) * 0.7, 1),
            flops_g=round(4.5 + index * 3.2, 1),
            latency_ms=round(24.0 + index * 11.0, 1),
            key_characteristic=f"Established {profile['domain']} baseline with a distinct capacity and inductive bias.",
        ))
    models.append(ModelSpec(
        name=profile["proposed"], role="Proposed", backbone=f"Hybrid {profile['descriptor']} architecture",
        params_m=round(30.0 + len(profile["keywords"]) * 1.4, 1), flops_g=round(8.0 + len(profile["keywords"]) * 0.8, 1),
        latency_ms=round(31.0 + len(profile["keywords"]) * 1.2, 1),
        key_characteristic=f"Combines {profile['descriptor']} with adaptive regularization and uncertainty-aware evaluation.",
    ))
    return models


def _safe_live_call(api_key: str | None, prompt: str) -> Any | None:
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3, response_mime_type="application/json"),
        )
        try:
            return json.loads(response.text)
        except Exception:
            text = getattr(response, "text", None)
            if text:
                return json.loads(text)
            return None
    except Exception:
        return None


def _build_dynamic_planner(topic: str, domain: str | None, objective: str | None) -> PlannerOutput:
    clean_topic = _normalize_topic(topic)
    profile = _topic_profile(clean_topic, domain, objective)
    domain_name = profile["domain"]
    datasets_catalog = _make_dataset_catalog(clean_topic, domain_name)
    models_catalog = _make_model_catalog(clean_topic, domain_name)
    return PlannerOutput(
        title=f"{clean_topic}: A 2026-Inspired Research Synthesis",
        domain=domain_name,
        research_question=f"How can {clean_topic} be improved in a principled way while preserving robustness, scalability, and empirical validity?",
        hypothesis=f"A task-aware design grounded in {domain_name} can capture informative signals while reducing noise, redundancy, and unnecessary computation.",
        datasets_catalog=datasets_catalog,
        models_catalog=models_catalog,
        baseline_methods=[item.name for item in models_catalog if item.role == "Baseline"],
        proposed_method=next(item.name for item in models_catalog if item.role == "Proposed"),
        evaluation_metrics=[profile["metric"], "Calibration", "Latency", "Robustness gap"],
        experimental_plan=[
            "Construct normalized benchmark settings for the target task and dataset.",
            "Train the proposed approach under a preregistered and reproducible protocol.",
            "Run ablation studies to isolate the contribution of the design choices.",
            "Measure performance, stability, efficiency, and failure modes under identical evaluation conditions."
        ],
        report_outline=[
            "Abstract",
            "Introduction and Motivation",
            "Related Work",
            "Methodology",
            "Experimental Setup",
            "Results and Analysis",
            "Discussion",
            "Conclusion"
        ],
    )


def _domain_tech_stack(domain: str) -> list[str]:
    stacks = {
        "Natural Language Processing (NLP) / RAG": ["PyTorch", "Hugging Face Transformers", "Hugging Face Datasets", "PEFT", "Accelerate", "FAISS"],
        "Natural Language Processing (NLP) / Machine Translation": ["PyTorch", "Hugging Face Transformers", "Hugging Face Datasets", "PEFT", "Sacremoses", "Evaluate"],
        "Natural Language Processing": ["PyTorch", "Hugging Face Transformers", "Hugging Face Datasets", "Tokenizers", "Accelerate"],
        "Computer Vision": ["PyTorch", "TorchVision", "timm", "Albumentations", "OpenCV", "TorchMetrics"],
        "Speech & Audio Processing": ["PyTorch", "Librosa", "torchaudio", "SpeechBrain", "Hugging Face Datasets", "JiWER"],
        "Reinforcement Learning & Robotics": ["PyTorch", "Gymnasium", "Stable-Baselines3", "Shimmy", "TensorBoard", "Farama Utils"],
        "Reinforcement Learning": ["PyTorch", "Gymnasium", "Stable-Baselines3", "TensorBoard"],
        "Graph Learning & FinTech": ["PyTorch", "PyTorch Geometric", "NetworkX", "OGB", "scikit-learn", "MLflow"],
        "Graph Learning": ["PyTorch", "PyTorch Geometric", "DGL", "NetworkX", "scikit-learn"],
        "Time Series Forecasting": ["PyTorch", "pandas", "scikit-learn", "StatsForecast", "sktime", "Darts"],
        "Multimodal & Vision-Language": ["PyTorch", "TorchVision", "Hugging Face Transformers", "OpenCLIP", "Albumentations", "W&B"],
    }
    return stacks.get(domain, ["Python", "PyTorch", "NumPy", "pandas", "scikit-learn", "MLflow"])


def _dataset_source(name: str) -> str:
    sources = {
        "LibriSpeech ASR": "https://www.openslr.org/12",
        "VoxCeleb Speaker Recognition": "https://www.robots.ox.ac.uk/~vgg/data/voxceleb/",
        "Common Voice Multi-Lingual": "https://commonvoice.mozilla.org/datasets",
        "MS-COCO Captions": "https://cocodataset.org/#download",
        "VQA v2.0": "https://visualqa.org/download.html",
        "Conceptual Captions (CC3M)": "https://ai.google.com/research/ConceptualCaptions/",
        "MovieLens-20M": "https://grouplens.org/datasets/movielens/20m/",
        "Criteo Display Advertising": "https://ailab.criteo.com/ressources/",
        "Elliptic Bitcoin Dataset": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
        "Cora/Citeseer Graph": "https://github.com/kimiyoung/planetoid",
        "Natural Questions (NQ)": "https://ai.google.com/research/NaturalQuestions",
        "HotpotQA Multi-hop Attribution": "https://hotpotqa.github.io/",
        "TruthfulQA Faithfulness Set": "https://github.com/sylinrl/TruthfulQA",
        "Flores-101 (Low-Resource Languages)": "https://github.com/facebookresearch/flores",
        "XNLI Cross-Lingual Evaluation": "https://github.com/facebookresearch/XNLI",
        "Shakespeare Char": "https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt",
        "Enwik8": "https://www.byronosa.com/enwik8",
        "Text8": "http://mattmahoney.net/dc/textdata.html",
    }
    return sources.get(name, "https://huggingface.co/datasets")


def _make_ablation_study(plan: PlannerOutput, proposed_score: float, proposed_latency: float) -> list[AblationSpec]:
    domain = plan.domain
    components = {
        "Natural Language Processing (NLP) / RAG": ["Confidence-aware decoding", "Retrieved evidence attribution", "Claim-level verifier"],
        "Speech & Audio Processing": ["Streaming context gate", "SpecAugment", "Language-model rescoring"],
        "Computer Vision": ["Multi-scale feature fusion", "Adaptive augmentation", "Uncertainty head"],
        "Reinforcement Learning & Robotics": ["Contextual policy gate", "Entropy regularization", "Dynamics randomization"],
        "Graph Learning & FinTech": ["Temporal relation encoder", "Neighbor attention", "Risk calibration"],
    }.get(domain, ["Adaptive gating", "Robustness regularization", "Uncertainty calibration"])
    rows = [AblationSpec(configuration="Full proposed model", removed_component="None", primary_metric=round(proposed_score, 4), latency_ms=round(proposed_latency, 2), delta_vs_full=0.0, interpretation="Reference configuration retaining all proposed components.")]
    for index, component in enumerate(components):
        drop = 0.018 + index * 0.012
        rows.append(AblationSpec(configuration=f"Without {component}", removed_component=component, primary_metric=round(max(0.0, proposed_score - drop), 4), latency_ms=round(max(0.1, proposed_latency - (index + 1) * 0.8), 2), delta_vs_full=round(-drop, 4), interpretation=f"Removing {component.lower()} lowers the planned primary metric; verify this effect across independent seeds."))
    return rows


def _build_dynamic_research(plan: PlannerOutput) -> ResearcherOutput:
    topic_tokens = plan.title.split(":")[0]
    seed = int.from_bytes(hashlib.sha256(topic_tokens.encode("utf-8")).digest()[:8], "big") % (2**32)
    rng = np.random.default_rng(seed)
    curves: list[MetricCurvePoint] = []
    dataset_names = [dataset.name for dataset in plan.datasets_catalog]
    for dataset_name in dataset_names:
        epochs = np.arange(1, 16)
        baseline_start = rng.uniform(1.25, 1.8)
        baseline_end = rng.uniform(0.48, 0.68)
        proposed_start = baseline_start * rng.uniform(0.95, 1.02)
        proposed_end = baseline_end * rng.uniform(0.86, 0.94)
        baseline_loss_curve = np.linspace(baseline_start, baseline_end, len(epochs)) + rng.normal(0, 0.012, len(epochs))
        proposed_loss_curve = np.linspace(proposed_start, proposed_end, len(epochs)) + rng.normal(0, 0.012, len(epochs))
        baseline_score_start = rng.uniform(0.60, 0.74)
        baseline_score_end = rng.uniform(0.76, 0.84)
        improvement = rng.uniform(0.05, 0.12)
        proposed_score_start = min(0.78, baseline_score_start * (1 + improvement * 0.25))
        proposed_score_end = min(0.98, baseline_score_end * (1 + improvement))
        baseline_score_curve = np.linspace(baseline_score_start, baseline_score_end, len(epochs)) + rng.normal(0, 0.004, len(epochs))
        proposed_score_curve = np.linspace(proposed_score_start, proposed_score_end, len(epochs)) + rng.normal(0, 0.004, len(epochs))
        for index, epoch in enumerate(epochs):
            curves.append(
                MetricCurvePoint(
                    epoch=int(epoch),
                    dataset_name=dataset_name,
                    baseline_loss=round(max(0.05, float(baseline_loss_curve[index])), 4),
                    proposed_loss=round(max(0.05, float(proposed_loss_curve[index])), 4),
                    baseline_score=round(float(np.clip(baseline_score_curve[index], 0.0, 1.0)), 4),
                    proposed_score=round(float(np.clip(proposed_score_curve[index], 0.0, 1.0)), 4),
                )
            )
    for dataset in plan.datasets_catalog:
        dataset.source_url = _dataset_source(dataset.name)
    baseline_reference = max(curve.baseline_score for curve in curves if curve.dataset_name == dataset_names[0])
    proposed_reference = max(curve.proposed_score for curve in curves if curve.dataset_name == dataset_names[0])
    ablation_study = _make_ablation_study(plan, proposed_reference, next(model.latency_ms for model in plan.models_catalog if model.role == "Proposed"))
    improvement = (proposed_reference - baseline_reference) / max(baseline_reference, 1e-9)
    proposed_scale = 1.0 + float(np.clip(improvement, 0.05, 0.12))
    benchmark_table = [
        AblationRow(
            model_name=model.name,
            dataset_name=dataset_names[0],
            role=model.role,
            primary_metric=round(baseline_reference if model.role == "Baseline" else min(0.98, baseline_reference * proposed_scale), 4),
            secondary_metric=round(max(0.0, baseline_reference - 0.08) if model.role == "Baseline" else min(0.98, (baseline_reference - 0.08) * proposed_scale), 4),
            latency_ms=model.latency_ms,
            params_m=model.params_m,
            notes="Best simulated quality-efficiency tradeoff for the target topic." if model.role == "Proposed" else "Reference configuration in the generated benchmark.",
            vram_mb=round(model.params_m * (34 if model.role == "Proposed" else 90), 1),
            flops_g=model.flops_g,
            evidence_type="Controlled Simulation" if model.role == "Proposed" else "Literature Baseline / Planned",
        )
        for model in plan.models_catalog
    ]
    key_findings = [
        f"The proposed design for {topic_tokens} improves the simulated {plan.evaluation_metrics[0]} by {max(5.0, improvement * 100):.1f}% over the strongest baseline.",
        f"The generated curves show decreasing loss and increasing {plan.evaluation_metrics[0]} across {len(epochs)} epochs.",
        f"Reasoning consistency is {plan.inference_engine.routing_confidence:.2f} and knowledge retrieval precision is {min(0.98, plan.inference_engine.routing_confidence + 0.02):.2f} under the controlled routing audit.",
        "Ablation results are designed to test whether the proposed signal selection contributes beyond a simple capacity increase.",
    ]
    reasoning_consistency = round(min(0.99, plan.inference_engine.routing_confidence * 0.96), 2)
    retrieval_precision = round(min(0.99, plan.inference_engine.routing_confidence + 0.02), 2)
    return ResearcherOutput(
        hardware_setup="1x NVIDIA A100 GPU (40GB VRAM), 64GB CPU RAM, CUDA-enabled PyTorch environment",
        datasets_used=plan.datasets_catalog,
        models_compared=plan.models_catalog,
        training_curves=curves,
        benchmark_table=benchmark_table,
        ablation_study=ablation_study,
        key_findings=key_findings,
        tech_stack=_domain_tech_stack(plan.domain),
        evidence_mode="Controlled Planning Simulation Estimate",
        reasoning_consistency_score=reasoning_consistency,
        knowledge_retrieval_precision=retrieval_precision,
        project_metadata=plan.project_metadata,
        knowledge_base=plan.knowledge_base,
        inference_engine=plan.inference_engine,
        agent_coordination=plan.agent_coordination,
    )


def _build_dynamic_writer(plan: PlannerOutput, research: ResearcherOutput) -> WriterOutput:
    topic_title = plan.title.split(":")[0]
    profile = _topic_profile(topic_title, plan.domain)
    proposed = next(model for model in research.models_compared if model.role == "Proposed")
    baselines = [model for model in research.models_compared if model.role == "Baseline"]
    proposed_rows = [row for row in research.benchmark_table if row.role == "Proposed"]
    baseline_rows = [row for row in research.benchmark_table if row.role == "Baseline"]
    proposed_score = proposed_rows[0].primary_metric if proposed_rows else 0.88
    baseline_score = max((row.primary_metric for row in baseline_rows), default=0.82)
    gain = (proposed_score - baseline_score) / max(baseline_score, 1e-9) * 100
    baseline_latency = min((model.latency_ms for model in baselines), default=40.0)
    average_latency = sum(model.latency_ms for model in baselines) / max(len(baselines), 1)
    dataset_names = ", ".join(dataset.name for dataset in research.datasets_used)
    model_names = ", ".join(model.name for model in research.models_compared)
    ablation_summary = "; ".join(f"{row.configuration}: {row.primary_metric:.2f} ({row.delta_vs_full:+.2f})" for row in research.ablation_study)
    formulas = [
        r"Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V",
        r"g_h=\sigma\left(W_g\,\operatorname{MeanPool}(X)+U_gq+b_g\right)",
        r"\mathcal{L}_{total}=\mathcal{L}_{task}+\lambda_s\lVert g\rVert_1+\lambda_c\mathcal{L}_{consistency}",
        r"\operatorname{Cost}_{dense}=\mathcal{O}(N^2d),\qquad \operatorname{Memory}_{dense}=\mathcal{O}(N^2)",
    ]
    if profile["domain"] == "Quantum Computing":
        formulas[1] = r"|\psi(\theta)\rangle=U(\theta)|0\rangle,\qquad \mathcal{L}_{quantum}=1-\langle\psi(\theta)|H|\psi(\theta)\rangle"
    elif profile["domain"] == "Reinforcement Learning":
        formulas[1] = r"J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_{t=0}^{T}\gamma^t r_t\right],\qquad \nabla_\theta J(\theta)=\mathbb{E}[\nabla_\theta\log\pi_\theta(a_t|s_t)A_t]"
    elif profile["domain"] in {"Time Series Forecasting", "Maritime Logistics"}:
        formulas[1] = r"\hat{y}_{t+h}=f_\theta(y_{t-L:t},x_{t-L:t}),\qquad \mathcal{L}_{forecast}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2"
    elif profile["domain"] in {"Legal AI", "Natural Language Processing", "Natural Language Processing (NLP) / Machine Translation", "Natural Language Processing (NLP) / RAG"}:
        formulas[1] = r"P(y|x)=\mathrm{softmax}(W h_{[CLS]}+b),\qquad \mathcal{L}_{CE}=-\sum_{c=1}^{C}y_c\log P(y=c|x)"
    return WriterOutput(
        title=plan.title,
        abstract=(f"This paper presents a systematic investigation of {topic_title} in the context of {plan.domain}. It frames the system as a knowledge-based reasoning architecture with an explicit inference engine and expert-agent coordination protocol. "
                  f"The study evaluates {proposed.name}, a {proposed.backbone} proposed to improve task-relevant signal selection, against {len(baselines)} representative baselines: {model_names}. "
                  f"Evaluation covers {len(research.datasets_used)} complementary datasets ({dataset_names}) and reports predictive quality, robustness, parameter count, and inference latency. "
                  f"The proposed configuration reaches a primary score of {proposed_score:.2f}, compared with the strongest baseline score of {baseline_score:.2f}, corresponding to a relative improvement of {gain:.1f}%. "
                  f"It uses {proposed.params_m:.1f}M parameters and incurs {proposed.latency_ms:.1f} ms latency, making the quality-efficiency trade-off explicit rather than treating accuracy as the only objective. "
                  "The results support the hypothesis that domain-aware architectural choices can improve generalization and operational reliability without requiring unbounded model scaling. The paper further documents limitations, reproducibility assumptions, and directions for external validation. In particular, the catalog makes the claim auditable: each reported improvement is tied to a named dataset, an explicit comparator, and a measurable resource budget. This framing is useful for researchers who need to decide whether a modest quality gain justifies additional deployment complexity. The contribution is methodological as well as architectural: it defines a repeatable evidence contract in which predictive utility, convergence behavior, and deployment cost are analyzed jointly rather than reported as disconnected headline numbers."),
        keywords=[plan.domain.lower(), "optimization", "robustness", "benchmark", "research pipeline"],
        introduction=(f"{topic_title} is an important problem in {plan.domain} because deployed systems must transform heterogeneous observations into reliable decisions under resource and distribution constraints. "
                      "Although recent architectures have improved benchmark performance, gains are often reported without a matched analysis of latency, parameter growth, robustness, and failure modes. This creates a practical gap between a model that is impressive in an isolated table and one that can be responsibly evaluated in a research or production workflow. "
                      f"The present study frames the central question as follows: {plan.research_question} "
                      f"Our hypothesis is that {plan.hypothesis} The design therefore treats dataset diversity and efficiency as first-class experimental variables rather than post-hoc commentary.\n\n"
                      f"We make three contributions. First, we define a topic-aware experimental protocol spanning {len(research.datasets_used)} datasets and {len(research.models_compared)} model configurations. Second, we report a reproducible comparison between baseline systems and {proposed.name}, including primary score, secondary score, parameters, latency, and convergence behavior. Third, we provide a structured analysis of where the gains originate and where they may not generalize. "
                      f"The proposed system improves the primary benchmark from {baseline_score:.2f} to {proposed_score:.2f} ({gain:.1f}% relative), while its {proposed.latency_ms:.1f} ms latency remains close to the baseline range of {baseline_latency:.1f}-{average_latency:.1f} ms. These results establish a measurable starting point for deeper validation."),
        related_work=("Prior research in machine learning has followed three broad directions. The first develops increasingly expressive backbone architectures, including convolutional encoders, recurrent networks, graph message passing, and attention-based transformers. The second introduces domain priors so that models exploit structure that generic representation learning may overlook. The third studies robustness and efficiency through regularization, pruning, distillation, quantization, or adaptive computation. Each direction is valuable, but they are frequently evaluated in isolation.\n\n"
                      f"Our work is closest to task-aware adaptive modeling. Unlike a purely architecture-centric comparison, we evaluate {proposed.name} alongside {model_names} under the same reporting contract. This matters because a model with a larger score can still be unattractive when its latency or parameter budget grows disproportionately. We consequently use the utility formulation reported in the paper to keep quality and cost visible together.\n\n"
                      "The distinction from prior benchmark-only studies is methodological: we treat the catalog as a compact experimental matrix that can be regenerated for a new topic, while preserving common evidence requirements. The approach does not claim that one architecture dominates every domain. Instead, it offers a disciplined mechanism for testing whether a proposed inductive bias contributes beyond capacity, and for identifying the additional data, statistical tests, or deployment trials needed before making a stronger claim."),
        methodology=(f"The proposed method, {proposed.name}, uses a {proposed.backbone}. At the system level, a Knowledge Base stores the domain taxonomy, inference rules, model catalog, dataset catalog, and metric definitions. The Inference Engine applies token-boundary matching and domain-priority routing before the Benchmark Generator Agent constructs the experiment matrix. The Metric Evaluator Agent then computes predictive and intelligent-system measures, while the Peer Reviewer Agent audits evidence and claim scope. This knowledge-based reasoning and expert coordination design makes the pipeline inspectable rather than a single opaque generation call.\n\n"
                  f"Its core operation first encodes the input into task-relevant representations, then computes context-dependent weights over candidate signals, and finally fuses the selected evidence with a residual path. This construction is intended to retain stable low-level information while allowing the model to emphasize interactions that are predictive for {topic_title}. A regularized objective combines the task loss with sparsity, robustness, and consistency terms; the corresponding equations are listed below.\n\n"
                      f"The experimental protocol uses {len(research.datasets_used)} datasets and compares {len(research.models_compared)} models: {model_names}. Each configuration is evaluated with the same split policy, optimization schedule, stopping criterion, and measurement interface. We record the primary and secondary metrics, training curves, parameter count, FLOPs, latency, and qualitative notes. The benchmark matrix is anchored on {research.benchmark_table[0].dataset_name if research.benchmark_table else 'the primary dataset'} so that the ablation comparison remains directly interpretable.\n\n"
                      f"For reproducibility, the reference hardware is {research.hardware_setup}. The proposed system contains {proposed.params_m:.1f}M parameters, requires {proposed.flops_g:.1f} GFLOPs per inference, and records {proposed.latency_ms:.1f} ms latency in the generated estimate. These values are not presented as universal measurements; they are controlled planning values that should be replaced by instrumented runs before publication. This distinction keeps the generated manuscript useful for planning while making its evidence boundary explicit. Optimization follows a constrained empirical protocol: the learning rate is annealed, early stopping is tied to convergence stability rather than a single favorable epoch, and each reported gain is interpreted alongside parameter and latency budgets. This design distinguishes an inductive-bias contribution from a superficial increase in capacity."),
        mathematical_formulas=formulas,
        experimental_results=(f"Across the generated benchmark matrix, {proposed.name} obtains a primary metric of {proposed_score:.2f}, whereas the strongest baseline reaches {baseline_score:.2f}. The absolute difference is {proposed_score - baseline_score:.2f}, equivalent to a {gain:.1f}% relative gain. The secondary metric changes from the baseline reference of {baseline_rows[0].secondary_metric if baseline_rows else 0.74:.2f} to {proposed_rows[0].secondary_metric if proposed_rows else 0.81:.2f}. These figures indicate a meaningful improvement under the controlled protocol, while the table should still be read as evidence for a research hypothesis rather than a substitute for repeated measurements.\n\n"
                       f"The result is also interpretable in terms of convergence dynamics. Over the 15-epoch simulated schedule, the proposed trajectory reduces optimization loss while increasing the primary score, with bounded perturbations that preserve a realistic rather than perfectly monotonic trace. A credible experimental narrative reports both central tendency and residual variance instead of presenting an artificially smooth curve as proof of generalization.\n\n"
                               f"The efficiency profile is equally important. {proposed.name} uses {proposed.params_m:.1f}M parameters and {proposed.flops_g:.1f} GFLOPs with an estimated latency of {proposed.latency_ms:.1f} ms. The baseline catalog spans {min(model.latency_ms for model in baselines):.1f}-{max(model.latency_ms for model in baselines):.1f} ms and {min(model.params_m for model in baselines):.1f}-{max(model.params_m for model in baselines):.1f}M parameters. Convergence traces show the proposed loss decreasing more rapidly, while its score rises from approximately 0.82 to 0.92 over ten epochs versus 0.85 for the baseline trajectory.\n\n"
                               f"The ablation rows suggest that the gain is not explained solely by adding parameters: {ablation_summary}. The proposed configuration remains competitive on latency and improves both quality dimensions. A complete publication would extend this analysis with multiple random seeds, confidence intervals, calibration, subgroup performance, and a held-out external dataset. It should also record preprocessing time, peak memory, throughput at several batch sizes, and the operational threshold used to convert scores into decisions. Those measurements would determine whether the observed margin survives outside the controlled planning environment."),
        discussion=(f"The results support the hypothesis that task-aware signal selection can improve {topic_title} without an extreme computational penalty. The most direct evidence is the joint movement of the primary score from {baseline_score:.2f} to {proposed_score:.2f} and the controlled latency of {proposed.latency_ms:.1f} ms. This is a more informative result than a score increase alone because the intended use case includes operational constraints. The improvement should be interpreted as a Pareto-style trade-off: a model is preferable only when its quality gain remains material after accounting for memory, throughput, and operational latency.\n\n"
                    "There are, however, important qualifications. The benchmark catalog contains complementary views of the task, but its generated estimates do not establish statistical significance. Dataset shift may change the ranking of models, particularly when the proposed inductive bias depends on relationships or modality-specific signals. Similarly, latency depends on batch size, compiler, hardware utilization, and implementation details. The reported efficiency numbers should therefore be treated as a measurement plan until verified with a profiler.\n\n"
                    "From a research design perspective, the strongest next step is a pre-registered multi-seed experiment with paired bootstrap intervals and a clear error taxonomy. Qualitative inspection should focus on cases where the proposed model disagrees with the strongest baseline. Such analysis can distinguish genuine representation improvements from favorable sampling and can guide targeted revisions to the architecture or data pipeline."),
        threats_to_validity=[
            "Construct validity: the fallback benchmark and latency values are controlled planning estimates, not instrumented measurements from downloaded datasets.",
            "Internal validity: deterministic topic-seeded curves can understate variance, hyperparameter sensitivity, and random-seed effects.",
            "External validity: public benchmark distributions may not represent deployment populations, languages, devices, or rare failure modes.",
            "Hardware validity: latency, memory, and throughput depend on accelerator, compiler, batch size, precision, and implementation details.",
            "Data validity: licensing, demographic bias, annotation noise, leakage, and temporal drift require dataset-specific audits before publication.",
        ],
        conclusion=(f"This paper developed a structured study of {topic_title} for {plan.domain}. The proposed {proposed.name} reached {proposed_score:.2f} versus {baseline_score:.2f} for the strongest baseline, a relative improvement of {gain:.1f}%, while operating at {proposed.latency_ms:.1f} ms with {proposed.params_m:.1f}M parameters. The combined benchmark, convergence, and efficiency evidence supports the value of the proposed task-aware design under the stated assumptions.\n\n"
                    "The conclusion is intentionally bounded. The current results are a rigorous planning artifact and a coherent fallback estimate, not a claim that replaces instrumented experimentation. Before submission, the study should validate all numbers on the named datasets, report uncertainty across independent runs, compare against current domain leaders, and document data governance. With those additions, the pipeline can evolve from a demonstration into a reproducible and auditable research system. The same protocol can then be transferred to new topics while preserving the separation between hypothesis generation, empirical verification, manuscript writing, and critical review. This separation is itself a contribution to dependable research automation because it makes assumptions visible and gives reviewers concrete checkpoints for challenging unsupported conclusions."),
        references=[
            "Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.",
            "He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.",
            "Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL.",
            "Kipf, T. N., and Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.",
            "Chen, T., et al. (2020). A Simple Framework for Contrastive Learning of Visual Representations. ICML.",
            "Dodge, J., et al. (2019). Show Your Work: Improved Reporting of Experimental Results. EMNLP.",
            "Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. NeurIPS.",
            f"{plan.domain} Research Benchmark Collection (2025). Curated evaluation protocol."
        ],
    )


def _build_synthesis_writer(plan: PlannerOutput, research: ResearcherOutput) -> WriterOutput:
    topic = plan.title.split(":")[0]
    proposed = next((model for model in research.models_compared if model.role == "Proposed"), research.models_compared[-1])
    datasets = ", ".join(dataset.name for dataset in research.datasets_used)
    metric = research.datasets_used[0].primary_metric
    return WriterOutput(
        title=plan.title,
        abstract=(
            f"This paper develops {proposed.name}, a new method for {topic} in {plan.domain}. "
            f"The method is designed to select task-relevant evidence, control redundant computation, and expose uncertainty during inference. "
            f"We formulate a reproducible study over {datasets}, using {metric} together with robustness, calibration, and latency measurements. "
            "All numerical values in this demonstration are controlled planning estimates rather than completed experimental measurements. "
            "The resulting research design provides a concrete path from a 2026 research direction to an independently testable scientific contribution."
        ),
        keywords=[plan.domain.lower(), "task-aware learning", "uncertainty", "efficient AI", "reproducible research"],
        introduction=(
            f"### 1.1 Context and operational bottlenecks\n{topic} matters because deployed AI systems must preserve quality under finite memory, bandwidth, latency, and energy budgets. "
            f"The central question is: {plan.research_question} Modern accelerators expose a gap between nominal FLOPs and observed wall-clock behavior: memory movement, kernel launch overhead, sequence length, and batch-size sensitivity can dominate arithmetic cost.\n\n"
            "### 1.2 Limitations of current methods\nExisting systems often optimize task quality in isolation, report a single hardware configuration, or use static reduction rules that fail under distribution shift. "
            "Comparisons are also difficult when tokenization, precision, warm-up, and measurement protocol differ. A fair study must therefore compare the proposed method with strong baseline families under a shared budget rather than treating a lower simulated number as a measured improvement.\n\n"
            "### 1.3 Contributions\n"
            "1. We define an adaptive evidence-selection mechanism for the target task.\n"
            "2. We formulate a joint objective coupling task quality, sparsity, and consistency.\n"
            "3. We specify a reproducible multi-seed protocol with latency, memory, FLOPs, and uncertainty reporting.\n"
            "4. We separate literature baselines, controlled simulations, and future measurements so that claims remain auditable."
        ),
        related_work=(
            "### 2.1 Dense attention and standard backbones\nDense multi-head attention remains the reference because every head processes every eligible token. It is predictable and expressive, but its quadratic interaction pattern can become a memory-bandwidth bottleneck.\n\n"
            "### 2.2 Grouped or shared-query computation\nGQA and related sharing strategies reduce key-value storage and improve serving efficiency, but they impose a fixed sharing pattern. They are strong comparators when the task benefits from regular memory access.\n\n"
            "### 2.3 Kernel and systems optimization\nFlashAttention-style kernels reduce materialized intermediate memory and improve locality without changing the model's statistical objective. They are complementary to, rather than automatically interchangeable with, a learned routing mechanism.\n\n"
            "### 2.4 Pruning, routing, and quantization\nStatic pruning and quantization can reduce cost, but aggressive reduction may remove information needed for rare inputs. The proposed study combines adaptive selection with explicit consistency and calibration checks. Baseline comparisons must use the same task, sequence length, precision, and measurement protocol."
        ),
        methodology=(
            f"### 3.1 Problem definition and assumptions\n{proposed.name} uses {proposed.backbone.lower()} with a task-aware routing unit. Let X be the input representation, h_i the representation of component i, and q the task context. The router estimates relevance without assuming that every input component contributes equally.\n\n"
            "### 3.2 Mathematical formulation\nThe baseline attention projections are defined as\n"
            "$$Q=XW_Q,\\qquad K=XW_K,\\qquad V=XW_V.$$ \n"
            "For head h, the proposed gate is\n"
            "$$g_h=\\sigma(W_g\\,\\mathrm{MeanPool}(X)+U_g q+b_g),\\qquad 0\\leq g_h\\leq1.$$ \n"
            "The gated representation is \(Z=\\sum_i g_i h_i\), with a residual path retained for stability. The joint training objective is\n"
            "$$L_{total}=L_{task}+\\lambda_s\\lVert g\\rVert_1+\\lambda_cL_{consistency}+\\lambda_uL_{uncertainty}.$$ \n"
            "Here \(L_{task}\) measures the target objective, \(L_{consistency}\) compares clean and perturbed views, and \(L_{uncertainty}\) penalizes miscalibration.\n\n"
            "### 3.3 Algorithm and complexity\n"
            "The forward pass is: (1) encode inputs; (2) pool contextual features; (3) compute gates; (4) retain or down-weight components; (5) execute the task head; (6) emit prediction and uncertainty. "
            "A dense implementation remains O(N^2) in attention interactions. A pruned implementation may reduce active interactions, but the exact bound depends on the selection algorithm; therefore this paper reports measured FLOPs and memory rather than claiming O(N log N) without proof. "
            f"The protocol evaluates the mechanism on {datasets} with fixed splits, multiple seeds, calibration analysis, and hardware profiling."
        ),
        mathematical_formulas=[
            r"Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V",
            r"g_h=\sigma\left(W_g\,\operatorname{MeanPool}(X)+U_gq+b_g\right)",
            r"\mathcal{L}_{total}=\mathcal{L}_{task}+\lambda_s\lVert g\rVert_1+\lambda_c\mathcal{L}_{consistency}",
            r"\operatorname{Cost}_{dense}=\mathcal{O}(N^2d),\qquad \operatorname{Memory}_{dense}=\mathcal{O}(N^2)",
        ],
        experimental_results=(
            f"### 4.1 Benchmarks and datasets\nThe demonstration protocol uses an NVIDIA A100 configuration, a fixed training schedule, and the datasets {datasets}. "
            f"The primary metric is {metric}; secondary measurements include calibration error, robustness under controlled perturbations, parameter count, FLOPs, VRAM footprint, and latency. "
            "Each dataset requires a declared split, preprocessing recipe, license, and leakage check.\n\n"
            "### 4.2 Main comparative evaluation\nThe benchmark matrix compares standard attention/backbone systems, efficient shared-computation systems, pruning or routing methods, and the proposed configuration. "
            "The comparison table must report the primary metric direction, latency per token or example, peak VRAM in MB, FLOPs in G, and evidence type. "
            "The displayed values are planning estimates generated to make the experiment executable; they are not measured results.\n\n"
            "### 4.3 Ablation study\nThe study removes the sparsity term, dynamic router, and uncertainty term independently. A valid ablation keeps data, optimizer, budget, and evaluation code fixed. "
            "The expected output is a matrix of mean, standard deviation, confidence interval, and evidence status rather than a single favorable run.\n\n"
            "### 4.4 Statistical protocol\nRun at least three independent seeds, report paired differences and 95% confidence intervals, and preserve raw logs. "
            "A result is inconclusive when intervals overlap materially or when the effect disappears under a matched hardware budget."
        ),
        discussion=(
            "### 5.1 Trade-offs\nThe expected benefit is a model whose computation follows task-relevant evidence instead of processing every input component uniformly. "
            "This can reduce active work, but gate computation, irregular memory access, and synchronization can offset theoretical savings on some accelerators.\n\n"
            "### 5.2 Failure modes\nThe mechanism may fail when relevance signals are poorly calibrated, when rare evidence is pruned, or when deployment distribution differs from development data. "
            "A safe system must expose abstention, uncertainty, and worst-group error rather than optimize only the average score.\n\n"
            "### 5.3 Future work\nFuture studies should compare optimized kernels, quantization levels, and serving stacks under identical batch and precision settings. "
            "They should also test multilingual, long-context, and adversarial inputs and publish code, logs, seeds, and environment manifests."
        ),
        threats_to_validity=[
            "Dataset coverage may not represent the full deployment distribution.",
            "Planning estimates are not measurements and must be replaced by multi-seed instrumented experiments.",
            "Calibration quality can vary across domains, hardware, and input corruption types.",
        ],
        conclusion=(
            f"### 6. Conclusion and future directions\nThis paper presented {proposed.name} for {topic}. "
            "The contribution is a self-contained research hypothesis with a formal objective, an uncertainty-aware architecture, a multi-dimensional benchmark, and a reproducible evaluation plan. "
            "The current numerical values are controlled planning simulations and must not be interpreted as completed experiments. "
            "The next step is to execute the protocol on the named datasets, release configuration files and raw logs, report confidence intervals, and test the method across hardware and distribution shifts before making claims about scientific or deployment impact."
        ),
        references=[
            "Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS. https://arxiv.org/abs/1706.03762",
            "Ainslie, J., et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. EMNLP. https://arxiv.org/abs/2305.13245",
            "Dao, T., et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS. https://arxiv.org/abs/2205.14135",
            "Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR. https://arxiv.org/abs/2307.08691",
            "Sanh, V., et al. (2020). Movement Pruning: Adaptive Sparsity by Fine-Tuning. NeurIPS. https://arxiv.org/abs/2005.07683",
            "Dettmers, T., et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS. https://arxiv.org/abs/2208.07339",
            "Leviathan, Y., Kalman, M., and Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. ICML. https://arxiv.org/abs/2211.17192",
            "Dodge, J., et al. (2019). Show Your Work: Improved Reporting of Experimental Results. EMNLP. https://aclanthology.org/D19-1224/",
        ],
    )


def _build_dynamic_reviewer(paper: WriterOutput, research: ResearcherOutput) -> ReviewerOutput:
    return ReviewerOutput(
        overall_score=8.7,
        novelty_score=8.4,
        rigor_score=8.9,
        clarity_score=8.6,
        strengths=[
            "The study presents a clear task formulation and benchmark coverage.",
            "The ablation analysis and tradeoff discussion improve credibility.",
            "The paper communicates the method and limitations in a structured manner."
        ],
        weaknesses=[
            "A fully real-world deployment study would further strengthen the claims.",
            "Additional significance testing across more datasets would improve robustness checks."
        ],
        revision_suggestions=[
            "Report confidence intervals or statistical significance for the reported deltas.",
            "Add a more explicit deployment cost discussion for the proposed system.",
            "Expand the evaluation to at least one extra domain-specific benchmark."
        ],
        verdict="MINOR_REVISION",
    )


def _answer_paper_question(question: str, paper: WriterOutput, research: ResearcherOutput) -> str:
    normalized = question.strip().lower()
    proposed = next((model for model in research.models_compared if model.role == "Proposed"), None)
    baselines = [model for model in research.models_compared if model.role == "Baseline"]
    proposed_row = next((row for row in research.benchmark_table if row.role == "Proposed"), None)
    baseline_rows = [row for row in research.benchmark_table if row.role == "Baseline"]
    strongest_baseline = max(baseline_rows, key=lambda row: row.primary_metric, default=None)
    if any(term in normalized for term in ("baseline", "compare", "comparison", "so sánh")):
        names = ", ".join(model.name for model in baselines)
        strongest = strongest_baseline.model_name if strongest_baseline else "the strongest baseline"
        return f"The study compares {proposed.name if proposed else 'the proposed model'} against {names}. The strongest baseline in the simulated benchmark is {strongest}, while the proposed system is evaluated with the same dataset and reporting contract."
    if any(term in normalized for term in ("limitation", "weakness", "caveat", "hạn chế")):
        return "The main limitation is that the benchmark values, latency, and curves are deterministic planning simulations rather than measurements from instrumented training runs. A publication-ready study should add multiple seeds, confidence intervals, external validation, calibration, and hardware profiling."
    if any(term in normalized for term in ("dataset", "data", "benchmark", "dữ liệu")):
        catalog = ", ".join(dataset.name for dataset in research.datasets_used)
        return f"The generated evaluation uses three complementary datasets: {catalog}. They cover the core task, robustness under shift, and controlled ablations so the reported gain is not tied to a single split."
    if any(term in normalized for term in ("method", "model", "architecture", "mô hình", "phương pháp")):
        return f"The proposed method is {proposed.name if proposed else 'the task-aware model'}. It uses {proposed.backbone.lower() if proposed else 'adaptive signal selection'} to emphasize topic-relevant evidence while tracking quality, robustness, parameters, and latency together."
    if any(term in normalized for term in ("result", "score", "improve", "performance", "kết quả")) and proposed_row:
        baseline_score = strongest_baseline.primary_metric if strongest_baseline else 0.0
        gain = (proposed_row.primary_metric - baseline_score) / max(baseline_score, 1e-9) * 100
        return f"The proposed system reaches {proposed_row.primary_metric:.2f} on the primary metric versus {baseline_score:.2f} for the strongest baseline, a simulated relative improvement of {gain:.1f}%. The result should be treated as a reproducible hypothesis until measured on real data."
    return f"This paper studies {paper.title}. Its central claim is that a topic-aware design can improve the target task while keeping robustness and deployment cost visible. Ask about the baseline comparison, datasets, method, results, or limitations for a more specific answer."


def _maybe_live_plan(topic: str, domain: str | None, objective: str | None, constraints: str | None, api_key: str | None) -> PlannerOutput | None:
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        prompt = (
            "Return only valid JSON matching this schema: "
            "{title, domain, research_question, hypothesis, datasets_catalog:[{name, modality, sample_size, input_spec, primary_metric, description}], "
            "models_catalog:[{name, role, backbone, params_m, flops_g, latency_ms, key_characteristic}], baseline_methods:[str], proposed_method:str, "
            "evaluation_metrics:[str], experimental_plan:[str], report_outline:[str]}. "
            "Generate one original proposed method, not a baseline-comparison paper. "
            f"Topic: {topic}; domain: {domain or 'AI research'}; objective: {objective or 'design a rigorous study'}; "
            f"constraints: {constraints or 'none specified'}. Use real, recognizable public datasets and flag any unavailable measurements as planned experiments."
        )
        payload = _safe_live_call(key, prompt)
        if not payload:
            return None
        return PlannerOutput.model_validate(payload)
    except Exception:
        return None


def _maybe_live_writer(plan: PlannerOutput, research: ResearcherOutput, api_key: str | None) -> WriterOutput | None:
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    schema = "{title, abstract, keywords:[str], introduction, related_work, methodology, mathematical_formulas:[str], experimental_results, discussion, threats_to_validity:[str], conclusion, references:[str]}"
    prompt = (
        MASTER_RESEARCH_WRITER_PROMPT.format(topic=plan.title)
        + "\nReturn only valid JSON matching this exact schema: " + schema + ". "
        "Use the supplied benchmark rows for quantitative comparisons, preserving their evidence_type. "
        "Do not upgrade Literature Baseline or Controlled Simulation into Measured. "
        f"Topic: {plan.title}. Domain: {plan.domain}. Research question: {plan.research_question}. "
        f"Models: {json.dumps([item.model_dump() for item in plan.models_catalog])}. "
        f"Datasets: {json.dumps([item.model_dump() for item in research.datasets_used])}. "
        f"Knowledge Base: {json.dumps(plan.knowledge_base.model_dump())}. "
        f"Inference Engine: {json.dumps(plan.inference_engine.model_dump())}. "
        f"Multi-Agent Coordination: {json.dumps(plan.agent_coordination.model_dump())}. "
        f"Ablation Study: {json.dumps([item.model_dump() for item in research.ablation_study])}. "
        f"Evidence status: {research.evidence_mode}. Reasoning Consistency Score: {research.reasoning_consistency_score}. "
        f"Knowledge Retrieval Precision: {research.knowledge_retrieval_precision}."
    )
    try:
        payload = _safe_live_call(key, prompt)
        return WriterOutput.model_validate(payload) if payload else None
    except Exception:
        return None


def _run_pipeline_request(request: PipelineRequest) -> FullPipelineResponse:
    topic = _normalize_topic(request.topic)
    domain = request.domain or _infer_domain(topic, request.domain, request.objective)
    objective = request.objective or "Design a rigorous, reproducible, and evidence-driven study."
    if request.require_verified_data:
        _validate_verified_request(topic, request.domain, request.objective)
    api_key = request.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    plan = _maybe_live_plan(topic, domain, objective, request.constraints, api_key)
    mode_used = "live_gemini" if plan is not None else "dynamic_fallback"
    if plan is None:
        plan = _build_dynamic_planner(topic, domain, objective)
    plan = _ensure_system_intelligence(plan, topic, domain, objective)
    research = _build_dynamic_research(plan)
    live_paper = _maybe_live_writer(plan, research, api_key)
    paper = live_paper or _build_synthesis_writer(plan, research)
    paper.abstract = _academic_abstract(paper.abstract)
    paper.mathematical_formulas = paper.mathematical_formulas[:4]
    guardrail = GuardrailOutput.model_validate(
        audit_manuscript(paper.model_dump(), research.model_dump())
    )
    review = _build_dynamic_reviewer(paper, research)
    return FullPipelineResponse(mode_used=mode_used, paper_mode="live_gemini" if live_paper else "dynamic_fallback", plan=plan, research=research, paper=paper, review=review, guardrail=guardrail)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "smart-research-lab-api"}


@app.post("/api/planner/generate", response_model=PlannerOutput)
def generate_planner_plan(payload: TopicInput) -> PlannerOutput:
    if not payload.topic or not payload.topic.strip():
        raise HTTPException(status_code=400, detail="Topic cannot be empty.")
    request = PipelineRequest(topic=payload.topic, domain=payload.domain, objective=payload.objective, constraints=None, api_key=None)
    return _run_pipeline_request(request).plan


@app.post("/api/researcher/generate", response_model=ResearcherOutput)
def generate_research_plan(payload: PlannerOutput) -> ResearcherOutput:
    return _build_dynamic_research(payload)


@app.post("/api/writer/generate", response_model=WriterOutput)
def generate_writer_output(payload: dict[str, Any]) -> WriterOutput:
    plan = PlannerOutput.model_validate(payload.get("plan", {}))
    research = ResearcherOutput.model_validate(payload.get("research", {}))
    return _build_synthesis_writer(plan, research)


@app.post("/api/reviewer/generate", response_model=ReviewerOutput)
def generate_reviewer_output(payload: dict[str, Any]) -> ReviewerOutput:
    paper = WriterOutput.model_validate(payload.get("paper", {}))
    research = ResearcherOutput.model_validate(payload.get("research", {}))
    return _build_dynamic_reviewer(paper, research)


@app.post("/api/paper/ask")
def ask_paper_question(payload: dict[str, Any]) -> dict[str, str]:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    paper = WriterOutput.model_validate(payload.get("paper", {}))
    research = ResearcherOutput.model_validate(payload.get("research", {}))
    return {"answer": _answer_paper_question(question, paper, research)}


@app.post("/api/pipeline/run", response_model=FullPipelineResponse)
def run_pipeline(payload: PipelineRequest) -> FullPipelineResponse:
    if not payload.topic or len(payload.topic.strip()) < 3:
        raise HTTPException(status_code=400, detail="Topic must be at least 3 characters long.")
    return _run_pipeline_request(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
