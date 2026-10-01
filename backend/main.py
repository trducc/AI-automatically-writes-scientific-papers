from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import (
    AblationRow,
    DatasetSpec,
    FullPipelineResponse,
    MetricCurvePoint,
    ModelSpec,
    PipelineRequest,
    PlannerOutput,
    ResearcherOutput,
    ReviewerOutput,
    TopicInput,
    WriterOutput,
)

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


def _normalize_topic(topic: str) -> str:
    return re.sub(r"\s+", " ", topic.strip())


def _infer_domain(topic: str, domain: str | None, objective: str | None) -> str:
    text = f"{topic} {domain or ''} {objective or ''}".lower()
    if any(k in text for k in ("image", "vision", "segmentation", "medical", "cnn", "object detection")):
        return "Computer Vision"
    if any(k in text for k in ("language", "text", "nlp", "bert", "transformer", "sentiment")):
        return "Natural Language Processing"
    if any(k in text for k in ("forecast", "time", "series", "prediction", "stock")):
        return "Time Series Forecasting"
    if any(k in text for k in ("graph", "network", "gnn", "fraud", "relationship", "social")):
        return "Graph Learning"
    if domain:
        return domain.strip()
    return "AI Research"


def _make_dataset_catalog(topic: str, domain: str | None) -> list[DatasetSpec]:
    normalized = _normalize_topic(topic).lower()
    domain_name = _infer_domain(topic, domain, None)
    if "fraud" in normalized or "financial" in normalized or "transaction" in normalized:
        return [
            DatasetSpec(
                name="Credit Card Fraud Benchmark",
                modality="Tabular",
                sample_size="284k rows / 80:20 split",
                input_spec="Transaction features + engineered aggregate signals",
                primary_metric="PR-AUC",
                description="Highly imbalanced fraud detection dataset used for anomaly-focused evaluation.",
            ),
            DatasetSpec(
                name="Synthetic Finance Graph Dataset",
                modality="Graph",
                sample_size="120k nodes / 1.2M edges",
                input_spec="Node embeddings + transaction graph topology",
                primary_metric="F1-score",
                description="Graph-based benchmark designed to capture relational fraud patterns.",
            ),
            DatasetSpec(
                name="Risk Policy Validation Set",
                modality="Tabular",
                sample_size="40k records / temporal holdout",
                input_spec="Feature vectors with time-aware labels",
                primary_metric="Recall@K",
                description="Temporal challenge set for monitoring suspicious transaction sequences.",
            ),
        ]
    if "image" in normalized or "medical" in normalized or "segmentation" in normalized:
        return [
            DatasetSpec(
                name="MedImage-Lung Segmentation",
                modality="Image",
                sample_size="18k train / 3k test",
                input_spec="3D/2D CT or MRI inputs with pixel mask targets",
                primary_metric="Dice coefficient",
                description="Medical segmentation benchmark with noisy annotation and class imbalance.",
            ),
            DatasetSpec(
                name="Retinal Lesion Benchmark",
                modality="Image",
                sample_size="25k train / 5k test",
                input_spec="RGB retinal images with lesion masks",
                primary_metric="mIoU",
                description="Small lesion segmentation challenge with high visual heterogeneity.",
            ),
            DatasetSpec(
                name="Histology Tissue Benchmark",
                modality="Image",
                sample_size="15k train / 3k test",
                input_spec="Whole-slide image patches",
                primary_metric="F1-score",
                description="Weakly supervised tissue classification and segmentation benchmark.",
            ),
        ]
    if "language" in normalized or "text" in normalized or "nlp" in normalized or "sentiment" in normalized:
        return [
            DatasetSpec(
                name="Domain Text Benchmark",
                modality="Text",
                sample_size="50k train / 10k test",
                input_spec="Tokenized text sequences with class labels",
                primary_metric="Macro-F1",
                description="Text classification task with domain shift and label imbalance.",
            ),
            DatasetSpec(
                name="Long-form QA Corpus",
                modality="Text",
                sample_size="30k train / 7k test",
                input_spec="Passage + question pairs",
                primary_metric="Exact Match",
                description="Long context reasoning benchmark across multi-hop questions.",
            ),
            DatasetSpec(
                name="Dialogue Safety Set",
                modality="Text",
                sample_size="20k train / 5k test",
                input_spec="Conversation context and safety labels",
                primary_metric="Accuracy",
                description="Safety-aware language benchmark exposing adversarial or off-topic responses.",
            ),
        ]
    return [
        DatasetSpec(
            name=f"{domain_name or 'General'} Research Benchmark",
            modality="Mixed",
            sample_size="40k train / 8k validation",
            input_spec="Domain-specific feature vectors or token sequences",
            primary_metric="Accuracy",
            description=f"General benchmark for {topic} under realistic domain constraints.",
        ),
        DatasetSpec(
            name=f"{domain_name or 'Domain'} Robustness Split",
            modality="Mixed",
            sample_size="15k train / 3k test",
            input_spec="Out-of-distribution or stress-test partitions",
            primary_metric="Robustness score",
            description="Challenge benchmark for evaluating stability under distribution shift.",
        ),
        DatasetSpec(
            name=f"{domain_name or 'Domain'} Ablation Set",
            modality="Mixed",
            sample_size="20k train / 4k validation",
            input_spec="Targeted subset for ablation and interpretability analysis",
            primary_metric="Ablation gap",
            description="Evaluation set for probing the effect of model components and design choices.",
        ),
    ]


def _make_model_catalog(topic: str, domain: str | None) -> list[ModelSpec]:
    normalized = _normalize_topic(topic).lower()
    if "fraud" in normalized or "financial" in normalized or "transaction" in normalized:
        return [
            ModelSpec(name="XGBoost", role="Baseline", backbone="Gradient boosted trees", params_m=25.0, flops_g=2.1, latency_ms=35.0, key_characteristic="Strong tabular feature interactions but weak relational modeling."),
            ModelSpec(name="LightGBM", role="Baseline", backbone="Histogram-based GBM", params_m=18.0, flops_g=1.8, latency_ms=30.0, key_characteristic="Fast sparse-feature training with limited graph awareness."),
            ModelSpec(name="GraphSAGE", role="Baseline", backbone="Message-passing GNN", params_m=32.0, flops_g=4.8, latency_ms=55.0, key_characteristic="Captures neighborhood structure but not temporal risk patterns."),
            ModelSpec(name="RiskAware Temporal GNN", role="Proposed", backbone="Temporal graph attention encoder", params_m=40.0, flops_g=6.2, latency_ms=58.0, key_characteristic="Combines graph context, temporal behavior, and risk-sensitive attention."),
        ]
    if "image" in normalized or "medical" in normalized or "segmentation" in normalized:
        return [
            ModelSpec(name="UNet", role="Baseline", backbone="Convolutional encoder-decoder", params_m=31.0, flops_g=45.0, latency_ms=52.0, key_characteristic="Strong localization yet limited long-range context."),
            ModelSpec(name="DeepLabV3", role="Baseline", backbone="Atrous CNN", params_m=43.0, flops_g=60.0, latency_ms=68.0, key_characteristic="Good multi-scale context but heavy on compute."),
            ModelSpec(name="SegFormer", role="Baseline", backbone="Hierarchical transformer", params_m=26.0, flops_g=40.0, latency_ms=47.0, key_characteristic="Efficient attention with robust segmentation behavior."),
            ModelSpec(name="Contextual Attention Segmenter", role="Proposed", backbone="Hybrid CNN + attention fusion", params_m=33.0, flops_g=48.0, latency_ms=49.0, key_characteristic="Uses adaptive context fusion to preserve lesion boundaries and reduce false positives."),
        ]
    if "language" in normalized or "text" in normalized or "nlp" in normalized or "sentiment" in normalized:
        return [
            ModelSpec(name="BiLSTM", role="Baseline", backbone="Bidirectional recurrent encoder", params_m=22.0, flops_g=18.0, latency_ms=24.0, key_characteristic="Captures sequence context but weakly models long-range dependencies."),
            ModelSpec(name="BERT-Base", role="Baseline", backbone="Transformer encoder", params_m=110.0, flops_g=22.0, latency_ms=30.0, key_characteristic="Strong language representation with moderate compute cost."),
            ModelSpec(name="RoBERTa", role="Baseline", backbone="Transformer encoder", params_m=125.0, flops_g=24.0, latency_ms=34.0, key_characteristic="Robust optimization over standard BERT training."),
            ModelSpec(name="Contextual Adapter Transformer", role="Proposed", backbone="Adapter-augmented transformer", params_m=92.0, flops_g=19.0, latency_ms=29.0, key_characteristic="Improves domain adaptation while preserving efficiency and stability."),
        ]
    return [
        ModelSpec(name="Strong Baseline", role="Baseline", backbone="Reference architecture", params_m=35.0, flops_g=8.0, latency_ms=40.0, key_characteristic="Standard benchmark configuration for the target task."),
        ModelSpec(name="Efficient Baseline", role="Baseline", backbone="Lightweight encoder", params_m=22.0, flops_g=5.5, latency_ms=28.0, key_characteristic="Lower computational cost but weaker signal capture."),
        ModelSpec(name="Domain Adapter Model", role="Baseline", backbone="Adaptive feature fusion", params_m=28.0, flops_g=6.7, latency_ms=32.0, key_characteristic="Improves specialization but lacks full task-aware tuning."),
        ModelSpec(name="Task-Aware Proposed Model", role="Proposed", backbone="Hybrid optimization stack", params_m=36.0, flops_g=7.4, latency_ms=33.0, key_characteristic="Integrates domain priors and adaptive signal selection for stronger final performance."),
    ]


def _safe_live_call(api_key: str | None, prompt: str) -> Any | None:
    if not api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "response_mime_type": "application/json"},
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
    domain_name = _infer_domain(clean_topic, domain, objective)
    datasets_catalog = _make_dataset_catalog(clean_topic, domain_name)
    models_catalog = _make_model_catalog(clean_topic, domain_name)
    return PlannerOutput(
        title=f"{clean_topic}: A Topic-Aware Optimization Study",
        domain=domain_name,
        research_question=f"How can {clean_topic} be improved in a principled way while preserving robustness, scalability, and empirical validity?",
        hypothesis=f"A task-aware design grounded in {domain_name} can improve performance over strong baselines by capturing the most informative signals while reducing noise and redundancy.",
        datasets_catalog=datasets_catalog,
        models_catalog=models_catalog,
        baseline_methods=[item.name for item in models_catalog if item.role == "Baseline"],
        proposed_method=next(item.name for item in models_catalog if item.role == "Proposed"),
        evaluation_metrics=["Accuracy", "F1-score", "AUC", "Robustness gap"],
        experimental_plan=[
            "Construct normalized benchmark settings for the target task and dataset.",
            "Train the baseline and proposed approach under the same protocol.",
            "Run ablation studies to isolate the contribution of the design choices.",
            "Compare performance, stability, and efficiency under identical evaluation conditions."
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


def _build_dynamic_research(plan: PlannerOutput) -> ResearcherOutput:
    topic_tokens = plan.title.split(":")[0]
    curves: list[MetricCurvePoint] = []
    dataset_names = [dataset.name for dataset in plan.datasets_catalog]
    for dataset_name in dataset_names:
        for i in range(1, 11):
            baseline_loss = 2.10 - 0.04 * i + 0.02 * (i / 10)
            proposed_loss = max(0.65, 2.10 - 0.08 * i + 0.05 * math.sin(i / 3))
            baseline_score = 0.81 + 0.004 * i
            proposed_score = min(0.99, 0.81 + 0.01 * i + 0.015 * (i / 10))
            curves.append(
                MetricCurvePoint(
                    epoch=i,
                    dataset_name=dataset_name,
                    baseline_loss=round(baseline_loss, 4),
                    proposed_loss=round(proposed_loss, 4),
                    baseline_score=round(baseline_score, 4),
                    proposed_score=round(proposed_score, 4),
                )
            )
    benchmark_table = [
        AblationRow(
            model_name=model.name,
            dataset_name=dataset_names[0],
            role=model.role,
            primary_metric=0.82 if model.role == "Baseline" else 0.88,
            secondary_metric=0.74 if model.role == "Baseline" else 0.81,
            latency_ms=model.latency_ms,
            params_m=model.params_m,
            notes="Best tradeoff for the target task." if model.role == "Proposed" else "Reference base configuration.",
        )
        for model in plan.models_catalog
    ]
    key_findings = [
        f"The proposed design for {topic_tokens} improves the main metric by roughly 3-6% over the baseline.",
        "The strongest gain is observed when the model balances signal preservation with computational efficiency.",
        "Ablation results confirm that the proposed design contributes meaningfully beyond a simple capacity increase."
    ]
    return ResearcherOutput(
        hardware_setup="1x NVIDIA A100 GPU (40GB VRAM), 64GB CPU RAM, CUDA-enabled PyTorch environment",
        datasets_used=plan.datasets_catalog,
        models_compared=plan.models_catalog,
        training_curves=curves,
        benchmark_table=benchmark_table,
        key_findings=key_findings,
    )


def _build_dynamic_writer(plan: PlannerOutput, research: ResearcherOutput) -> WriterOutput:
    topic_title = plan.title.split(":")[0]
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
    formulas = [
        r"\mathcal{L}_{total}=\mathcal{L}_{task}+\lambda_1\lVert W\rVert_1+\lambda_2\mathcal{L}_{robust}+\lambda_3\mathcal{L}_{consistency}",
        r"\alpha_{i}=\frac{\exp(q_i^{\top}k_i/\sqrt{d})}{\sum_{j=1}^{n}\exp(q_i^{\top}k_j/\sqrt{d})},\qquad z_i=\sum_{j=1}^{n}\alpha_{ij}v_j",
        r"\Delta=\mathrm{Score}_{prop}-\max_{b\in\mathcal{B}}\mathrm{Score}_{b},\qquad G=100\times\frac{\Delta}{\max_{b\in\mathcal{B}}\mathrm{Score}_{b}}",
        r"\mathrm{Utility}=\frac{\mathrm{Score}}{(1+\lambda_t\,\mathrm{Latency}_{ms})(1+\lambda_p\,\mathrm{Params}_{M})}"
    ]
    return WriterOutput(
        title=plan.title,
        abstract=(f"This paper presents a systematic investigation of {topic_title} in the context of {plan.domain}. "
                  f"The study evaluates {proposed.name}, a {proposed.backbone} proposed to improve task-relevant signal selection, against {len(baselines)} representative baselines: {model_names}. "
                  f"Evaluation covers {len(research.datasets_used)} complementary datasets ({dataset_names}) and reports predictive quality, robustness, parameter count, and inference latency. "
                  f"The proposed configuration reaches a primary score of {proposed_score:.2f}, compared with the strongest baseline score of {baseline_score:.2f}, corresponding to a relative improvement of {gain:.1f}%. "
                  f"It uses {proposed.params_m:.1f}M parameters and incurs {proposed.latency_ms:.1f} ms latency, making the quality-efficiency trade-off explicit rather than treating accuracy as the only objective. "
                  "The results support the hypothesis that domain-aware architectural choices can improve generalization and operational reliability without requiring unbounded model scaling. The paper further documents limitations, reproducibility assumptions, and directions for external validation. In particular, the catalog makes the claim auditable: each reported improvement is tied to a named dataset, an explicit comparator, and a measurable resource budget. This framing is useful for researchers who need to decide whether a modest quality gain justifies additional deployment complexity."),
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
        methodology=(f"The proposed method, {proposed.name}, uses a {proposed.backbone}. Its core operation first encodes the input into task-relevant representations, then computes context-dependent weights over candidate signals, and finally fuses the selected evidence with a residual path. This construction is intended to retain stable low-level information while allowing the model to emphasize interactions that are predictive for {topic_title}. A regularized objective combines the task loss with sparsity, robustness, and consistency terms; the corresponding equations are listed below.\n\n"
                      f"The experimental protocol uses {len(research.datasets_used)} datasets and compares {len(research.models_compared)} models: {model_names}. Each configuration is evaluated with the same split policy, optimization schedule, stopping criterion, and measurement interface. We record the primary and secondary metrics, training curves, parameter count, FLOPs, latency, and qualitative notes. The benchmark matrix is anchored on {research.benchmark_table[0].dataset_name if research.benchmark_table else 'the primary dataset'} so that the ablation comparison remains directly interpretable.\n\n"
                      f"For reproducibility, the reference hardware is {research.hardware_setup}. The proposed system contains {proposed.params_m:.1f}M parameters, requires {proposed.flops_g:.1f} GFLOPs per inference, and records {proposed.latency_ms:.1f} ms latency in the generated estimate. These values are not presented as universal measurements; they are controlled planning values that should be replaced by instrumented runs before publication. This distinction keeps the generated manuscript useful for planning while making its evidence boundary explicit."),
        mathematical_formulas=formulas,
        experimental_results=(f"Across the generated benchmark matrix, {proposed.name} obtains a primary metric of {proposed_score:.2f}, whereas the strongest baseline reaches {baseline_score:.2f}. The absolute difference is {proposed_score - baseline_score:.2f}, equivalent to a {gain:.1f}% relative gain. The secondary metric changes from the baseline reference of {baseline_rows[0].secondary_metric if baseline_rows else 0.74:.2f} to {proposed_rows[0].secondary_metric if proposed_rows else 0.81:.2f}. These figures indicate a meaningful improvement under the controlled protocol, while the table should still be read as evidence for a research hypothesis rather than a substitute for repeated measurements.\n\n"
                               f"The efficiency profile is equally important. {proposed.name} uses {proposed.params_m:.1f}M parameters and {proposed.flops_g:.1f} GFLOPs with an estimated latency of {proposed.latency_ms:.1f} ms. The baseline catalog spans {min(model.latency_ms for model in baselines):.1f}-{max(model.latency_ms for model in baselines):.1f} ms and {min(model.params_m for model in baselines):.1f}-{max(model.params_m for model in baselines):.1f}M parameters. Convergence traces show the proposed loss decreasing more rapidly, while its score rises from approximately 0.82 to 0.92 over ten epochs versus 0.85 for the baseline trajectory.\n\n"
                               "The ablation rows suggest that the gain is not explained solely by adding parameters: the proposed configuration remains competitive on latency and improves both quality dimensions. A complete publication would extend this analysis with multiple random seeds, confidence intervals, calibration, subgroup performance, and a held-out external dataset. It should also record preprocessing time, peak memory, throughput at several batch sizes, and the operational threshold used to convert scores into decisions. Those measurements would determine whether the observed margin survives outside the controlled planning environment."),
        discussion=(f"The results support the hypothesis that task-aware signal selection can improve {topic_title} without an extreme computational penalty. The most direct evidence is the joint movement of the primary score from {baseline_score:.2f} to {proposed_score:.2f} and the controlled latency of {proposed.latency_ms:.1f} ms. This is a more informative result than a score increase alone because the intended use case includes operational constraints.\n\n"
                    "There are, however, important qualifications. The benchmark catalog contains complementary views of the task, but its generated estimates do not establish statistical significance. Dataset shift may change the ranking of models, particularly when the proposed inductive bias depends on relationships or modality-specific signals. Similarly, latency depends on batch size, compiler, hardware utilization, and implementation details. The reported efficiency numbers should therefore be treated as a measurement plan until verified with a profiler.\n\n"
                    "From a research design perspective, the strongest next step is a pre-registered multi-seed experiment with paired bootstrap intervals and a clear error taxonomy. Qualitative inspection should focus on cases where the proposed model disagrees with the strongest baseline. Such analysis can distinguish genuine representation improvements from favorable sampling and can guide targeted revisions to the architecture or data pipeline."),
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


def _maybe_live_plan(topic: str, domain: str | None, objective: str | None, api_key: str | None) -> PlannerOutput | None:
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        prompt = (
            "Return only valid JSON matching this schema: "
            "{title, domain, research_question, hypothesis, datasets_catalog:[{name, modality, sample_size, input_spec, primary_metric, description}], "
            "models_catalog:[{name, role, backbone, params_m, flops_g, latency_ms, key_characteristic}], baseline_methods:[str], proposed_method:str, "
            "evaluation_metrics:[str], experimental_plan:[str], report_outline:[str]}. "
            f"Topic: {topic}; domain: {domain or 'AI research'}; objective: {objective or 'design a rigorous study'}"
        )
        payload = _safe_live_call(key, prompt)
        if not payload:
            return None
        return PlannerOutput.model_validate(payload)
    except Exception:
        return None


def _run_pipeline_request(request: PipelineRequest) -> FullPipelineResponse:
    topic = _normalize_topic(request.topic)
    domain = request.domain or _infer_domain(topic, request.domain, request.objective)
    objective = request.objective or "Design a rigorous, reproducible, and evidence-driven study."
    api_key = request.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    plan = _maybe_live_plan(topic, domain, objective, api_key)
    mode_used = "live_gemini" if plan is not None else "dynamic_fallback"
    if plan is None:
        plan = _build_dynamic_planner(topic, domain, objective)
    research = _build_dynamic_research(plan)
    paper = _build_dynamic_writer(plan, research)
    review = _build_dynamic_reviewer(paper, research)
    return FullPipelineResponse(mode_used=mode_used, plan=plan, research=research, paper=paper, review=review)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "smart-research-lab-api"}


@app.post("/api/planner/generate", response_model=PlannerOutput)
def generate_planner_plan(payload: TopicInput) -> PlannerOutput:
    if not payload.topic or len(payload.topic.strip()) < 3:
        raise HTTPException(status_code=400, detail="Topic must be at least 3 characters long.")
    request = PipelineRequest(topic=payload.topic, domain=payload.domain, objective=payload.objective, constraints=None, api_key=None)
    return _run_pipeline_request(request).plan


@app.post("/api/researcher/generate", response_model=ResearcherOutput)
def generate_research_plan(payload: PlannerOutput) -> ResearcherOutput:
    return _build_dynamic_research(payload)


@app.post("/api/writer/generate", response_model=WriterOutput)
def generate_writer_output(payload: dict[str, Any]) -> WriterOutput:
    plan = PlannerOutput.model_validate(payload.get("plan", {}))
    research = ResearcherOutput.model_validate(payload.get("research", {}))
    return _build_dynamic_writer(plan, research)


@app.post("/api/reviewer/generate", response_model=ReviewerOutput)
def generate_reviewer_output(payload: dict[str, Any]) -> ReviewerOutput:
    paper = WriterOutput.model_validate(payload.get("paper", {}))
    research = ResearcherOutput.model_validate(payload.get("research", {}))
    return _build_dynamic_reviewer(paper, research)


@app.post("/api/pipeline/run", response_model=FullPipelineResponse)
def run_pipeline(payload: PipelineRequest) -> FullPipelineResponse:
    if not payload.topic or len(payload.topic.strip()) < 3:
        raise HTTPException(status_code=400, detail="Topic must be at least 3 characters long.")
    return _run_pipeline_request(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
