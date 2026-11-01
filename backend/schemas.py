from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TopicInput(BaseModel):
    topic: str = Field(..., min_length=3, description="Research topic in English")
    domain: str | None = Field(default=None, description="Optional domain or subject area")
    objective: str | None = Field(default=None, description="Optional objective or desired goal")
    constraints: list[str] = Field(default_factory=list, description="Optional constraints")


class DatasetSpec(BaseModel):
    name: str = Field(..., description="Dataset name")
    modality: str = Field(..., description="Data modality (Image, Text, Graph, Tabular, etc.)")
    sample_size: str = Field(..., description="Dataset size summary")
    input_spec: str = Field(..., description="Input format or feature shape")
    primary_metric: str = Field(..., description="Primary evaluation metric")
    description: str = Field(..., description="Dataset description and challenge")
    source_url: str = Field(default="", description="Canonical dataset or benchmark URL")


class AblationSpec(BaseModel):
    configuration: str = Field(..., description="Ablation configuration")
    removed_component: str = Field(..., description="Component removed from the proposed system")
    primary_metric: float = Field(..., description="Primary metric for this configuration")
    latency_ms: float = Field(..., ge=0.0, description="Measured or planned latency")
    delta_vs_full: float = Field(..., description="Metric delta relative to the full configuration")
    interpretation: str = Field(..., description="Interpretation of the ablation result")


class ProjectMetadata(BaseModel):
    institution: str
    faculty: str
    subject: str
    instructor: str
    student_name: str
    student_id: str


class KnowledgeBaseSpec(BaseModel):
    taxonomy: list[str] = Field(default_factory=list)
    routing_rules: list[str] = Field(default_factory=list)
    catalog_scope: str = "Domain models, datasets, metrics, and implementation libraries"


class InferenceEngineSpec(BaseModel):
    matching_strategy: str = "Token-boundary matching with domain-priority routing"
    selected_domain: str = ""
    matched_keywords: list[str] = Field(default_factory=list)
    routing_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    routed_configuration: list[str] = Field(default_factory=list)


class AgentCoordinationSpec(BaseModel):
    agents: list[str] = Field(default_factory=list)
    protocol: str = "Shared typed state: plan -> evidence -> metrics -> manuscript -> review"
    coordination_claim: str = "Agents expose intermediate artifacts for traceability and peer review."


class ModelSpec(BaseModel):
    name: str = Field(..., description="Model name")
    role: Literal["Baseline", "Proposed"] = Field(..., description="Baseline or proposed model")
    backbone: str = Field(..., description="Backbone or architecture family")
    params_m: float = Field(..., ge=0.0, description="Parameter count in millions")
    flops_g: float = Field(..., ge=0.0, description="Floating point operations in GFLOPs")
    latency_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds")
    key_characteristic: str = Field(..., description="Technical characteristic")
    vram_mb: float = Field(default=0.0, ge=0.0, description="Peak VRAM footprint in MB when available")
    evidence_type: str = Field(default="Planned / Unverified", description="Measured, Literature Baseline, Controlled Simulation, or Planned")


class PlannerOutput(BaseModel):
    title: str = Field(..., description="Title of the proposed research")
    domain: str = Field(..., description="Research domain")
    research_question: str = Field(..., description="Main research question")
    hypothesis: str = Field(..., description="Hypothesis to investigate")
    datasets_catalog: list[DatasetSpec] = Field(..., min_length=1)
    models_catalog: list[ModelSpec] = Field(..., min_length=1)
    baseline_methods: list[str] = Field(..., min_length=1)
    proposed_method: str = Field(..., description="Proposed method")
    evaluation_metrics: list[str] = Field(..., min_length=1)
    experimental_plan: list[str] = Field(..., min_length=1)
    report_outline: list[str] = Field(..., min_length=1)
    project_metadata: ProjectMetadata | None = None
    knowledge_base: KnowledgeBaseSpec = Field(default_factory=KnowledgeBaseSpec)
    inference_engine: InferenceEngineSpec = Field(default_factory=InferenceEngineSpec)
    agent_coordination: AgentCoordinationSpec = Field(default_factory=AgentCoordinationSpec)


class MetricCurvePoint(BaseModel):
    epoch: int = Field(..., ge=0)
    dataset_name: str = Field(..., description="Dataset associated with the curve")
    baseline_loss: float = Field(..., ge=0.0)
    proposed_loss: float = Field(..., ge=0.0)
    baseline_score: float = Field(..., ge=0.0)
    proposed_score: float = Field(..., ge=0.0)


class AblationRow(BaseModel):
    model_name: str = Field(..., description="Model variant name")
    dataset_name: str = Field(..., description="Dataset name")
    role: str = Field(..., description="Baseline or Proposed role")
    primary_metric: float = Field(..., description="Main performance metric")
    secondary_metric: float = Field(..., description="Secondary metric")
    latency_ms: float = Field(..., ge=0.0)
    params_m: float = Field(..., ge=0.0)
    notes: str = Field(..., description="Interpretation of the result")
    vram_mb: float = Field(default=0.0, ge=0.0)
    flops_g: float = Field(default=0.0, ge=0.0)
    evidence_type: str = Field(default="Planned / Unverified")


class ResearcherOutput(BaseModel):
    hardware_setup: str = Field(..., description="Hardware configuration summary")
    datasets_used: list[DatasetSpec] = Field(..., min_length=1)
    models_compared: list[ModelSpec] = Field(..., min_length=1)
    training_curves: list[MetricCurvePoint] = Field(..., min_length=1)
    benchmark_table: list[AblationRow] = Field(..., min_length=1)
    ablation_study: list[AblationSpec] = Field(default_factory=list)
    key_findings: list[str] = Field(..., min_length=1)
    tech_stack: list[str] = Field(default_factory=list)
    evidence_mode: str = Field(default="Controlled Planning Simulation Estimate")
    reasoning_consistency_score: float = Field(default=0.0, ge=0.0, le=1.0)
    knowledge_retrieval_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    project_metadata: ProjectMetadata | None = None
    knowledge_base: KnowledgeBaseSpec = Field(default_factory=KnowledgeBaseSpec)
    inference_engine: InferenceEngineSpec = Field(default_factory=InferenceEngineSpec)
    agent_coordination: AgentCoordinationSpec = Field(default_factory=AgentCoordinationSpec)


class WriterOutput(BaseModel):
    title: str = Field(...)
    abstract: str = Field(...)
    keywords: list[str] = Field(..., min_length=1)
    introduction: str = Field(...)
    related_work: str = Field(...)
    methodology: str = Field(...)
    mathematical_formulas: list[str] = Field(..., min_length=1)
    experimental_results: str = Field(...)
    discussion: str = Field(...)
    threats_to_validity: list[str] = Field(default_factory=list)
    conclusion: str = Field(...)
    references: list[str] = Field(..., min_length=1)


class ReviewerOutput(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=10.0)
    novelty_score: float = Field(..., ge=0.0, le=10.0)
    rigor_score: float = Field(..., ge=0.0, le=10.0)
    clarity_score: float = Field(..., ge=0.0, le=10.0)
    strengths: list[str] = Field(..., min_length=1)
    weaknesses: list[str] = Field(..., min_length=1)
    revision_suggestions: list[str] = Field(..., min_length=1)
    verdict: Literal["ACCEPT", "MINOR_REVISION", "MAJOR_REVISION", "REJECT"]


class GuardrailOutput(BaseModel):
    status: Literal["passed", "needs_review"]
    issues: list[dict[str, str]] = Field(default_factory=list)
    formula_count: int = Field(default=0, ge=0)
    row_count: int = Field(default=0, ge=0)
    evidence_labels: list[str] = Field(default_factory=list)


class PlannerExperimentStep(BaseModel):
    step_id: str = Field(...)
    name: str = Field(...)
    goal: str = Field(...)
    method: str = Field(...)
    expected_outcome: str = Field(...)


class PlannerResponse(BaseModel):
    status: Literal["ok", "error"] = "ok"
    topic: str = Field(...)
    plan: PlannerOutput = Field(...)


class PipelineRequest(BaseModel):
    topic: str = Field(..., min_length=3)
    domain: str | None = Field(default=None)
    objective: str | None = Field(default=None)
    constraints: str | None = Field(default=None)
    api_key: str | None = Field(default=None)
    require_verified_data: bool = Field(default=False, description="Require verified local or downloaded data before manuscript generation")


class FullPipelineResponse(BaseModel):
    mode_used: Literal["live_gemini", "dynamic_fallback"]
    paper_mode: Literal["live_gemini", "dynamic_fallback"] = "dynamic_fallback"
    plan: PlannerOutput
    research: ResearcherOutput
    paper: WriterOutput
    review: ReviewerOutput
    guardrail: GuardrailOutput
