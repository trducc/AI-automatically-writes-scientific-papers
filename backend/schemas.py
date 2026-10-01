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


class ModelSpec(BaseModel):
    name: str = Field(..., description="Model name")
    role: Literal["Baseline", "Proposed"] = Field(..., description="Baseline or proposed model")
    backbone: str = Field(..., description="Backbone or architecture family")
    params_m: float = Field(..., ge=0.0, description="Parameter count in millions")
    flops_g: float = Field(..., ge=0.0, description="Floating point operations in GFLOPs")
    latency_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds")
    key_characteristic: str = Field(..., description="Technical characteristic")


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


class ResearcherOutput(BaseModel):
    hardware_setup: str = Field(..., description="Hardware configuration summary")
    datasets_used: list[DatasetSpec] = Field(..., min_length=1)
    models_compared: list[ModelSpec] = Field(..., min_length=1)
    training_curves: list[MetricCurvePoint] = Field(..., min_length=1)
    benchmark_table: list[AblationRow] = Field(..., min_length=1)
    key_findings: list[str] = Field(..., min_length=1)


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


class FullPipelineResponse(BaseModel):
    mode_used: Literal["live_gemini", "dynamic_fallback"]
    plan: PlannerOutput
    research: ResearcherOutput
    paper: WriterOutput
    review: ReviewerOutput
