from __future__ import annotations

import json
from html import escape
from typing import Any

import pandas as pd
import requests
import streamlit as st

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    px = None
    HAS_PLOTLY = False

st.set_page_config(page_title="Smart Research Lab", page_icon="SRL", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root{--ink:#e8eefc;--muted:#93a2bf;--panel:rgba(17,27,48,.8);--line:rgba(135,160,202,.24);--cyan:#50d9e9}
[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:radial-gradient(circle at 82% 0%,#173b60 0,#0b1220 43%,#070b14 100%)}
[data-testid="stAppViewContainer"] *,[data-testid="stSidebar"] *{color:var(--ink)}
[data-testid="stSidebar"]{background:rgba(7,13,25,.96);border-right:1px solid var(--line)}
.block-container{max-width:1540px;padding-top:1.25rem;padding-bottom:4rem}.hero{background:linear-gradient(115deg,rgba(24,57,93,.96),rgba(15,27,50,.78));border:1px solid rgba(80,217,233,.32);border-radius:20px;padding:2rem 2.3rem;margin-bottom:1.1rem;box-shadow:0 20px 55px #0004}.hero h1{color:#f8fbff!important;margin:0 0 .35rem;font-size:2.35rem}.hero p{color:#b9c8e2!important;margin:0}.badge,.domain-badge{display:inline-block;padding:.3rem .68rem;border-radius:999px;font-size:.75rem;font-weight:750}.domain-badge{background:var(--cyan);color:#06131d!important;margin-top:.9rem}.glass-card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:1.1rem;min-height:185px;box-shadow:0 14px 32px #0003}.glass-card h4{margin:0 0 .45rem;color:#f8fbff!important}.glass-card p,.glass-card small{color:var(--muted)!important}.callout{background:#50d9e914;border:1px solid #50d9e947;border-left:4px solid var(--cyan);border-radius:12px;padding:1rem 1.2rem;margin:.5rem 0 1rem}[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:.85rem 1rem}[data-testid="stMetric"] label{color:var(--muted)!important}[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#f8fbff!important}[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input,div[data-baseweb="select"]>div,[role="option"]{background:#101a2d!important;color:var(--ink)!important;border-color:var(--line)!important}.stTabs [data-baseweb="tab-list"]{gap:.4rem;background:#0a111f8f;padding:.35rem;border-radius:13px}.stTabs [data-baseweb="tab"]{border-radius:9px;color:var(--muted)!important}.stTabs [aria-selected="true"]{background:#50d9e929;color:#e8ffff!important}.stButton>button[kind="primary"]{background:linear-gradient(110deg,#16a6c0,#3ccf9a);color:#04131c!important;border:0;font-weight:800;min-height:3rem}
</style>""", unsafe_allow_html=True)

PRESETS = {
    "Dynamic Attention Head Gating for Efficient Transformer Inference": {"domain":"NLP / Efficiency","objective":"Improve inference efficiency while preserving language modeling quality","constraints":"CPU-friendly evaluation, transparent ablations, reproducible metrics"},
    "Vision Transformer with Multi-Scale Patch Embedding for CIFAR-10 & Medical Imaging": {"domain":"Computer Vision","objective":"Improve visual representation quality across natural and medical images","constraints":"Compare multi-scale features, report latency and robustness"},
    "Spatio-Temporal Graph Neural Networks for Real-Time Financial Fraud Detection": {"domain":"Graph Learning / FinTech","objective":"Detect suspicious transaction patterns with high recall and low latency","constraints":"Imbalanced labels, temporal split, interpretable decisions"},
    "Contrastive Multimodal Learning for Automated Radiology Report Generation": {"domain":"Healthcare AI","objective":"Align radiology images and clinical language for accurate report generation","constraints":"Cross-modal alignment, clinical safety, external validation"},
}
CUSTOM = "Custom Topic (Tự nhập đề tài bất kỳ...)"


def apply_preset() -> None:
    choice = st.session_state.get("preset", CUSTOM)
    if choice in PRESETS:
        for key, value in {"topic": choice, **PRESETS[choice]}.items():
            st.session_state[key] = value


def run_pipeline(topic: str, domain: str, objective: str, constraints: str) -> dict[str, Any] | None:
    payload = {"topic": topic, "domain": domain, "objective": objective, "constraints": constraints}
    try:
        response = requests.post("http://localhost:8000/api/pipeline/run", json=payload, timeout=60)
        if response.ok:
            return response.json()
    except requests.RequestException:
        pass
    try:
        from backend.main import _run_pipeline_request
        from backend.schemas import PipelineRequest
        return _run_pipeline_request(PipelineRequest(**payload)).model_dump(mode="json")
    except Exception:
        return None


def paper_markdown(paper: dict[str, Any], research: dict[str, Any]) -> str:
    table = pd.DataFrame(research.get("benchmark_table", [])).to_markdown(index=False) if research.get("benchmark_table") else "No benchmark rows available."
    sections = [("Abstract", paper["abstract"]),("Keywords", ", ".join(paper["keywords"])),("1. Introduction", paper["introduction"]),("2. Related Work", paper["related_work"]),("3. Proposed Methodology", paper["methodology"]),("4. Experimental Setup & Results", paper["experimental_results"]),("5. Discussion", paper["discussion"]),("6. Conclusion", paper["conclusion"])]
    refs = "\n".join(f"{i}. {item}" for i, item in enumerate(paper["references"], 1))
    return "# " + paper["title"] + "\n\n" + "\n\n".join(f"## {title}\n{body}" for title, body in sections) + f"\n\n{table}\n\n## References\n{refs}\n"


def render_header(result: dict[str, Any]) -> None:
    plan, research, review = result["plan"], result["research"], result["review"]
    proposed = next(item for item in research["models_compared"] if item["role"] == "Proposed")
    baseline_rows = [item for item in research["benchmark_table"] if item["role"] == "Baseline"]
    proposed_row = next(item for item in research["benchmark_table"] if item["role"] == "Proposed")
    best = max((item["primary_metric"] for item in baseline_rows), default=0.0)
    gain = (proposed_row["primary_metric"] - best) / max(best, 1e-9) * 100
    st.markdown(f"<section class='hero'><h1>{escape(plan['title'])}</h1><p>Enterprise AI Research Lab · Multi-Agent Research Pipeline</p><span class='domain-badge'>{escape(plan['domain'])}</span></section>", unsafe_allow_html=True)
    cards = st.columns(4)
    cards[0].metric("Primary Benchmark Score", f"{proposed_row['primary_metric']:.2f}", f"+{gain:.1f}% vs best baseline")
    cards[1].metric("Efficiency / Latency", f"{proposed['latency_ms']:.1f} ms", f"{proposed['params_m']:.1f}M params")
    cards[2].metric("Experimental Scope", f"{len(research['datasets_used'])} datasets", f"{len(research['models_compared'])} models")
    cards[3].metric("Automated Peer Review", f"{review['overall_score']:.1f} / 10", review["verdict"])


def render_plan(plan: dict[str, Any]) -> None:
    st.markdown(f"<div class='callout'><strong>Research Question</strong><br>{escape(plan['research_question'])}<br><br><strong>Core Hypothesis</strong><br>{escape(plan['hypothesis'])}</div>", unsafe_allow_html=True)
    st.subheader("📦 Datasets Catalog")
    columns = st.columns(3)
    for index, dataset in enumerate(plan["datasets_catalog"]):
        with columns[index % 3]:
            st.markdown(f"<article class='glass-card'><h4>{escape(dataset['name'])}</h4><span class='badge'>{escape(dataset['modality'])}</span><p><b>Scale:</b> {escape(dataset['sample_size'])}<br><b>Input:</b> {escape(dataset['input_spec'])}<br><b>Metric:</b> {escape(dataset['primary_metric'])}</p><small>{escape(dataset['description'])}</small></article>", unsafe_allow_html=True)
    st.subheader("🧠 Model Zoo & Architecture Registry")
    models = pd.DataFrame(plan["models_catalog"])
    st.dataframe(models[["name","role","backbone","params_m","flops_g","latency_ms","key_characteristic"]], use_container_width=True, hide_index=True)
    if HAS_PLOTLY:
        chart = models.melt(id_vars="name", value_vars=["params_m","latency_ms"], var_name="Measure", value_name="Value")
        st.plotly_chart(px.bar(chart, x="name", y="Value", color="Measure", barmode="group", template="plotly_dark", title="Parameters vs latency"), use_container_width=True)


def render_benchmarks(research: dict[str, Any]) -> None:
    selected = st.selectbox("Dataset focus", [item["name"] for item in research["datasets_used"]])
    curves = pd.DataFrame(research["training_curves"])
    curves = curves[curves["dataset_name"] == selected]
    left, right = st.columns(2)
    if HAS_PLOTLY:
        loss = curves.melt(id_vars="epoch", value_vars=["baseline_loss","proposed_loss"], var_name="Series", value_name="Loss")
        score = curves.melt(id_vars="epoch", value_vars=["baseline_score","proposed_score"], var_name="Series", value_name="Primary Metric")
        left.plotly_chart(px.line(loss, x="epoch", y="Loss", color="Series", markers=True, template="plotly_dark", title="Training / validation loss"), use_container_width=True)
        right.plotly_chart(px.line(score, x="epoch", y="Primary Metric", color="Series", markers=True, template="plotly_dark", title="Performance growth"), use_container_width=True)
    st.subheader("Benchmark & Ablation Matrix")
    st.dataframe(pd.DataFrame(research["benchmark_table"]), use_container_width=True, hide_index=True)
    st.subheader("Key Empirical Findings")
    for finding in research["key_findings"]:
        st.markdown(f"- {finding}")


def render_method(plan: dict[str, Any], paper: dict[str, Any]) -> None:
    st.subheader("Architecture Flow")
    st.code("Input data → Domain encoder → Task-aware signal selection → Proposed model → Prediction + uncertainty", language="text")
    st.write(paper["methodology"])
    st.subheader("Mathematical Formulation")
    for formula in paper["mathematical_formulas"]:
        st.latex(formula)
    st.subheader(f"PyTorch Core: {plan['proposed_method']}")
    st.code("""class TaskAwareBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x):
        weights = torch.sigmoid(self.gate(self.norm(x)))
        return x + weights * self.projection(self.norm(x))
""", language="python")


def render_paper(paper: dict[str, Any], research: dict[str, Any]) -> None:
    markdown = paper_markdown(paper, research)
    st.title(paper["title"])
    st.caption("Full Academic Manuscript · IEEE/NeurIPS-inspired structure")
    sections = [("Abstract", paper["abstract"]),("Keywords", ", ".join(paper["keywords"])),("1. Introduction", paper["introduction"]),("2. Related Work", paper["related_work"]),("3. Proposed Methodology", paper["methodology"]),("4. Experimental Setup & Results", paper["experimental_results"]),("5. Discussion", paper["discussion"]),("6. Conclusion", paper["conclusion"])]
    for title, body in sections:
        st.subheader(title)
        st.write(body)
    st.subheader("References")
    for index, reference in enumerate(paper["references"], 1):
        st.markdown(f"{index}. {reference}")
    html = "<html><body style='font-family:Georgia;max-width:900px;margin:40px auto;line-height:1.65'><h1>" + escape(paper["title"]) + "</h1>" + "".join(f"<h2>{escape(title)}</h2><p>{escape(body)}</p>" for title, body in sections) + "</body></html>"
    downloads = st.columns(3)
    downloads[0].download_button("📥 Download Paper (.MD)", markdown, "research_paper.md", "text/markdown", use_container_width=True)
    downloads[1].download_button("📥 Download Structured Data (.JSON)", json.dumps({"paper":paper,"research":research}, indent=2), "research_result.json", "application/json", use_container_width=True)
    downloads[2].download_button("🖨️ Download Printable Report (.HTML)", html, "research_report.html", "text/html", use_container_width=True)


def render_review(review: dict[str, Any]) -> None:
    metrics = st.columns(4)
    for column, label, key in zip(metrics, ("Novelty","Technical Rigor","Clarity","Overall Score"), ("novelty_score","rigor_score","clarity_score","overall_score")):
        column.metric(label, f"{review[key]:.1f} / 10")
    st.markdown(f"<div class='callout'><strong>Verdict: {review['verdict']}</strong></div>", unsafe_allow_html=True)
    columns = st.columns(3)
    for column, title, key in zip(columns, ("✅ Strengths","⚠️ Weaknesses","🛠️ Actionable Revisions"), ("strengths","weaknesses","revision_suggestions")):
        with column:
            st.subheader(title)
            for item in review[key]:
                st.markdown(f"- {item}")


def main() -> None:
    first = next(iter(PRESETS))
    for key, value in {"topic":first, **PRESETS[first]}.items():
        st.session_state.setdefault(key, value)
    with st.sidebar:
        st.markdown("## Research Control Room")
        st.selectbox("⚡ Quick Demo Presets", [*PRESETS, CUSTOM], key="preset", on_change=apply_preset)
        st.text_area("Research Topic", key="topic", height=90)
        st.text_input("Domain", key="domain")
        st.text_input("Primary Objective", key="objective")
        st.text_area("Constraints", key="constraints", height=75)
        run_clicked = st.button("🚀 Run Autonomous Research Pipeline", type="primary", use_container_width=True)
    if run_clicked or "full_pipeline_result" not in st.session_state:
        with st.status("🤖 Orchestrating Multi-Agent Research Pipeline...", expanded=True) as status:
            for agent in ("Planner", "Researcher", "Writer", "Reviewer"):
                st.write(f"{agent} Agent · processing structured research state")
            result = run_pipeline(st.session_state.topic, st.session_state.domain, st.session_state.objective, st.session_state.constraints)
            if not result:
                status.update(label="Pipeline unavailable", state="error")
                st.stop()
            st.session_state.full_pipeline_result = result
            status.update(label="✅ Pipeline completed · manuscript ready", state="complete", expanded=False)
    result = st.session_state["full_pipeline_result"]
    render_header(result)
    plan, research, paper, review = result["plan"], result["research"], result["paper"], result["review"]
    tabs = st.tabs(["🎯 Research Plan & Catalogs","📊 Benchmark & Ablation Charts","📐 Architecture & Mathematical Formulation","📄 Full Academic Paper","⚖️ Peer Review & Rubric"])
    with tabs[0]: render_plan(plan)
    with tabs[1]: render_benchmarks(research)
    with tabs[2]: render_method(plan, paper)
    with tabs[3]: render_paper(paper, research)
    with tabs[4]: render_review(review)


if __name__ == "__main__":
    main()
