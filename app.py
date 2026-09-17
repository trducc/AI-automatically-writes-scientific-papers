from __future__ import annotations

import argparse
import base64
import io
import os
import pickle
import random
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    px = None
    go = None
    HAS_PLOTLY = False

from smart_research_lab import ResearchState, latest_artifact, run_workflow

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
RESULTS_ROOT = ROOT / "results" / "nanoGPT"
DATASETS = ("shakespeare_char", "enwik8", "text8")
RUN_LABELS = {
    "run_0": "Baseline",
    "run_1": "Gating (L1 1e-4)",
    "run_2": "Gating (L1 1e-3)",
    "run_3": "Gating (L1 5e-4)",
    "run_4": "Gating (L1 2e-3)",
}
L1_WEIGHTS = {"run_0": 0.0, "run_1": 1e-4, "run_2": 1e-3, "run_3": 5e-4, "run_4": 2e-3}
MODEL_OPTIONS = {
    "sshleifer/tiny-gpt2 (Hugging Face)": "tiny-gpt2",
    "hf-internal-testing/tiny-random-gpt2 (Ultra-light)": "tiny-random-gpt2",
    "Large LLM / External API (GPU Required)": None,
}

st.set_page_config(page_title="Smart Research Lab", page_icon="SR", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] { background: #ffffff; }
    [data-testid="stAppViewContainer"] * { color: #172b4d; }
    [data-testid="stSidebar"] { border-right: 1px solid #d9e2ec; }
    .block-container { max-width: 1440px; padding-top: 1.8rem; padding-bottom: 4rem; }
    .hero { background: linear-gradient(135deg, #102a43, #1f4e5f); color: #ffffff; padding: 2rem 2.3rem; border-radius: 14px; margin-bottom: 1.4rem; }
    .hero h1, .hero h1 * { color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; margin: 0 0 .35rem; font-size: 2.3rem; }
    .hero p { color: #d9f0f2 !important; margin: 0; }
    .badge { display: inline-block; margin-top: 1rem; padding: .35rem .7rem; border: 1px solid #8ed1c7; border-radius: 999px; color: #d9fff5 !important; font-size: .76rem; font-weight: 700; }
    [data-testid="stMetric"] { background: #f5f8fa; border: 1px solid #d9e2ec; border-radius: 10px; padding: .8rem 1rem; }
    [data-testid="stMetric"] label, [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #172b4d !important; }
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, div[data-baseweb="select"] > div, [data-testid="stSelectbox"] [role="combobox"] { background: #ffffff !important; color: #172b4d !important; -webkit-text-fill-color: #172b4d !important; border-color: #b8c5d1 !important; }
    [data-testid="stSelectbox"] [data-baseweb="select"], [data-testid="stSelectbox"] [data-baseweb="select"] > div, [data-testid="stSelectbox"] [data-baseweb="select"] * { background: #ffffff !important; color: #172b4d !important; -webkit-text-fill-color: #172b4d !important; }
    [data-testid="stSelectbox"] svg { fill: #172b4d !important; color: #172b4d !important; }
    div[data-baseweb="popover"], div[data-baseweb="menu"], div[data-baseweb="popover"] *, div[data-baseweb="menu"] *, [role="option"] { background: #ffffff !important; color: #172b4d !important; -webkit-text-fill-color: #172b4d !important; }
    [data-testid="stRadio"] label, [data-testid="stRadio"] label span, [data-testid="stRadio"] label div { color: #172b4d !important; }
    code { background: #eef2f7 !important; color: #52606d !important; border: 1px solid #d9e2ec; }
    </style>
    """,
    unsafe_allow_html=True,
)


def is_demo_default() -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--demo", action="store_true")
    return parser.parse_known_args()[0].demo


def safe_json(path: Path, fallback: Any = None) -> Any:
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError):
        return fallback


def safe_numpy(path: Path) -> dict[str, Any]:
    try:
        loaded = np.load(path, allow_pickle=True)
        return loaded.item() if hasattr(loaded, "item") else {}
    except (OSError, ValueError, TypeError, EOFError):
        return {}


def load_experiment_data() -> dict[str, Any]:
    data: dict[str, Any] = {"artifact": None, "runs": {}, "warnings": []}
    try:
        artifact = latest_artifact()
    except OSError as error:
        data["warnings"].append(f"Could not locate experiment artifact: {error}")
        return data
    if artifact is None:
        data["warnings"].append("No nanoGPT artifact directory was found.")
        return data
    data["artifact"] = artifact
    for index in range(5):
        run_name = f"run_{index}"
        run_dir = artifact / run_name
        try:
            final_info = safe_json(run_dir / "final_info.json", {})
            all_results = safe_numpy(run_dir / "all_results.npy")
            if not final_info and not all_results:
                data["warnings"].append(f"{run_name} is missing usable metrics.")
                continue
            data["runs"][run_name] = {"final_info": final_info or {}, "all_results": all_results or {}, "path": run_dir}
        except (OSError, ValueError, TypeError) as error:
            data["warnings"].append(f"Skipped {run_name}: {error}")
    return data


@st.cache_resource(show_spinner=False)
def load_hf_model(model_name: str):
    """Lazy-load one CPU-only HF model and keep its BPE tokenizer isolated."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_ids = {"tiny-gpt2": "sshleifer/tiny-gpt2", "tiny-random-gpt2": "hf-internal-testing/tiny-random-gpt2"}
    model_id = model_ids[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cpu")
    model.eval()
    return tokenizer, model, model_id


def extract_summary(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, run in data.get("runs", {}).items():
        for dataset, values in (run.get("final_info") or {}).items():
            means = values.get("means", {}) if isinstance(values, dict) else {}
            gate = means.get("mean_head_gate_activation_mean")
            rows.append({
                "Run": RUN_LABELS.get(run_name, run_name),
                "Run ID": run_name,
                "Dataset": dataset,
                "L1 weight": L1_WEIGHTS.get(run_name, 0.0),
                "Final loss": means.get("final_train_loss_mean"),
                "Validation loss": means.get("best_val_loss_mean"),
                "Train time (s)": means.get("total_train_time_mean"),
                "Sparsity ratio": None if gate is None else 1 - float(gate),
            })
    return pd.DataFrame(rows)


def build_dynamics(data: dict[str, Any], dataset: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_name, run in data.get("runs", {}).items():
        for key, series in (run.get("all_results") or {}).items():
            if not key.startswith(f"{dataset}_") or not isinstance(series, list):
                continue
            metric_type = "Validation loss" if "val_info" in key else "Training loss" if "train_info" in key else ""
            if not metric_type:
                continue
            for point in series:
                if isinstance(point, dict) and "iter" in point:
                    loss = point.get("val/loss", point.get("loss"))
                    if loss is not None:
                        rows.append({"Step": point["iter"], "Loss": loss, "Metric": metric_type, "Run": RUN_LABELS.get(run_name, run_name)})
    return pd.DataFrame(rows)


def render_pipeline(state: ResearchState | None) -> None:
    st.subheader("Multi-agent workflow")
    columns = st.columns(4)
    completed = bool(state and state.status == "completed")
    for column, name in zip(columns, ("Planner", "Researcher", "Writer", "Reviewer")):
        if completed or (state and any(log.agent == name for log in state.logs)):
            column.success(f"PASS  {name}")
        else:
            column.info(f"WAIT  {name}")


def render_dashboard(data: dict[str, Any], state: ResearchState | None) -> None:
    st.header("Dashboard & Metrics")
    for warning in data["warnings"]:
        st.warning(warning)
    summary = extract_summary(data)
    if summary.empty:
        st.info("No experiment metrics are available yet.")
        return
    cards = st.columns(4)
    cards[0].metric("Runs loaded", summary["Run ID"].nunique())
    cards[1].metric("Datasets", summary["Dataset"].nunique())
    cards[2].metric("Best validation loss", f"{summary['Validation loss'].min():.4f}")
    cards[3].metric("Median sparsity", f"{summary['Sparsity ratio'].median():.2%}")
    dataset = st.selectbox("Dataset", DATASETS)
    dynamics = build_dynamics(data, dataset)
    st.subheader(f"Interactive training dynamics: {dataset}")
    if not dynamics.empty and HAS_PLOTLY:
        figure = px.line(dynamics, x="Step", y="Loss", color="Run", line_dash="Metric", markers=True, title="Training and validation loss")
        figure.update_layout(template="plotly_white", legend_title_text="Experiment")
        st.plotly_chart(figure, width="stretch")
    elif not dynamics.empty:
        st.line_chart(dynamics.pivot_table(index="Step", columns=["Run", "Metric"], values="Loss"))
    else:
        st.info("Training trace is unavailable for this dataset; summary metrics remain available.")
    st.subheader("Five-run comparison")
    st.dataframe(summary, width="stretch", hide_index=True)
    if HAS_PLOTLY:
        chart = px.bar(summary, x="Run", y="Validation loss", color="Dataset", barmode="group", title="Validation loss by run")
        chart.update_layout(template="plotly_white")
        st.plotly_chart(chart, width="stretch")
    else:
        st.bar_chart(summary.pivot(index="Run", columns="Dataset", values="Validation loss"))
    st.subheader("Attention gate heatmap (measured mean activation converted to sparsity)")
    heat = summary.pivot_table(index="Run", columns="Dataset", values="Sparsity ratio", aggfunc="mean").fillna(0)
    if HAS_PLOTLY:
        heatmap = go.Figure(go.Heatmap(z=heat.to_numpy(), x=list(heat.columns), y=list(heat.index), colorscale="Blues", zmin=0, zmax=1, text=np.round(heat.to_numpy(), 4), texttemplate="%{text}"))
        heatmap.update_layout(template="plotly_white", xaxis_title="Dataset", yaxis_title="Regularization run")
        st.plotly_chart(heatmap, width="stretch")
    else:
        st.dataframe(heat, width="stretch")


def render_sandbox() -> None:
    st.header("Text Generation Sandbox")
    st.caption("Hugging Face BPE tokenizers are isolated from the nanoGPT character-level path.")
    selected = st.selectbox("Model", list(MODEL_OPTIONS))
    if MODEL_OPTIONS[selected] is None:
        st.info("Hệ thống đã thiết kế interface sẵn sàng cắm LLM lớn (Llama, GPT-4) khi deploy trên server GPU.")
        return
    prompt = st.text_area("Prompt", "Research topic: Dynamic Attention Head Gating\nAbstract:", height=120)
    left, middle, right = st.columns(3)
    temperature = left.slider("Temperature", 0.1, 1.2, 0.8, 0.1)
    max_tokens = middle.slider("Max new tokens", 20, 150, 80)
    top_k = right.slider("Top-K", 1, 100, 40)
    top_p = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05)
    if st.button("Generate text", type="primary", width="stretch"):
        try:
            from pretrained_reference import generate_reference
            started = time.perf_counter()
            result = generate_reference(prompt, MODEL_OPTIONS[selected], max_tokens, temperature, top_k, top_p)
            latency_ms = (time.perf_counter() - started) * 1000
            tokenizer, _, _ = load_hf_model(MODEL_OPTIONS[selected])
            token_count = len(tokenizer(result["text"], add_special_tokens=False)["input_ids"])
            st.session_state["sandbox_result"] = result | {"latency_ms": latency_ms, "token_count": token_count, "tokens_per_second": token_count / max(latency_ms / 1000, 1e-9)}
        except Exception as error:
            st.error(f"Generation failed safely: {error}")
    result = st.session_state.get("sandbox_result")
    if result:
        st.code(result["text"], language="text")
        c1, c2, c3 = st.columns(3)
        c1.metric("Inference latency", f"{result['latency_ms']:.1f} ms")
        c2.metric("Generated tokens", result["token_count"])
        c3.metric("CPU tokens / second", f"{result['tokens_per_second']:.1f}")


def render_architecture() -> None:
    st.header("Architecture & Implementation")
    st.write("Dynamic Head Gating learns a sigmoid gate per attention head and adds an L1 penalty to encourage capacity allocation.")
    st.latex(r"L = L_{CE} + \\lambda \\frac{1}{LH}\\sum_{l=1}^{L}\\sum_{h=1}^{H}\\sigma(g_{l,h})")
    st.code("""# Dynamic gate inside CausalSelfAttention\nself.head_gate = nn.Parameter(torch.ones(config.n_head) * 3.0)\ngates = torch.sigmoid(self.head_gate).view(1, self.n_head, 1, 1)\ny = attention_output * gates\n\n# L1 sparsity regularization inside GPT.forward\ngate_l1 = sum(torch.sigmoid(block.attn.head_gate).sum()\n              for block in self.transformer.h)\nloss = cross_entropy + l1_lambda * gate_l1 / total_heads""", language="python")
    st.info("The nanoGPT path remains character-level. Hugging Face sandbox models use independent BPE tokenizers.")


def dataset_metadata(dataset: str) -> dict[str, Any]:
    directory = DATA_ROOT / dataset
    result: dict[str, Any] = {"Dataset": dataset, "Status": "Unavailable"}
    try:
        train = directory / "train.bin"
        val = directory / "val.bin"
        meta: dict[str, Any] = {}
        if (directory / "meta.pkl").exists():
            with (directory / "meta.pkl").open("rb") as handle:
                meta = pickle.load(handle)
        result.update({"Train size (MB)": round(train.stat().st_size / 1_000_000, 3), "Validation size (MB)": round(val.stat().st_size / 1_000_000, 3), "Vocab size": meta.get("vocab_size", "unknown"), "Status": "Ready"})
    except (OSError, ValueError, TypeError, EOFError, KeyError) as error:
        result["Error"] = str(error)
    return result


def render_dataset_explorer() -> None:
    st.header("Dataset Explorer")
    dataset = st.selectbox("Dataset", DATASETS, key="explorer_dataset")
    st.dataframe(pd.DataFrame([dataset_metadata(name) for name in DATASETS]), width="stretch", hide_index=True)
    split = st.radio("Split", ("train", "val"), horizontal=True)
    if st.button("Load random sample", width="stretch"):
        path = DATA_ROOT / dataset / f"{split}.bin"
        try:
            raw = np.memmap(path, dtype=np.uint16, mode="r")
            start = random.randint(0, max(0, len(raw) - 500))
            with (DATA_ROOT / dataset / "meta.pkl").open("rb") as handle:
                meta = pickle.load(handle)
            text = "".join(meta["itos"][int(value)] for value in raw[start:start + 500])
            st.code(text, language="text")
        except (OSError, ValueError, TypeError, EOFError, KeyError) as error:
            st.warning(f"Sample unavailable: {error}")


def render_report_artifacts(data: dict[str, Any], state: ResearchState | None) -> None:
    st.header("Report & Artifacts")
    artifact = data.get("artifact")
    if artifact is None:
        st.warning("No artifact directory is available.")
        return
    try:
        files = [path for path in artifact.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".npy", ".png", ".txt", ".md", ".pdf", ".pt"}]
    except OSError as error:
        st.warning(f"Could not enumerate artifacts: {error}")
        files = []
    pdf_path = Path(state.pdf_path) if state and state.pdf_path else artifact / "latex" / "template.pdf"
    if pdf_path.exists():
        encoded = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
        st.subheader("Report viewer")
        st.markdown(f'<iframe src="data:application/pdf;base64,{encoded}" width="100%" height="720"></iframe>', unsafe_allow_html=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            try:
                archive.write(path, path.relative_to(artifact))
            except (OSError, ValueError):
                continue
    st.download_button("Download complete artifact bundle", buffer.getvalue(), "smart_research_artifacts.zip", "application/zip", width="stretch")
    st.caption(f"Bundle includes {len(files)} available files. Checkpoints are included when present; this experiment may only contain metrics and plots.")


def render_workflow(state: ResearchState | None, topic: str, demo: bool) -> ResearchState | None:
    if st.button("Generate demo report", type="primary", width="stretch"):
        with st.status("Running shared-state workflow...", expanded=True) as status:
            try:
                state = run_workflow(topic, demo=demo)
                st.session_state["research_state"] = state.model_dump(mode="json")
                status.update(label="Workflow completed", state="complete")
            except Exception as error:
                st.error(f"Workflow failed safely: {error}")
    saved = st.session_state.get("research_state")
    return ResearchState.model_validate(saved) if saved else state


def main() -> None:
    st.markdown('<div class="hero"><h1>Smart Research Lab</h1><p>Professional research workflow for experiments, evidence, generation, and review.</p><span class="badge">CPU-SAFE · MODULAR · FALLBACK READY</span></div>', unsafe_allow_html=True)
    with st.sidebar:
        st.header("Research control room")
        topic = st.text_area("Research topic", "Dynamic Attention Head Gating: Soft Head Pruning and Capacity Allocation in Transformer Language Models", height=110)
        demo = st.toggle("Demo mode", value=is_demo_default())
        st.caption("Precomputed experiment artifacts are used in demo mode.")
        page = st.radio("Navigate", ("Dashboard & Metrics", "Text Generation Sandbox", "Architecture & Implementation", "Dataset Explorer", "Report & Artifacts"))
        artifact = latest_artifact()
        if artifact:
            st.caption(f"Artifact: {artifact.name}")
    state = ResearchState.model_validate(st.session_state["research_state"]) if st.session_state.get("research_state") else None
    if page == "Dashboard & Metrics":
        data = load_experiment_data()
        state = render_workflow(state, topic, demo)
        render_pipeline(state)
        render_dashboard(data, state)
    elif page == "Text Generation Sandbox":
        render_sandbox()
    elif page == "Architecture & Implementation":
        render_architecture()
    elif page == "Dataset Explorer":
        render_dataset_explorer()
    else:
        render_report_artifacts(load_experiment_data(), state)


if __name__ == "__main__":
    main()
