from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from smart_research_lab import ResearchState, latest_artifact, run_workflow


st.set_page_config(
    page_title="Smart Research Lab",
    page_icon="SR",
    layout="wide",
    initial_sidebar_state="expanded",
)


STATUS_LABELS = {
    "idle": "Ready",
    "planning": "Planner",
    "researching": "Researcher",
    "writing": "Writer",
    "reviewing": "Reviewer",
    "completed": "Completed",
    "needs_revision": "Needs revision",
}


def is_demo_default() -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--demo", action="store_true")
    args, _ = parser.parse_known_args()
    return args.demo


def render_pipeline(state: ResearchState | None) -> None:
    st.subheader("Multi-agent workflow")
    names = ["Planner", "Researcher", "Writer", "Reviewer"]
    active = state.active_agent if state else ""
    completed = bool(state and state.status == "completed")
    columns = st.columns(4)
    for column, name in zip(columns, names):
        if completed:
            marker = "PASS"
            tone = "success"
        elif active == name:
            marker = "RUN"
            tone = "warning"
        elif state and any(log.agent == name for log in state.logs):
            marker = "DONE"
            tone = "success"
        else:
            marker = "WAIT"
            tone = "info"
        with column:
            getattr(st, tone)(f"{marker}  {name}")


def extract_metrics(state: ResearchState) -> pd.DataFrame:
    rows: list[dict] = []
    runs = state.experiment_results.get("runs", {})
    for run_name, payload in runs.items():
        for dataset, values in payload.items():
            means = values.get("means", {})
            rows.append(
                {
                    "Run": "Baseline" if run_name == "run_0" else "Gated variant",
                    "Dataset": dataset,
                    "Best validation loss": means.get("best_val_loss_mean"),
                    "Train loss": means.get("final_train_loss_mean"),
                    "Gate activation": means.get("mean_head_gate_activation_mean"),
                    "Time (s)": means.get("total_train_time_mean"),
                }
            )
    return pd.DataFrame(rows)


def render_results(state: ResearchState) -> None:
    if not state.experiment_results:
        st.info("No experiment artifact loaded yet.")
        return
    st.subheader("Experiment results")
    metrics = extract_metrics(state)
    if not metrics.empty:
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        chart_data = metrics.pivot(index="Dataset", columns="Run", values="Best validation loss")
        st.bar_chart(chart_data)
    artifact_dir = Path(state.artifact_dir) if state.artifact_dir else None
    if artifact_dir:
        image = artifact_dir / "val_loss_shakespeare_char.png"
        if image.exists():
            st.image(str(image), caption="Validation loss: Shakespeare character dataset", use_container_width=True)
    pdf = Path(state.pdf_path)
    if pdf.exists():
        pdf_bytes = pdf.read_bytes()
        encoded_pdf = base64.b64encode(pdf_bytes).decode("ascii")
        st.subheader("Generated paper")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{encoded_pdf}" width="100%" height="720" style="border:1px solid #444;border-radius:8px"></iframe>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<a href="data:application/pdf;base64,{encoded_pdf}" target="_blank" rel="noopener">Open PDF in Chrome</a>',
            unsafe_allow_html=True,
        )
        download_col, draft_col = st.columns(2)
        with download_col:
            st.download_button(
                "Download research paper PDF",
                data=pdf_bytes,
                file_name="smart_research_lab_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with draft_col:
            st.download_button(
                "Download draft (.md)",
                data=state.draft,
                file_name="smart_research_lab_draft.md",
                mime="text/markdown",
                use_container_width=True,
            )


def render_review(state: ResearchState) -> None:
    if state.review is None:
        return
    st.subheader("Reviewer decision")
    score_columns = st.columns(5)
    scores = {
        "Faithfulness": state.review.faithfulness,
        "Relevance": state.review.relevance,
        "Task completion": state.review.task_completion,
        "Reproducibility": state.review.reproducibility,
        "Overall": state.review.overall,
    }
    for column, (label, value) in zip(score_columns, scores.items()):
        column.metric(label, f"{value:.2f}")
    if state.review.decision == "PASS":
        st.success("PASS - artifact is complete enough for demo review.")
    else:
        st.error("FAIL - revision is required.")
    with st.expander("Reviewer feedback"):
        for item in state.review.feedback:
            st.write(f"- {item}")


def main() -> None:
    st.title("Smart Research Lab")
    st.caption("Planner -> Researcher -> Writer -> Reviewer | Shared-state research workflow")

    with st.sidebar:
        st.header("Research setup")
        topic = st.text_area(
            "Topic",
            value="Dynamic attention-head gating for efficient character-level language models",
            height=100,
        )
        demo = st.toggle("Demo mode", value=is_demo_default(), help="Reads the saved nanoGPT artifact and skips API calls and training.")
        max_papers = st.slider("Maximum evidence items", 3, 12, 4)
        st.caption(f"Evidence budget: {max_papers} items")
        start = st.button("Generate paper", type="primary", use_container_width=True)
        artifact = latest_artifact()
        if artifact:
            st.caption(f"Artifact: {artifact.name}")

    if start:
        with st.status("Running shared-state workflow...", expanded=True) as status:
            state = run_workflow(topic, demo=demo)
            st.session_state["research_state"] = state.model_dump(mode="json")
            status.update(label=f"Workflow {STATUS_LABELS.get(state.status, state.status)}", state="complete")

    saved = st.session_state.get("research_state")
    state = ResearchState.model_validate(saved) if saved else None
    render_pipeline(state)

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Execution log")
        if state and state.logs:
            for log in state.logs:
                st.write(f"`{log.timestamp}` **{log.agent}**: {log.message}")
        else:
            st.info("Start a run to see agent activity.")
        if state:
            with st.expander("Shared state JSON"):
                st.json(state.model_dump(mode="json"))
    with right:
        if state:
            render_review(state)
        else:
            st.info("Reviewer scores will appear here.")

    if state:
        render_results(state)
        st.subheader("Draft preview")
        st.markdown(state.draft)


if __name__ == "__main__":
    main()
