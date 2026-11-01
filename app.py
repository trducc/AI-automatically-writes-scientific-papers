from __future__ import annotations

import base64
import json
import re
import tempfile
import time
import unicodedata
import zipfile
from datetime import datetime
from io import BytesIO
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from backend.guardrail_agent import audit_manuscript
from chart_generator import generate_research_charts
from academic_sanitizer import sanitize_academic_topic

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    px = None
    HAS_PLOTLY = False

st.set_page_config(page_title="Phòng thí nghiệm nghiên cứu AI", page_icon="AI", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root{--ink:#e8eefc;--muted:#93a2bf;--panel:rgba(17,27,48,.8);--line:rgba(135,160,202,.24);--cyan:#50d9e9}
[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:radial-gradient(circle at 82% 0%,#173b60 0,#0b1220 43%,#070b14 100%)}
[data-testid="stAppViewContainer"] *,[data-testid="stSidebar"] *{color:var(--ink)}
[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input,[data-testid="stSelectbox"] [data-baseweb="select"]>div,div[data-baseweb="select"]>div,[role="option"]{background:#101a2d!important;color:var(--ink)!important;border:1px solid var(--line)!important;box-shadow:none!important}
[data-testid="stSidebar"] div[data-baseweb="select"],[data-testid="stSidebar"] div[data-baseweb="select"]>div{background-color:#101a2d!important;color:#e8eefc!important;border-color:var(--line)!important}
[data-testid="stSidebar"] div[data-baseweb="select"] span,[data-testid="stSidebar"] div[data-baseweb="select"] input{color:#e8eefc!important;-webkit-text-fill-color:#e8eefc!important}
[data-testid="stSidebar"] div[data-baseweb="select"] svg{fill:#e8eefc!important;color:#e8eefc!important}
[data-testid="stSidebar"] [role="option"]{background-color:#101a2d!important;color:#e8eefc!important}
[data-testid="stSidebar"] input:not([type="checkbox"]),[data-testid="stSidebar"] textarea{background-color:rgba(15,23,42,.92)!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important;border-color:var(--line)!important}
[data-testid="stSidebar"] input:not([type="checkbox"])::placeholder,[data-testid="stSidebar"] textarea::placeholder{color:#93a2bf!important;-webkit-text-fill-color:#93a2bf!important}
[data-testid="stSidebar"] .stTextInput label,[data-testid="stSidebar"] .stTextArea label,[data-testid="stSidebar"] .stSelectbox label{color:#f8fafc!important;font-weight:500!important}
[data-testid="stSidebar"] .stTextInput div:has(> input),[data-testid="stSidebar"] .stTextArea div:has(> textarea){background-color:rgba(15,23,42,.92)!important}
[data-testid="stSidebar"] .stSelectbox div:has(> input[role="combobox"]){background-color:#101a2d!important}
.stSelectbox .react-aria-ComboBox>div,.stSelectbox div:has(> input[role="combobox"]){background-color:#101a2d!important;color:#e8eefc!important;border:1px solid var(--line)!important;border-radius:10px!important}
.stSelectbox input[role="combobox"]{background-color:transparent!important;color:#ffffff!important;-webkit-text-fill-color:#ffffff!important}
.stSelectbox [role="option"]{background-color:#101a2d!important;color:#e8eefc!important}
div[data-testid="stCodeBlock"],div[data-testid="stCodeBlock"]>div,div[data-testid="stCodeBlock"] pre,div[data-testid="stCodeBlock"] code,.stCodeBlock,.stCodeBlock>div,.stCodeBlock pre,.stCodeBlock code{background:#101a2d!important;color:#e8eefc!important;-webkit-text-fill-color:#e8eefc!important;border-color:var(--line)!important}
div[data-testid="stCodeBlock"] pre *,div[data-testid="stCodeBlock"] code *,.stCodeBlock pre *,.stCodeBlock code *{background:transparent!important;color:#e8eefc!important;-webkit-text-fill-color:#e8eefc!important}
.stTabs [role="tablist"]{background:#0a111f!important;overflow-x:auto!important;scrollbar-color:#50d9e9 #101a2d;white-space:nowrap}
.stTabs [role="tab"]{background:#101a2d!important;color:#cbd5e1!important;white-space:nowrap!important}
.stTabs [role="tab"][aria-selected="true"]{background:#173b60!important;color:#ffffff!important;border-bottom:2px solid var(--cyan)!important}
.stSelectbox label,.stTextInput label,.stTextArea label{color:var(--ink)!important}
.stDownloadButton>button,.stButton>button{background:#101a2d!important;color:var(--ink)!important;border:1px solid var(--line)!important;border-radius:10px;font-weight:700;white-space:normal;line-height:1.2;min-height:3rem}
.stDownloadButton>button:hover,.stButton>button:hover{background:#17304d!important;border-color:var(--cyan)!important;color:#f8fbff!important}
[data-testid="stSidebar"]{background:rgba(7,13,25,.96);border-right:1px solid var(--line)}
.block-container{width:min(100% - 2rem,1540px);max-width:1540px;padding-top:1.25rem;padding-bottom:4rem}.hero{background:linear-gradient(115deg,rgba(24,57,93,.96),rgba(15,27,50,.78));border:1px solid rgba(80,217,233,.32);border-radius:20px;padding:clamp(1.2rem,3vw,2rem) clamp(1.1rem,3vw,2.3rem);margin-bottom:1.1rem;box-shadow:0 20px 55px #0004}.hero h1{color:#f8fbff!important;margin:0 0 .55rem;font-size:clamp(1.55rem,3.4vw,2.35rem);line-height:1.12;overflow-wrap:anywhere;word-break:normal}.hero p{color:#b9c8e2!important;margin:0;line-height:1.45}.badge,.domain-badge{display:inline-block;padding:.3rem .68rem;border-radius:999px;font-size:.75rem;font-weight:750}.domain-badge{background:var(--cyan);color:#06131d!important;margin-top:.9rem}.glass-card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:1.1rem;min-height:185px;box-shadow:0 14px 32px #0003;overflow-wrap:anywhere}.glass-card h4{margin:0 0 .45rem;color:#f8fbff!important}.glass-card p,.glass-card small{color:var(--muted)!important}.callout{background:#50d9e914;border:1px solid #50d9e947;border-left:4px solid var(--cyan);border-radius:12px;padding:1rem 1.2rem;margin:.5rem 0 1rem;overflow-wrap:anywhere}[data-testid="stMetric"]{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:.85rem 1rem;min-width:0}[data-testid="stMetric"] label{color:var(--muted)!important}[data-testid="stMetric"] [data-testid="stMetricValue"]{color:#f8fbff!important}[data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input,div[data-baseweb="select"]>div,[role="option"]{background:#101a2d!important;color:var(--ink)!important;border-color:var(--line)!important}.stTabs [data-baseweb="tab-list"]{gap:.4rem;background:#0a111f8f;padding:.35rem;border-radius:13px;overflow-x:auto;scrollbar-width:thin}.stTabs [data-baseweb="tab"]{border-radius:9px;color:var(--muted)!important;white-space:normal!important;text-align:center;min-height:2.5rem}.stTabs [aria-selected="true"]{background:#50d9e929;color:#e8ffff!important}.stButton>button[kind="primary"]{background:linear-gradient(110deg,#16a6c0,#3ccf9a);color:#04131c!important;border:0;font-weight:800;min-height:3rem}
.pdf-open-link{display:flex;align-items:center;justify-content:center;min-height:3rem;padding:.5rem 1rem;border-radius:.5rem;background:#d95757;color:#fff!important;font-weight:800;text-decoration:none!important;text-align:center}
.pdf-open-link:hover{background:#ee6c6c;color:#fff!important}
.stDataFrame,.stTable{width:100%;max-width:100%;overflow-x:auto;border-radius:12px}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
[data-testid="stImage"],[data-testid="stImage"]>div,[data-testid="stImage"] img{width:100%!important;max-width:100%!important;background:transparent!important;border:0!important;border-radius:0!important;box-shadow:none!important;padding:0!important}
[data-testid="stImage"]{margin:0 0 1rem!important}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:rgba(17,27,48,.48)}
.artifact-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem}
.artifact-card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:1rem;min-width:0;overflow-wrap:anywhere}
.artifact-card strong{color:#f8fbff!important}.artifact-card small{color:var(--muted)!important}
@media (max-width: 760px){.block-container{width:calc(100% - 1rem);padding-left:.5rem;padding-right:.5rem}.stTabs [data-baseweb="tab"]{font-size:.78rem;padding:.45rem .55rem}.glass-card{min-height:0}.architecture-flow{padding:.8rem}.dataset-command{font-size:.8rem}}
</style>
<style>
[data-testid="stMetric"]{min-width:0!important}
[data-testid="stMetric"] label{white-space:normal!important;line-height:1.2!important}
[data-testid="stMetric"] [data-testid="stMetricValue"]{white-space:nowrap!important}
.architecture-flow{display:flex;align-items:center;gap:.55rem;flex-wrap:wrap;background:linear-gradient(135deg,#101a2d,#172b48);border:1px solid rgba(80,217,233,.35);border-radius:14px;padding:1.1rem 1.25rem;margin:.35rem 0 1.15rem;box-shadow:0 12px 28px #0003}
.architecture-flow .flow-step{display:inline-flex;align-items:center;min-height:2.35rem;padding:.55rem .8rem;border:1px solid rgba(135,160,202,.35);border-radius:9px;background:#1b3150;color:#f4f8ff!important;font-weight:700;line-height:1.25}
.architecture-flow .flow-arrow{color:#50d9e9!important;font-size:1.25rem;font-weight:800}
.dataset-command{background:#101a2d;border:1px solid rgba(135,160,202,.35);border-radius:12px;padding:1.05rem 1.25rem;margin-top:.55rem;overflow-x:auto}
.dataset-command code{background:transparent!important;color:#f4f8ff!important;font-family:Consolas,"Courier New",monospace;font-size:.95rem;white-space:pre}
.dataset-command .muted-command{color:#cbd5e1!important}
.dataset-command .path-command{color:#50d9e9!important}
.dataset-command .flag-command{color:#f59e0b!important}
.architecture-code{background:#101a2d;border:1px solid rgba(135,160,202,.35);border-radius:12px;padding:1.2rem 1.35rem;margin-top:.55rem;overflow-x:auto}
.architecture-code pre{margin:0;background:transparent!important;color:#e8eefc!important;font-family:Consolas,"Courier New",monospace;font-size:.95rem;line-height:1.55;white-space:pre}
.pdf-preview{background:#f4f6fa;border:1px solid rgba(135,160,202,.3);border-radius:16px;padding:.6rem;overflow:hidden}
[data-testid="stSidebar"]{min-width:18rem;max-width:22rem}
@media (max-width: 760px){.block-container{width:calc(100% - 1rem);padding-left:.5rem;padding-right:.5rem}.hero h1{font-size:1.5rem}.stTabs [data-baseweb="tab"]{font-size:.78rem;padding:.35rem}.architecture-flow{gap:.35rem}.architecture-flow .flow-step{font-size:.82rem}}
</style>""", unsafe_allow_html=True)

PRESETS = {
    "Dynamic Attention Head Gating for Efficient Transformer Inference": {"domain":"NLP / Efficiency","objective":"Improve inference efficiency while preserving language modeling quality","constraints":"CPU-friendly evaluation, transparent ablations, reproducible metrics"},
    "Vision Transformer with Multi-Scale Patch Embedding for CIFAR-10 & Medical Imaging": {"domain":"Computer Vision","objective":"Improve visual representation quality across natural and medical images","constraints":"Compare multi-scale features, report latency and robustness"},
    "Image Captioning with CLIP": {"domain":"Multimodal & Vision-Language","objective":"Generate faithful image captions using CLIP-aligned visual and textual representations","constraints":"Evaluate caption quality, cross-modal alignment, grounding, and hallucination risk"},
    "Spatio-Temporal Graph Neural Networks for Real-Time Financial Fraud Detection": {"domain":"Graph Learning / FinTech","objective":"Detect suspicious transaction patterns with high recall and low latency","constraints":"Imbalanced labels, temporal split, interpretable decisions"},
    "Contrastive Multimodal Learning for Automated Radiology Report Generation": {"domain":"Healthcare AI","objective":"Align radiology images and clinical language for accurate report generation","constraints":"Cross-modal alignment, clinical safety, external validation"},
}
CUSTOM = "Custom Topic (Enter any research topic...)"


def apply_preset() -> None:
    choice = st.session_state.get("preset", CUSTOM)
    if choice in PRESETS:
        for key, value in {"topic": choice, **PRESETS[choice]}.items():
            st.session_state[key] = value


def run_pipeline(topic: str, domain: str, objective: str, constraints: str) -> dict[str, Any] | None:
    topic = sanitize_academic_topic(topic)
    payload = {
        "topic": topic,
        "domain": domain,
        "objective": objective,
        "constraints": constraints,
        "require_verified_data": False,
    }
    try:
        response = requests.post("http://localhost:8000/api/pipeline/run", json=payload, timeout=60)
        if response.ok:
            return response.json()
        try:
            return {"error": response.json().get("detail", "The request was rejected by the research validator.")}
        except ValueError:
            return {"error": f"The backend rejected the request with HTTP {response.status_code}."}
    except requests.RequestException:
        pass
    try:
        from backend.main import _run_pipeline_request
        from backend.schemas import PipelineRequest
        return _run_pipeline_request(PipelineRequest(**payload)).model_dump(mode="json")
    except Exception as error:
        detail = getattr(error, "detail", str(error))
        return {"error": str(detail)}


def paper_markdown(paper: dict[str, Any], research: dict[str, Any]) -> str:
    table = pd.DataFrame(research.get("benchmark_table", [])).to_markdown(index=False) if research.get("benchmark_table") else "No benchmark rows available."
    models = pd.DataFrame(research.get("models_compared", []))[['name', 'role', 'backbone', 'params_m', 'latency_ms']].to_markdown(index=False) if research.get("models_compared") else "No model catalog available."
    datasets = pd.DataFrame(research.get("datasets_used", []))[['name', 'modality', 'sample_size', 'primary_metric', 'source_url']].to_markdown(index=False) if research.get("datasets_used") else "No dataset catalog available."
    ablations = pd.DataFrame(research.get("ablation_study", [])).to_markdown(index=False) if research.get("ablation_study") else "No ablation rows available."
    stack = ", ".join(research.get("tech_stack", [])) or "Not specified"
    threats = "\n".join(f"- {item}" for item in paper.get("threats_to_validity", [])) or "Not specified"
    formulas = "\n".join(f"{index}. `{formula}`" for index, formula in enumerate(paper.get("mathematical_formulas", []), 1))
    metadata = research.get("project_metadata") or {}
    metadata_text = "\n".join(f"- {key.replace('_', ' ').title()}: {value}" for key, value in metadata.items()) or "Not specified"
    knowledge = research.get("knowledge_base", {})
    inference = research.get("inference_engine", {})
    coordination = research.get("agent_coordination", {})
    sections = [("Abstract", paper["abstract"]),("Keywords", ", ".join(paper["keywords"])),("1. Introduction", paper["introduction"]),("2. Related Work", paper["related_work"]),("3. Proposed Methodology", paper["methodology"]),("4. Experiments & Comparative Evaluation", paper["experimental_results"]),("5. Discussion & Failure Mode Analysis", paper["discussion"]),("6. Conclusion & Future Directions", paper["conclusion"]),("Threats to Validity", threats)]
    refs = "\n".join(f"{i}. {item}" for i, item in enumerate(paper["references"], 1))
    return "# " + paper["title"] + f"\n\n## PTIT Project Metadata\n{metadata_text}\n\n" + "\n\n".join(f"## {title}\n{body}" for title, body in sections) + f"\n\n## Mathematical Formulation\n{formulas}\n\n## Intelligent-System Architecture\nKnowledge Base taxonomy: {', '.join(knowledge.get('taxonomy', []))}\n\nInference Engine: {inference.get('matching_strategy', 'Unknown')}\nSelected domain: {inference.get('selected_domain', 'Unknown')}\nMatched keywords: {', '.join(inference.get('matched_keywords', []))}\n\nMulti-Agent Coordination: {', '.join(coordination.get('agents', []))}\n\n## Models Used\n{models}\n\n## Datasets Used\n{datasets}\n\n## Technical Stack\n{stack}\n\n## Benchmark Matrix\n{table}\n\n## Ablation Study\n{ablations}\n\n## Evidence Status\n{research.get('evidence_mode', 'Unknown')}\n\n## Intelligent-System Metrics\nReasoning Consistency Score: {research.get('reasoning_consistency_score', 0):.2f}\nKnowledge Retrieval Precision: {research.get('knowledge_retrieval_precision', 0):.2f}\n\n## 7. References\n{refs}\n"


def ieee_html_report(paper: dict[str, Any], research: dict[str, Any]) -> str:
    sections = [
        ("Abstract", paper["abstract"]),
        ("1. Introduction", paper["introduction"]),
        ("2. Related Work", paper["related_work"]),
        ("3. Proposed Methodology", paper["methodology"]),
        ("4. Experiments & Comparative Evaluation", paper["experimental_results"]),
        ("5. Discussion & Failure Mode Analysis", paper["discussion"]),
        ("6. Conclusion & Future Directions", paper["conclusion"]),
    ]
    body = "".join(f"<section><h2>{escape(title)}</h2><p>{escape(text).replace(chr(10), '<br>')}</p></section>" for title, text in sections)
    formulas = "".join(f"<div class='formula'>{escape(formula)}</div>" for formula in paper["mathematical_formulas"])
    references = "".join(f"<li>{escape(reference)}</li>" for reference in paper["references"])
    models = pd.DataFrame(research.get("models_compared", [])).to_html(index=False, border=0, classes="results-table", columns=["name", "role", "backbone", "params_m", "latency_ms"])
    datasets = pd.DataFrame(research.get("datasets_used", [])).to_html(index=False, border=0, classes="results-table", columns=["name", "modality", "sample_size", "primary_metric", "source_url"])
    table = pd.DataFrame(research.get("benchmark_table", [])).to_html(index=False, border=0, classes="results-table")
    ablations = pd.DataFrame(research.get("ablation_study", [])).to_html(index=False, border=0, classes="results-table")
    threats = "".join(f"<li>{escape(item)}</li>" for item in paper.get("threats_to_validity", []))
    stack = escape(", ".join(research.get("tech_stack", [])) or "Not specified")
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><title>{escape(paper['title'])}</title>
<style>
@page {{ size: A4; margin: 15mm; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; color: #172033; background: #edf1f7; font-family: Georgia, 'Times New Roman', serif; line-height: 1.48; }}
.paper {{ max-width: 1120px; margin: 24px auto; padding: 34px 42px; background: #fff; box-shadow: 0 10px 30px #17203322; }}
header {{ text-align: center; border-bottom: 1px solid #aeb8c7; padding-bottom: 18px; margin-bottom: 20px; }}
h1 {{ font-size: 25px; line-height: 1.16; margin: 0 auto 10px; max-width: 900px; }}
.meta {{ color: #536176; font: 12px Arial, sans-serif; }}
.columns {{ column-count: 2; column-gap: 30px; column-rule: 1px solid #d8dee8; }}
section, .formula, .table-wrap, .references {{ break-inside: avoid; }}
h2 {{ font-size: 15px; color: #183b63; margin: 0 0 7px; }}
p {{ font-size: 11px; text-align: justify; margin: 0 0 15px; }}
.formula {{ font-family: 'Times New Roman', serif; font-size: 12px; text-align: center; padding: 9px; margin: 8px 0; background: #f3f6fa; border-left: 3px solid #278ea5; }}
.table-wrap {{ column-span: all; margin-top: 18px; }}
.results-table {{ width: 100%; border-collapse: collapse; font: 10px Arial, sans-serif; }}
.results-table th {{ background: #183b63; color: white; text-align: left; }}
.results-table th, .results-table td {{ padding: 5px 6px; border: 1px solid #d8dee8; }}
.references {{ column-span: all; margin-top: 18px; font-size: 10px; }}
.references li {{ margin-bottom: 5px; }}
footer {{ margin-top: 24px; padding-top: 10px; border-top: 1px solid #aeb8c7; color: #68758a; font: 10px Arial, sans-serif; text-align: center; }}
@media print {{ body {{ background: white; }} .paper {{ margin: 0; padding: 0; box-shadow: none; }} }}
</style></head><body><main class='paper'>
<header><h1>{escape(paper['title'])}</h1><div class='meta'>Smart Research Lab · Academic manuscript</div></header>
<div class='columns'>{body}<section><h2>Mathematical Formulation</h2>{formulas}</section></div>
<div class='table-wrap'><h2>Models Used</h2>{models}<h2>Datasets Used</h2>{datasets}<h2>Benchmark Matrix</h2>{table}<h2>Ablation Study</h2>{ablations}<h2>Technical Stack</h2><p>{stack}</p><h2>Evidence Status</h2><p>{escape(research.get('evidence_mode', 'Unknown'))}</p><h2>Threats to Validity</h2><ul>{threats}</ul></div>
<div class='references'><h2>References</h2><ol>{references}</ol></div>
<footer>Smart Research Lab · Generated research artifact. Simulated metrics should be validated with instrumented experiments before publication.</footer>
</main></body></html>"""


def pdf_report(paper: dict[str, Any], research: dict[str, Any]) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=paper["title"],
        author="Smart Research Lab",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("PaperTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#183b63"), spaceAfter=8)
    heading_style = ParagraphStyle("PaperHeading", parent=styles["Heading2"], textColor=colors.HexColor("#183b63"), spaceBefore=10, spaceAfter=5)
    body_style = ParagraphStyle("PaperBody", parent=styles["BodyText"], leading=14, spaceAfter=7)
    small_style = ParagraphStyle("PaperSmall", parent=styles["BodyText"], fontSize=8, leading=10)
    story = [Paragraph(escape(paper["title"]), title_style)]
    story.append(Paragraph("Smart Research Lab · Academic manuscript", small_style))
    story.append(Spacer(1, 8))

    sections = [
        ("Abstract", paper["abstract"]),
        ("Keywords", ", ".join(paper["keywords"])),
        ("1. Introduction", paper["introduction"]),
        ("2. Related Work", paper["related_work"]),
        ("3. Proposed Methodology", paper["methodology"]),
        ("4. Experiments & Comparative Evaluation", paper["experimental_results"]),
        ("5. Discussion & Failure Mode Analysis", paper["discussion"]),
        ("6. Conclusion & Future Directions", paper["conclusion"]),
    ]

    subheading_style = ParagraphStyle(
        "PaperSubheading",
        parent=heading_style,
        fontSize=10,
        leading=12,
        textColor=colors.HexColor("#365b80"),
        spaceBefore=7,
        spaceAfter=3,
    )

    def add_academic_text(content: str) -> None:
        for raw_block in content.split("\n\n"):
            block = raw_block.strip()
            if not block:
                continue
            lines = block.splitlines()
            if lines and lines[0].startswith("### "):
                story.append(Paragraph(escape(lines[0][4:]), subheading_style))
                block = "\n".join(lines[1:]).strip()
            if block:
                story.append(Paragraph(escape(_plain_academic_text(block)).replace("\n", "<br/>"), body_style))

    for heading, content in sections:
        story.append(Paragraph(escape(heading), heading_style))
        add_academic_text(content)

    story.append(Paragraph("Mathematical Formulation", heading_style))
    for formula in paper.get("mathematical_formulas", []):
        formula_image = _formula_image(formula)
        if formula_image:
            story.append(Image(BytesIO(formula_image), width=150 * mm, height=18 * mm))
        else:
            story.append(Paragraph(escape(_clean_formula(formula)), ParagraphStyle("FormulaFallback", parent=body_style, alignment=TA_CENTER, backColor=colors.HexColor("#f3f6fa"), borderColor=colors.HexColor("#278ea5"), borderWidth=1, borderPadding=6)))

    story.append(Paragraph("Threats to Validity", heading_style))
    for threat in paper.get("threats_to_validity", []):
        story.append(Paragraph("• " + escape(threat), body_style))

    def add_table(title: str, rows: list[list[Any]], widths: list[float]) -> None:
        story.append(Paragraph(title, heading_style))
        table = Table([[Paragraph(escape(str(cell)), small_style) for cell in row] for row in rows], colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183b63")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c7d0dc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f7f9fc")),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)

    models = research.get("models_compared", [])
    add_table("Models Used", [["Name", "Role", "Backbone", "Params (M)", "Latency (ms)"]] + [
        [item["name"], item["role"], item["backbone"], item["params_m"], item["latency_ms"]] for item in models
    ], [42 * mm, 25 * mm, 58 * mm, 24 * mm, 25 * mm])
    datasets = research.get("datasets_used", [])
    add_table("Datasets Used", [["Dataset", "Modality", "Sample size", "Metric"]] + [
        [item["name"], item["modality"], item["sample_size"], item["primary_metric"]] for item in datasets
    ], [42 * mm, 38 * mm, 57 * mm, 37 * mm])
    benchmark = research.get("benchmark_table", [])
    add_table("Benchmark Matrix", [["Model", "Dataset", "Role", "Primary", "Latency", "VRAM MB", "FLOPs G", "Evidence"]] + [
        [item["model_name"], item["dataset_name"], item["role"], item["primary_metric"], item["latency_ms"], item.get("vram_mb", 0), item.get("flops_g", 0), item.get("evidence_type", "Planned")]
        for item in benchmark
    ], [31 * mm, 28 * mm, 16 * mm, 16 * mm, 17 * mm, 18 * mm, 15 * mm, 25 * mm])
    with tempfile.TemporaryDirectory(prefix="smart-research-charts-") as chart_dir:
        for chart_path in generate_research_charts(research, chart_dir):
            story.append(Image(BytesIO(chart_path.read_bytes()), width=170 * mm, height=70 * mm))
            story.append(Spacer(1, 5))
    story.append(Paragraph("7. References", heading_style))
    for index, reference in enumerate(paper.get("references", []), 1):
        story.append(Paragraph(f"{index}. {escape(reference)}", small_style))
    document.build(story)
    return buffer.getvalue()


def _clean_formula(formula: str) -> str:
    value = str(formula).replace(r"\qqad", r"\qquad").strip()
    value = value.replace(r"\mathcal{L}", "L").replace(r"\lambda", "lambda")
    value = value.replace(r"\lVert", "||").replace(r"\rVert", "||")
    return value.replace("{", "").replace("}", "")


def _math_formula(formula: str) -> str:
    return (
        str(formula)
        .replace(r"\qqad", r"\qquad")
        .replace(r"\lVert", r"\Vert")
        .replace(r"\rVert", r"\Vert")
        .strip()
    )


def _plain_academic_text(text: str) -> str:
    value = re.sub(r"\$\$.*?\$\$", "[mathematical formulation]", str(text), flags=re.DOTALL)
    value = re.sub(r"\\\(.*?\\\)", "[mathematical expression]", value, flags=re.DOTALL)
    replacements = {
        r"\mathcal": "L",
        r"\sigma": "sigma",
        r"\lambda": "lambda",
        r"\qquad": " ",
        r"\lVert": "||",
        r"\rVert": "||",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value.replace("\\", "")


def _formula_image(formula: str) -> bytes | None:
    """Render LaTeX math to a transparent PNG so ReportLab never prints raw commands."""
    try:
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(10, 0.8), dpi=180)
        figure.text(0.5, 0.5, f"${_math_formula(formula)}$", ha="center", va="center", fontsize=14)
        output = BytesIO()
        figure.savefig(output, format="png", dpi=180, transparent=True, bbox_inches="tight", pad_inches=0.12)
        plt.close(figure)
        return output.getvalue()
    except (ImportError, RuntimeError, ValueError):
        return None


def _pdf_open_link(pdf: bytes) -> str:
    encoded_pdf = base64.b64encode(pdf).decode("ascii")
    return (
        '<a class="pdf-open-link" href="data:application/pdf;base64,'
        f'{encoded_pdf}" target="_blank" rel="noopener noreferrer">'
        "🔍 Mở PDF ở tab mới</a>"
    )


def _latex_escape(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _latex_source(paper: dict[str, Any], research: dict[str, Any]) -> str:
    metadata = research.get("project_metadata") or {}
    sections = [
        ("Abstract", paper["abstract"]),
        ("Keywords", ", ".join(paper["keywords"])),
        ("1. Introduction", paper["introduction"]),
        ("2. Related Work", paper["related_work"]),
        ("3. Proposed Methodology", paper["methodology"]),
        ("4. Experimental Setup and Results", paper["experimental_results"]),
        ("5. Discussion", paper["discussion"]),
        ("6. Conclusion", paper["conclusion"]),
    ]
    body = []
    for heading, content in sections:
        body.append(f"\\section{{{_latex_escape(heading)}}}\n{_latex_escape(content)}")
    formulas = "\n".join(f"\\[\n{formula}\n\\]" for formula in paper.get("mathematical_formulas", []))
    references = "\n".join(
        f"\\item {_latex_escape(reference)}" for reference in paper.get("references", [])
    )
    threats = "".join(
        f"\\item {_latex_escape(item)}\n" for item in paper.get("threats_to_validity", [])
    )
    return rf"""\documentclass[10pt,conference]{{IEEEtran}}
\usepackage{{amsmath,amssymb}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{parskip}}
\usepackage{{xcolor}}
\usepackage{{hyperref}}
\title{{{_latex_escape(paper["title"])}}}
\\author{{Smart Research Lab}}
\date{{}}
\begin{{document}}
\maketitle
{chr(10).join(body)}
\section{{Mathematical Formulation}}
{formulas}
\section{{Threats to Validity}}
\begin{{itemize}}
{threats}
\end{{itemize}}
\section{{References}}
\begin{{enumerate}}
{references}
\end{{enumerate}}
\end{{document}}
"""


def persist_artifacts(paper: dict[str, Any], research: dict[str, Any], pdf: bytes) -> tuple[bytes, Path]:
    root = Path(__file__).resolve().parent / "results" / "artifacts"
    slug = re.sub(r"[^a-z0-9]+", "-", paper["title"].lower()).strip("-")[:70] or "research-paper"
    run_dir = root / f"{slug}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "guardrail_report.json").write_text(
        json.dumps(audit_manuscript(paper, research), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "paper.pdf").write_bytes(pdf)
    (run_dir / "paper.tex").write_text(_latex_source(paper, research), encoding="utf-8")
    (run_dir / "benchmark_matrix.json").write_text(
        json.dumps(
            {
                "evidence_status": research.get("evidence_mode", "Unknown"),
                "topic": paper["title"],
                "rows": research.get("benchmark_table", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        generate_research_charts(research, charts_dir)
        import matplotlib.pyplot as plt

        curves = pd.DataFrame(research.get("training_curves", []))
        if not curves.empty:
            figure, axis = plt.subplots(figsize=(8, 4.5))
            for column, label in (("proposed_loss", "Loss"), ("proposed_score", "Primary metric")):
                if column in curves:
                    axis.plot(curves["epoch"], curves[column], marker="o", label=label)
            axis.set_title("Training curves")
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.25)
            axis.legend()
            figure.tight_layout()
            figure.savefig(charts_dir / "loss_curve.png", dpi=160)
            plt.close(figure)
        benchmark = pd.DataFrame(research.get("benchmark_table", []))
        if not benchmark.empty and {"model_name", "primary_metric"}.issubset(benchmark.columns):
            figure, axis = plt.subplots(figsize=(8, 4.5))
            axis.bar(benchmark["model_name"], benchmark["primary_metric"])
            axis.set_title("Benchmark matrix: primary metric")
            axis.tick_params(axis="x", labelrotation=25)
            figure.tight_layout()
            figure.savefig(charts_dir / "benchmark_matrix.png", dpi=160)
            plt.close(figure)
    except ImportError:
        (charts_dir / "README.txt").write_text("Install matplotlib to generate PNG charts.\n", encoding="utf-8")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in run_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(run_dir))
    zip_bytes = zip_buffer.getvalue()
    (run_dir / "artifacts.zip").write_bytes(zip_bytes)
    return zip_bytes, run_dir


def answer_paper_question(question: str, paper: dict[str, Any], research: dict[str, Any]) -> str:
    payload = {"question": question, "paper": paper, "research": research}
    try:
        response = requests.post("http://localhost:8000/api/paper/ask", json=payload, timeout=15)
        if response.ok:
            return response.json().get("answer", "No answer available.")
    except requests.RequestException:
        pass
    try:
        from backend.main import _answer_paper_question
        from backend.schemas import ResearcherOutput, WriterOutput
        return _answer_paper_question(question, WriterOutput.model_validate(paper), ResearcherOutput.model_validate(research))
    except Exception:
        return "I could not analyze that question from the generated manuscript. Try asking about the method, baselines, datasets, results, or limitations."


def render_header(result: dict[str, Any]) -> None:
    plan, research, review = result["plan"], result["research"], result["review"]
    proposed = next(item for item in research["models_compared"] if item["role"] == "Proposed")
    proposed_row = next(item for item in research["benchmark_table"] if item["role"] == "Proposed")
    st.markdown(f"<section class='hero'><h1>{escape(plan['title'])}</h1><p>Phòng thí nghiệm nghiên cứu AI · Hệ thống đa tác tử</p><span class='domain-badge'>{escape(plan['domain'])}</span></section>", unsafe_allow_html=True)
    metric_rows = [st.columns(3), st.columns(3)]
    cards = metric_rows[0] + metric_rows[1]
    cards[0].metric("Planned Primary Score", f"{proposed_row['primary_metric']:.2f}", "controlled estimate")
    cards[1].metric("Latency", f"{proposed['latency_ms']:.1f} ms", f"{proposed['params_m']:.1f}M params")
    cards[2].metric("Experiment Scope", f"{len(research['datasets_used'])} datasets", "1 proposed method")
    cards[3].metric("Review Score", f"{review['overall_score']:.1f} / 10", review["verdict"])
    cards[4].metric("Reasoning Consistency", f"{research.get('reasoning_consistency_score', 0):.2f}", "routing audit")
    cards[5].metric("Knowledge Precision", f"{research.get('knowledge_retrieval_precision', 0):.2f}", "catalog alignment")


def render_plan(plan: dict[str, Any]) -> None:
    st.markdown(f"<div class='callout'><strong>Research Question</strong><br>{escape(plan['research_question'])}<br><br><strong>Core Hypothesis</strong><br>{escape(plan['hypothesis'])}</div>", unsafe_allow_html=True)
    architecture = st.columns(3)
    architecture[0].markdown("**Knowledge Base**\n\n" + "\n".join(f"- {item}" for item in plan.get("knowledge_base", {}).get("taxonomy", [])))
    architecture[1].markdown("**Inference Engine**\n\n" + plan.get("inference_engine", {}).get("matching_strategy", "Not specified") + "\n\nDomain: " + plan.get("inference_engine", {}).get("selected_domain", "Not specified"))
    architecture[2].markdown("**Multi-Agent Coordination**\n\n" + "\n".join(f"- {item}" for item in plan.get("agent_coordination", {}).get("agents", [])))
    st.subheader("📦 Datasets Catalog")
    columns = st.columns(3)
    for index, dataset in enumerate(plan["datasets_catalog"]):
        with columns[index % 3]:
            st.markdown(f"<article class='glass-card'><h4>{escape(dataset['name'])}</h4><span class='badge'>{escape(dataset['modality'])}</span><p><b>Scale:</b> {escape(dataset['sample_size'])}<br><b>Input:</b> {escape(dataset['input_spec'])}<br><b>Metric:</b> {escape(dataset['primary_metric'])}</p><small>{escape(dataset['description'])}</small></article>", unsafe_allow_html=True)
    st.subheader("🧠 Kho mô hình & kiến trúc")
    models = pd.DataFrame([item for item in plan["models_catalog"] if item["role"] == "Proposed"])
    st.dataframe(models[["name","backbone","params_m","flops_g","latency_ms","key_characteristic"]], width="stretch", hide_index=True)
    if HAS_PLOTLY:
        chart = models.melt(id_vars="name", value_vars=["params_m","latency_ms"], var_name="Measure", value_name="Value")
        st.plotly_chart(px.bar(chart, x="name", y="Value", color="Measure", barmode="group", template="plotly_dark", title="Parameters vs latency"), width="stretch")


def render_benchmarks(research: dict[str, Any]) -> None:
    selected = st.selectbox("Chọn tập dữ liệu", [item["name"] for item in research["datasets_used"]])
    curves = pd.DataFrame(research["training_curves"])
    curves = curves[curves["dataset_name"] == selected]
    left, right = st.columns(2)
    if HAS_PLOTLY:
        loss = curves[["epoch", "proposed_loss"]].rename(columns={"proposed_loss": "Loss"})
        score = curves[["epoch", "proposed_score"]].rename(columns={"proposed_score": "Primary Metric"})
        left.plotly_chart(px.line(loss, x="epoch", y="Loss", markers=True, template="plotly_dark", title="Training / validation loss"), width="stretch")
        right.plotly_chart(px.line(score, x="epoch", y="Primary Metric", markers=True, template="plotly_dark", title="Performance growth"), width="stretch")
    st.subheader("So sánh với bài gốc và các baseline")
    benchmark = pd.DataFrame(research["benchmark_table"])
    st.caption("Bảng này giữ cả baseline và phương pháp đề xuất để đối chiếu trực tiếp; các giá trị mô phỏng vẫn có nhãn bằng chứng.")
    st.dataframe(benchmark, width="stretch", hide_index=True)
    if HAS_PLOTLY:
        left, middle, right = st.columns(3)
        left.plotly_chart(px.bar(benchmark, x="model_name", y="primary_metric", title="Primary metric", template="plotly_dark"), width="stretch")
        middle.plotly_chart(px.bar(benchmark, x="model_name", y="latency_ms", title="Latency (ms)", template="plotly_dark"), width="stretch")
        right.plotly_chart(px.bar(benchmark, x="model_name", y="params_m", title="Parameters (M)" , template="plotly_dark"), width="stretch")
    if research.get("ablation_study"):
        st.subheader("Thử nghiệm loại bỏ từng thành phần")
        ablation = pd.DataFrame(research["ablation_study"])
        st.dataframe(ablation, width="stretch", hide_index=True)
        if HAS_PLOTLY:
            st.plotly_chart(px.bar(ablation, x="configuration", y="primary_metric", color="removed_component", title="Effect of removing each component", template="plotly_dark"), width="stretch")
    st.subheader("Phát hiện chính")
    for finding in research["key_findings"]:
        st.markdown(f"- {finding}")


def render_verified_datasets() -> None:
    st.subheader("Verified Local Datasets")
    st.caption("Only files present on this machine are marked Ready. Catalog-only datasets are never treated as measured evidence.")
    data_root = Path(__file__).resolve().parent / "data"
    entries = [
        ("Shakespeare Char", data_root / "shakespeare_char" / "input.txt", "Character language modeling", "https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt"),
        ("Enwik8", data_root / "enwik8" / "enwik8", "Character language modeling", "https://www.byronosa.com/enwik8"),
        ("Text8", data_root / "text8" / "text8", "Character language modeling", "http://mattmahoney.net/dc/textdata.html"),
        ("AG News", data_root / "huggingface" / "ag_news" / "train.jsonl", "Text classification", "https://huggingface.co/datasets/fancyzhx/ag_news"),
        ("IMDb Large Movie Review", data_root / "huggingface" / "imdb" / "train.jsonl", "Sentiment classification", "https://huggingface.co/datasets/stanfordnlp/imdb"),
    ]
    rows = []
    for name, path, task, source in entries:
        ready = path.is_file() and path.stat().st_size > 0
        rows.append({"Dataset": name, "Task": task, "Status": "Ready" if ready else "Not downloaded", "Size (MB)": round(path.stat().st_size / 1_000_000, 2) if ready else 0.0, "Source": source})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    available = [row["Dataset"] for row in rows if row["Status"] == "Ready"]
    if not available:
        st.info("No verified local dataset is available yet. Run the preparation command shown below.")
        return
    selected = st.selectbox("Inspect a local dataset", available, key="verified_dataset_select")
    selected_path = next(path for name, path, _, _ in entries if name == selected)
    if selected_path.suffix == ".jsonl":
        sample = pd.read_json(selected_path, lines=True).head(20)
        st.dataframe(sample, width="stretch", hide_index=True)
    else:
        st.info(f"Binary/text corpus available at `{selected_path.relative_to(data_root.parent)}` ({selected_path.stat().st_size:,} bytes).")
    st.markdown(
        '<div class="dataset-command"><code>.<span class="muted-command">\\env\\Scripts\\python.exe</span> '
        '<span class="path-command">data\\prepare_hf_text.py</span> '
        '<span class="flag-command">--dataset</span> ag_news</code></div>',
        unsafe_allow_html=True,
    )


def render_method(plan: dict[str, Any], paper: dict[str, Any]) -> None:
    st.subheader("Luồng kiến trúc")
    flow_steps = [
        "Input data",
        "Domain encoder",
        "Task-aware signal selection",
        "Proposed model",
        "Prediction + uncertainty",
    ]
    flow_html = '<div class="architecture-flow">'
    for index, step in enumerate(flow_steps):
        if index:
            flow_html += '<span class="flow-arrow" aria-hidden="true">→</span>'
        flow_html += f'<span class="flow-step">{escape(step)}</span>'
    flow_html += "</div>"
    st.markdown(flow_html, unsafe_allow_html=True)
    st.write(paper["methodology"])
    st.subheader("Mathematical Formulation")
    for formula in paper["mathematical_formulas"]:
        st.latex(formula)
    st.subheader(f"PyTorch Core: {plan['proposed_method']}")
    pytorch_code = """class TaskAwareBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Linear(dim, dim)
        self.projection = nn.Linear(dim, dim)

    def forward(self, x):
        weights = torch.sigmoid(self.gate(self.norm(x)))
        return x + weights * self.projection(self.norm(x))
"""
    st.markdown(
        f'<div class="architecture-code"><pre>{escape(pytorch_code)}</pre></div>',
        unsafe_allow_html=True,
    )


def render_paper(paper: dict[str, Any], research: dict[str, Any]) -> None:
    pdf = st.session_state["pdf_bytes"]
    st.markdown(
        f"<section class='hero'><h1>{escape(paper['title'])}</h1>"
        "<p>Bản thảo PDF đã sẵn sàng · Bố cục học thuật</p>"
        f"<span class='domain-badge'>{escape(str(research.get('inference_engine', {}).get('selected_domain', 'AI Research')))}</span></section>",
        unsafe_allow_html=True,
    )
    st.markdown("#### Xem trước bản thảo")
    try:
        import pymupdf as fitz
        document = fitz.open(stream=pdf, filetype="pdf")
        st.session_state["pdf_pages_images"] = []
        for page_number, page in enumerate(document, 1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
            page_image = pixmap.tobytes("png")
            st.session_state["pdf_pages_images"].append(page_image)
            st.image(page_image, width="stretch")
            st.caption(f"Trang {page_number}")
        document.close()
    except (ImportError, RuntimeError, ValueError) as error:
        st.warning(f"Không thể dựng ảnh xem trước PDF: {error}. Hãy tải file PDF bên dưới.")
    action_column, open_column = st.columns(2)
    with action_column:
        st.download_button(
            "📥 Tải bài báo PDF",
            pdf,
            "research_paper.pdf",
            "application/pdf",
            width="stretch",
        )
    with open_column:
        st.markdown(_pdf_open_link(pdf), unsafe_allow_html=True)
    st.download_button(
        "📦 Tải toàn bộ artifacts (.ZIP)",
        st.session_state["zip_bytes"],
        "research_artifacts.zip",
        "application/zip",
        width="stretch",
    )
    with st.expander("Xem nội dung bản thảo và thông tin tái lập"):
        sections = [
            ("Abstract", paper["abstract"]),
            ("Keywords", ", ".join(paper["keywords"])),
            ("1. Introduction", paper["introduction"]),
            ("2. Related Work", paper["related_work"]),
            ("3. Proposed Methodology", paper["methodology"]),
            ("4. Experimental Setup & Results", paper["experimental_results"]),
            ("5. Discussion", paper["discussion"]),
            ("6. Conclusion", paper["conclusion"]),
        ]
        for title, body in sections:
            st.subheader(title)
            st.write(body)
        st.subheader("Mathematical Formulation")
        for formula in paper.get("mathematical_formulas", []):
            st.latex(formula)
        st.subheader("Threats to Validity")
        for threat in paper.get("threats_to_validity", []):
            st.markdown(f"- {threat}")
        st.subheader("References")
        for index, reference in enumerate(paper["references"], 1):
            st.markdown(f"{index}. {reference}")
        st.subheader("Models and Datasets")
        model_column, dataset_column = st.columns(2)
        with model_column:
            st.dataframe(pd.DataFrame(research["models_compared"])[["name", "backbone", "params_m", "latency_ms"]], width="stretch", hide_index=True)
        with dataset_column:
            st.dataframe(pd.DataFrame(research["datasets_used"])[["name", "modality", "sample_size", "primary_metric", "source_url"]], width="stretch", hide_index=True)
        st.caption("Technical stack: " + (", ".join(research.get("tech_stack", [])) or "Not specified"))
    st.divider()
    st.subheader("💬 Hỏi về bài báo")
    question = st.text_input("Đặt câu hỏi về bài báo", placeholder="Ví dụ: Bài báo so sánh với baseline nào?", key="paper_question")
    if question.strip():
        with st.spinner("Đang phân tích bản thảo..."):
            answer = answer_paper_question(question, paper, research)
        st.markdown(f"<div class='callout'><strong>Research Assistant</strong><br>{escape(answer)}</div>", unsafe_allow_html=True)


def render_artifacts() -> None:
    artifact_dir = Path(st.session_state["artifact_dir"])
    st.subheader("📦 Kho tệp kết quả")
    st.caption("Các tệp được giữ trong phiên làm việc; chuyển tab hoặc tải xuống không làm mất kết quả.")
    files = [
        ("📄 PDF", "pdf_bytes", "application/pdf", "research_paper.pdf"),
        ("🧾 LaTeX", "tex_bytes", "text/plain", "paper.tex"),
        ("📊 Benchmark JSON", "benchmark_bytes", "application/json", "benchmark_matrix.json"),
        ("🛡️ Guardrail report", "guardrail_bytes", "application/json", "guardrail_report.json"),
        ("📦 ZIP", "zip_bytes", "application/zip", "research_artifacts.zip"),
    ]
    cards = st.columns(3)
    for index, (label, state_key, mime, filename) in enumerate(files):
        with cards[index % len(cards)]:
            st.markdown(
                f"<article class='artifact-card'><strong>{label}</strong><br><small>{filename}</small></article>",
                unsafe_allow_html=True,
            )
            st.download_button(
                f"Tải {filename}",
                st.session_state[state_key],
                filename,
                mime,
                key=f"download_{state_key}",
                width="stretch",
            )
    st.divider()
    st.caption(f"Thư mục kết quả: {artifact_dir.name}")


def render_review(review: dict[str, Any]) -> None:
    metrics = st.columns(4)
    for column, label, key in zip(metrics, ("Tính mới","Độ chặt chẽ","Độ rõ ràng","Điểm tổng"), ("novelty_score","rigor_score","clarity_score","overall_score")):
        column.metric(label, f"{review[key]:.1f} / 10")
    st.markdown(f"<div class='callout'><strong>Đánh giá: {review['verdict']}</strong></div>", unsafe_allow_html=True)
    columns = st.columns(3)
    for column, title, key in zip(columns, ("✅ Điểm mạnh","⚠️ Điểm yếu","🛠️ Đề xuất sửa"), ("strengths","weaknesses","revision_suggestions")):
        with column:
            st.subheader(title)
            for item in review[key]:
                st.markdown(f"- {item}")


def main() -> None:
    first = next(iter(PRESETS))
    for key, value in {"topic":first, **PRESETS[first]}.items():
        st.session_state.setdefault(key, value)
    for key, value in {
        "full_pipeline_result": None,
        "full_paper_text": "",
        "pdf_file_path": "",
        "benchmark_data": [],
        "pdf_bytes": b"",
        "tex_bytes": b"",
        "benchmark_bytes": b"",
        "guardrail_bytes": b"",
        "zip_bytes": b"",
        "artifact_dir": "",
    }.items():
        st.session_state.setdefault(key, value)
    with st.sidebar:
        st.markdown("## Bảng điều khiển nghiên cứu")
        st.selectbox("⚡ Chủ đề mẫu", [*PRESETS, CUSTOM], key="preset", on_change=apply_preset)
        st.text_area("Chủ đề nghiên cứu", key="topic", height=90)
        st.text_input("Lĩnh vực", key="domain")
        st.text_input("Mục tiêu chính", key="objective")
        st.text_area("Ràng buộc", key="constraints", height=75)
        run_clicked = st.button("🚀 Bắt đầu sinh bài báo", type="primary", width="stretch")
    if run_clicked or not st.session_state["full_pipeline_result"]:
        with st.status("🤖 Đang điều phối hệ thống nghiên cứu AI...", expanded=True) as status:
            status.write("⏳ Đang khởi tạo Planner Agent và phân loại chủ đề...")
            result = run_pipeline(st.session_state.topic, st.session_state.domain, st.session_state.objective, st.session_state.constraints)
            if not result:
                status.update(label="Pipeline unavailable", state="error")
                st.stop()
            if "error" in result:
                status.update(label="Request rejected", state="error", expanded=True)
                st.error(result["error"])
                st.stop()
            status.write("✍️ Writer Agent đã hoàn tất bản thảo học thuật...")
            status.write("🛡️ Guardrail đang kiểm tra công thức, benchmark và evidence labels...")
            status.write("📄 Đang biên dịch PDF và đóng gói artifacts...")
            export_research = result["research"]
            paper = result["paper"]
            pdf = pdf_report(paper, export_research)
            zip_bytes, artifact_dir = persist_artifacts(paper, export_research, pdf)
            st.session_state.full_pipeline_result = result
            st.session_state.full_paper_text = paper_markdown(paper, export_research)
            st.session_state.pdf_bytes = pdf
            st.session_state.zip_bytes = zip_bytes
            st.session_state.artifact_dir = str(artifact_dir)
            st.session_state.pdf_file_path = str(artifact_dir / "paper.pdf")
            st.session_state.benchmark_data = export_research.get("benchmark_table", [])
            st.session_state.tex_bytes = (artifact_dir / "paper.tex").read_bytes()
            st.session_state.benchmark_bytes = (artifact_dir / "benchmark_matrix.json").read_bytes()
            st.session_state.guardrail_bytes = (artifact_dir / "guardrail_report.json").read_bytes()
            status.update(label="✅ Pipeline completed · manuscript ready", state="complete", expanded=False)
    result = st.session_state["full_pipeline_result"]
    if result.get("paper_mode") == "dynamic_fallback" and result.get("mode_used") == "live_gemini":
        st.warning("Gemini produced a live plan, but the manuscript uses the fallback writer because the live response was invalid. Benchmarks remain Controlled Planning Simulation Estimates.")
    elif result.get("paper_mode") == "dynamic_fallback":
        st.warning("Simulation mode: benchmark, latency, and training curves are planning estimates, not measurements from a real dataset. Set GEMINI_API_KEY for live planning; validate all numbers with instrumented experiments before submission.")
    else:
        st.success("Live planner and writer are enabled with Gemini. Benchmark values still require validation with instrumented experiments before publication.")
    render_header(result)
    plan, research, paper, review = result["plan"], result["research"], result["paper"], result["review"]
    tabs = st.tabs([
        "⚙️ Cấu hình chủ đề",
        "📄 Xem trước bản thảo",
        "📊 So sánh & biểu đồ",
        "📦 Tệp kết quả",
    ])
    with tabs[0]:
        render_plan(plan)
        render_method(plan, paper)
        render_verified_datasets()
        render_review(review)
    with tabs[1]:
        render_paper(paper, research)
    with tabs[2]:
        render_benchmarks(research)
    with tabs[3]:
        render_artifacts()


if __name__ == "__main__":
    main()
