"""Reusable Streamlit presentation helpers for the ContextCheck dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.utils import MODEL_DIR, RESULTS_PATH


LABEL_META = {
    "entailment": {
        "title": "Entailment",
        "icon": "✓",
        "color": "#16a085",
        "description": "The premise supports or implies the hypothesis.",
    },
    "neutral": {
        "title": "Neutral",
        "icon": "≈",
        "color": "#d99b18",
        "description": "The premise does not establish either relationship.",
    },
    "contradiction": {
        "title": "Contradiction",
        "icon": "×",
        "color": "#d2565d",
        "description": "The premise conflicts with the hypothesis.",
    },
}


def inject_styles() -> None:
    """Apply the app's visual language consistently across pages."""
    st.markdown(
        """
        <style>
        :root { --ink: #f4f7ff; --muted: #9aa8c3; --line: #263656; --accent: #8196ff; }
        .stApp { background: radial-gradient(circle at 85% 4%, #182750 0, #0b1120 33rem), #0b1120; color: var(--ink); }
        [data-testid="stHeader"] { background: rgba(11,17,32,.78); }
        [data-testid="stSidebar"] { background: #080d19; border-right: 1px solid #1b2944; }
        [data-testid="stSidebar"] * { color: #e6ebff !important; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #9aa8c3 !important; }
        [data-testid="stSidebarNav"] { padding-top: 1.5rem; }
        [data-testid="stSidebarNav"] li span { font-size: .9rem; }
        [data-testid="stSidebarNav"] li:hover { background: #172442; border-radius: 8px; }
        h1, h2, h3 { letter-spacing: -0.025em; color: var(--ink); }
        h1 { font-weight: 800; }
        h2 { margin-top: 2rem; }
        .hero { position: relative; overflow: hidden; padding: 2.8rem 3rem; border-radius: 26px; color: white;
                background: linear-gradient(120deg, #17234a 0%, #304eae 54%, #6d79ed 100%);
                border: 1px solid rgba(151,168,255,.35); box-shadow: 0 22px 65px rgba(25,41,108,.38);
                margin-bottom: 1.5rem; }
        .hero:after { content: ""; position: absolute; width: 260px; height: 260px; right: -85px; top: -105px;
                      border-radius: 50%; border: 1px solid rgba(255,255,255,.2);
                      box-shadow: 0 0 0 24px rgba(255,255,255,.04), 0 0 0 50px rgba(255,255,255,.03); }
        .hero h1 { color: white; font-size: 3.5rem; margin: .35rem 0 0; position: relative; z-index: 1; }
        .hero p { color: #e8edff; max-width: 760px; font-size: 1.08rem; margin: .65rem 0 0; position: relative; z-index: 1; }
        .eyebrow { color: #b9c9ff; text-transform: uppercase; letter-spacing: .14em;
                   font-size: .73rem; font-weight: 700; }
        .card { background: rgba(17,27,48,.84); border: 1px solid var(--line); border-radius: 16px;
                padding: 1.2rem 1.3rem; box-shadow: 0 10px 28px rgba(0,0,0,.14); height: 100%; }
        .card h3 { margin-top: 0; font-size: 1.05rem; }
        .card p { color: var(--muted); line-height: 1.55; }
        .status-pill { display: inline-block; border-radius: 999px; padding: .3rem .7rem;
                       font-weight: 700; font-size: .78rem; background: #1c2b50; color: #b9c9ff; }
        .result-card { border-radius: 18px; padding: 1.2rem 1.35rem; background: #111b31;
                       border-left: 6px solid var(--result-color); box-shadow: 0 8px 22px rgba(30,45,80,.07); }
        .result-card h2 { margin: 0; }
        .muted { color: var(--muted); }
        .stButton > button[kind="primary"] { border-radius: 10px; font-weight: 700; color: white;
             background: linear-gradient(100deg,#596de0,#8a78ef); border: 0; min-height: 2.7rem; }
        .stButton > button[kind="primary"]:hover { border: 1px solid #b6c1ff; color: white; }
        [data-testid="stMetric"] { background: rgba(17,27,48,.9); border: 1px solid var(--line);
                                   padding: .8rem 1rem; border-radius: 13px; }
        [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #f4f7ff !important; }
        [data-testid="stAlert"] { border-radius: 12px; }
        [data-baseweb="select"] > div, [data-testid="stTextArea"] textarea,
        [data-testid="stFileUploaderDropzone"] { background: #10192c; color: #f4f7ff;
             border-color: #2a3c62; border-radius: 12px; }
        [data-testid="stTextArea"] textarea:focus { border-color: #8798ff; box-shadow: 0 0 0 1px #8798ff; }
        [data-testid="stFileUploaderDropzone"] { border-style: dashed; }
        label, [data-testid="stWidgetLabel"] p { color: #c7d1e8 !important; font-weight: 600; }
        hr { border-color: #243452; }
        .input-panel { background: #10192c; border: 1px solid #263656; border-radius: 16px;
                       padding: 1rem 1rem .5rem; height: 100%; }
        .input-panel h3 { color: #eef2ff; margin: 0 0 .45rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render compact navigation branding and model readiness."""
    with st.sidebar:
        st.markdown("## 🧭 ContextCheck")
        st.caption("Transformer-based natural language inference")
        st.divider()
        if MODEL_DIR.is_dir():
            st.markdown('<span class="status-pill">● Model ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill">○ Training required</span>', unsafe_allow_html=True)
            st.caption("Run the training script or Colab notebook to enable live analysis.")
        st.divider()
        st.caption("Compare text relationships — not internet facts.")


def render_hero(eyebrow: str, title: str, description: str) -> None:
    """Render a consistent page hero."""
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f"<h1>{title}</h1><p>{description}</p></div>",
        unsafe_allow_html=True,
    )


def load_saved_results() -> dict:
    """Load measured results when available, otherwise return an empty mapping."""
    if not RESULTS_PATH.is_file():
        return {}
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def render_label_result(label: str, score: float) -> None:
    """Render a styled prediction summary."""
    meta = LABEL_META[label]
    st.markdown(
        f'<div class="result-card" style="--result-color:{meta["color"]}">'
        f'<div class="eyebrow" style="color:{meta["color"]}">Model decision</div>'
        f'<h2>{meta["icon"]}&nbsp; {meta["title"]}</h2>'
        f'<p class="muted">{meta["description"]} Top prediction score: '
        f"<strong>{score:.1%}</strong></p></div>",
        unsafe_allow_html=True,
    )
