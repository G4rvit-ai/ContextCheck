"""ContextCheck Streamlit home page."""

import streamlit as st

from src.ui import inject_styles, load_saved_results, render_hero, render_sidebar
from src.utils import is_model_ready

st.set_page_config(
    page_title="ContextCheck | Natural Language Inference",
    page_icon="🧭",
    layout="wide",
)
inject_styles()
render_sidebar()

render_hero(
    "Natural language inference lab",
    "ContextCheck",
    "Understand the relationship between a passage and a statement using a "
    "fine-tuned Transformer model.",
)
st.info(
    "ContextCheck compares only the two texts you provide. It does not search the "
    "internet or verify whether a statement is universally true."
)

try:
    results = load_saved_results()
except RuntimeError as error:
    st.error(str(error))
    results = {}
transformer = results.get("transformer")
status_col, metric_col, guide_col = st.columns(3)
with status_col:
    st.markdown('<div class="card"><h3>Model status</h3>', unsafe_allow_html=True)
    if is_model_ready():
        st.success("Fine-tuned model available")
        st.write("The analysis workspace is ready for live predictions.")
    else:
        st.warning("Training artifacts not found")
        st.write("Run training from Colab before using live inference.")
    st.markdown("</div>", unsafe_allow_html=True)
with metric_col:
    st.markdown('<div class="card"><h3>Evaluation status</h3>', unsafe_allow_html=True)
    if transformer:
        st.metric("Held-out accuracy", f"{transformer['accuracy']:.1%}")
        st.caption(f"{transformer['test_examples']:,} test examples")
    else:
        st.write("No measured results yet.")
        st.caption("Metrics appear automatically after evaluation.")
    st.markdown("</div>", unsafe_allow_html=True)
with guide_col:
    st.markdown('<div class="card"><h3>Recommended first step</h3>', unsafe_allow_html=True)
    st.write("Open **Text Analysis**, load a sample pair, and inspect the three class scores.")
    st.caption("Use Batch Analysis for repeatable CSV experiments.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("## Three possible relationships")
columns = st.columns(3, gap="medium")
outcomes = [
    ("✓ Entailment", "The premise supports or implies the hypothesis.", "#16a085"),
    ("× Contradiction", "The premise conflicts with the hypothesis.", "#d2565d"),
    ("≈ Neutral", "The premise does not establish either relationship.", "#d99b18"),
]
for column, (title, description, color) in zip(columns, outcomes):
    with column:
        st.markdown(
            f'<div class="card"><h3 style="color:{color}">{title}</h3>'
            f"<p>{description}</p></div>",
            unsafe_allow_html=True,
        )

st.markdown("## A simple, auditable workflow")
step_columns = st.columns(4, gap="medium")
steps = [
    ("01", "Write", "Provide a premise and a hypothesis."),
    ("02", "Encode", "DistilBERT reads both texts as a pair."),
    ("03", "Score", "A classifier produces three softmax scores."),
    ("04", "Learn", "Review measured metrics and confusion patterns."),
]
for column, (number, title, description) in zip(step_columns, steps):
    with column:
        st.markdown(
            f'<div class="card"><span class="eyebrow">{number}</span>'
            f"<h3>{title}</h3><p>{description}</p></div>",
            unsafe_allow_html=True,
        )

st.markdown("## Project capabilities")
capability_columns = st.columns(2, gap="medium")
capabilities = [
    ("Interactive analysis", "Test one pair at a time with an interpretable score distribution."),
    ("Batch experiments", "Upload up to 10,000 pairs, retain invalid rows, and download results."),
    ("Measured evaluation", "Inspect accuracy, macro/weighted F1, reports, and confusion matrices."),
    ("Academic comparison", "Compare contextual DistilBERT with a TF-IDF baseline."),
]
for column, (title, description) in zip(capability_columns * 2, capabilities):
    with column:
        st.markdown(
            f'<div class="card"><h3>{title}</h3><p>{description}</p></div>',
            unsafe_allow_html=True,
        )

st.caption("Stack: Python · PyTorch · Hugging Face · scikit-learn · Streamlit · Plotly")
