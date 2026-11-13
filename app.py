"""ContextCheck Streamlit home page."""

import streamlit as st

st.set_page_config(
    page_title="ContextCheck | Natural Language Inference",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 ContextCheck")
st.subheader("Understand the relationship between two pieces of text")
st.write(
    "ContextCheck is an academic Natural Language Inference (NLI) project. "
    "It compares a **premise** (context) with a **hypothesis** (statement) and "
    "predicts whether the premise entails, contradicts, or does not settle it."
)
st.info(
    "ContextCheck compares only the two texts you provide. It does not search "
    "the internet or verify whether a statement is universally true."
)

columns = st.columns(3)
outcomes = [
    ("🟢 Entailment", "The premise supports or implies the hypothesis."),
    ("🔴 Contradiction", "The premise conflicts with the hypothesis."),
    ("🟡 Neutral", "The premise does not establish either relationship."),
]
for column, (title, description) in zip(columns, outcomes):
    with column:
        st.markdown(f"### {title}")
        st.write(description)

st.markdown("## How it works")
st.write(
    "A fine-tuned DistilBERT Transformer reads the premise and hypothesis as a "
    "paired input. Self-attention builds contextual representations of the words "
    "and their relationships, and a classification head produces scores for the "
    "three NLI labels."
)
st.markdown("## Explore the project")
st.markdown(
    "Use the pages in the sidebar to analyze a text pair, process a CSV, inspect "
    "measured evaluation results, compare models, or read about the project."
)
st.caption(
    "Stack: Python · PyTorch · Hugging Face Transformers/Datasets · "
    "scikit-learn · Streamlit · Plotly"
)
