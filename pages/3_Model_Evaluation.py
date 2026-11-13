"""Visualize measured test results for the fine-tuned Transformer."""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.labels import LABEL_NAMES
from src.utils import RESULTS_PATH

st.set_page_config(page_title="Model Evaluation | ContextCheck", page_icon="📊", layout="wide")
st.title("Transformer Model Evaluation")

if not RESULTS_PATH.is_file():
    st.warning(
        "No evaluation results are available yet. Train the Transformer with "
        "`python -m src.train_transformer` to generate measured test results."
    )
    st.stop()

results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
result = results.get("transformer")
if not result:
    st.warning("The saved results do not contain a Transformer evaluation yet.")
    st.stop()

st.caption(f"Measured on {result['test_examples']:,} held-out SNLI test examples.")
metric_columns = st.columns(5)
metric_values = [
    ("Accuracy", "accuracy"),
    ("Macro precision", "precision_macro"),
    ("Macro recall", "recall_macro"),
    ("Macro F1", "f1_macro"),
    ("Weighted F1", "f1_weighted"),
]
for column, (label, key) in zip(metric_columns, metric_values):
    column.metric(label, f"{result[key]:.3f}")

confusion = pd.DataFrame(
    result["confusion_matrix"],
    index=[name.title() for name in LABEL_NAMES],
    columns=[name.title() for name in LABEL_NAMES],
)
st.subheader("Confusion matrix")
figure = px.imshow(
    confusion,
    text_auto=True,
    color_continuous_scale="Blues",
    labels={"x": "Predicted label", "y": "True label", "color": "Examples"},
)
st.plotly_chart(figure, use_container_width=True)

st.subheader("Classification report")
report = pd.DataFrame(result["classification_report"]).T
st.dataframe(report.style.format(precision=3), use_container_width=True)
