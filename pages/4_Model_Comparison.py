"""Compare measured Transformer and TF-IDF baseline metrics."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import inject_styles, load_saved_results, render_sidebar
from src.utils import RESULTS_PATH

st.set_page_config(page_title="Model Comparison | ContextCheck", page_icon="⚖️", layout="wide")
inject_styles()
render_sidebar()
st.title("Model Comparison")

if not RESULTS_PATH.is_file():
    st.warning(
        "No measured results are available. Run the baseline and Transformer "
        "training scripts to populate this comparison."
    )
    st.stop()

try:
    results = load_saved_results()
except RuntimeError as error:
    st.error(str(error))
    st.stop()
available = {
    "DistilBERT": results.get("transformer"),
    "TF-IDF + Logistic Regression": results.get("baseline"),
}
available = {name: data for name, data in available.items() if data is not None}
if not available:
    st.warning("No model evaluation results are saved yet.")
    st.stop()

comparison = pd.DataFrame(
    [
        {
            "Model": name,
            "Test examples": data["test_examples"],
            "Accuracy": data["accuracy"],
            "Macro precision": data["precision_macro"],
            "Macro recall": data["recall_macro"],
            "Macro F1": data["f1_macro"],
            "Weighted F1": data["f1_weighted"],
        }
        for name, data in available.items()
    ]
)
st.dataframe(
    comparison.style.format(
        {
            "Accuracy": "{:.3f}",
            "Macro precision": "{:.3f}",
            "Macro recall": "{:.3f}",
            "Macro F1": "{:.3f}",
            "Weighted F1": "{:.3f}",
        }
    ),
    use_container_width=True,
)

accuracy_column, f1_column = st.columns(2)
with accuracy_column:
    st.subheader("Accuracy")
    st.plotly_chart(
        px.bar(comparison, x="Model", y="Accuracy", color="Model", range_y=[0, 1]),
        use_container_width=True,
    )
with f1_column:
    st.subheader("Macro F1")
    st.plotly_chart(
        px.bar(comparison, x="Model", y="Macro F1", color="Model", range_y=[0, 1]),
        use_container_width=True,
    )
st.markdown(
    """
### Why compare these approaches?

TF-IDF represents word and short-phrase counts, so the baseline is efficient and
interpretable but does not directly model contextual meaning or word order.
DistilBERT uses contextual Transformer representations and self-attention to
model interactions across the paired texts. Its advantage, if any, must be
judged from the measured results above; it is not guaranteed for every dataset
or training setup.
"""
)
