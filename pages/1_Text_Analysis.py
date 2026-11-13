"""Interactive single-pair NLI analysis."""

import plotly.express as px
import pandas as pd
import streamlit as st

from src.inference import NliPredictor
from src.utils import MODEL_DIR

st.set_page_config(page_title="Text Analysis | ContextCheck", page_icon="📝", layout="wide")
st.title("Text Analysis")
st.write("Compare a premise with a hypothesis using the fine-tuned SNLI model.")
st.info(
    "Model scores reflect the model's predictions for this input pair; they are "
    "not guarantees of factual correctness."
)

examples = {
    "World Cup": (
        "India won the Cricket World Cup in 1983.",
        "India won its first Cricket World Cup in 1983.",
    ),
    "Conflicting date": (
        "The company was established in 2010.",
        "The company was established in 2015.",
    ),
    "Unstated detail": (
        "A family is walking through a park.",
        "The family is carrying a red umbrella.",
    ),
}
selected_example = st.selectbox("Try a sample pair", ["Choose an example", *examples])
if st.button("Load example", disabled=selected_example == "Choose an example"):
    st.session_state["premise_input"] = examples[selected_example][0]
    st.session_state["hypothesis_input"] = examples[selected_example][1]

with st.form("nli_analysis"):
    premise = st.text_area(
        "Premise / Context",
        key="premise_input",
        height=150,
        placeholder="Enter the context or passage...",
    )
    hypothesis = st.text_area(
        "Hypothesis / Statement",
        key="hypothesis_input",
        height=120,
        placeholder="Enter the statement to compare...",
    )
    submitted = st.form_submit_button("Analyze Relationship", type="primary")

if submitted:
    if not premise.strip() or not hypothesis.strip():
        st.error("Please provide non-empty text in both fields.")
    else:
        try:
            @st.cache_resource
            def load_predictor(model_path):
                return NliPredictor(model_path)

            with st.spinner("Analyzing the text pair..."):
                prediction = load_predictor(str(MODEL_DIR)).predict(premise, hypothesis)
            label = prediction["label"]
            colors = {
                "entailment": "🟢",
                "contradiction": "🔴",
                "neutral": "🟡",
            }
            st.markdown(f"## {colors[label]} Prediction: {label.upper()}")
            st.metric("Top model score", f"{prediction['score']:.1%}")
            scores = pd.DataFrame(
                {
                    "Relationship": list(prediction["scores"]),
                    "Prediction score": list(prediction["scores"].values()),
                }
            )
            chart = px.bar(
                scores,
                x="Relationship",
                y="Prediction score",
                color="Relationship",
                color_discrete_map={
                    "entailment": "#16856b",
                    "neutral": "#d99b18",
                    "contradiction": "#c4414b",
                },
                range_y=[0, 1],
                text=scores["Prediction score"].map(lambda value: f"{value:.1%}"),
            )
            chart.update_layout(showlegend=False, yaxis_title="Softmax prediction score")
            st.plotly_chart(chart, use_container_width=True)
            st.caption("All three softmax scores sum to 1; they are not factual-certainty estimates.")
        except FileNotFoundError as error:
            st.error(str(error))
        except (ValueError, RuntimeError, OSError) as error:
            st.error(f"Could not analyze this pair: {error}")
