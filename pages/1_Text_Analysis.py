"""Interactive single-pair NLI analysis."""

import plotly.express as px
import pandas as pd
import streamlit as st

from src.inference import NliPredictor
from src.ui import inject_styles, render_hero, render_label_result, render_sidebar, LABEL_META
from src.utils import MODEL_DIR

st.set_page_config(page_title="Text Analysis | ContextCheck", page_icon="📝", layout="wide")
inject_styles()
render_sidebar()
render_hero(
    "Inference workspace",
    "Read between the lines.",
    "Place a context beside a statement and let ContextCheck map the relationship "
    "with transparent three-class scores.",
)
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
    premise_column, hypothesis_column = st.columns(2, gap="large")
    with premise_column:
        st.markdown(
            '<div class="input-panel"><h3>01&nbsp; Context</h3>',
            unsafe_allow_html=True,
        )
        premise = st.text_area(
            "Premise / Context",
            key="premise_input",
            height=190,
            label_visibility="collapsed",
            placeholder="Paste the passage, context, or evidence here...",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with hypothesis_column:
        st.markdown(
            '<div class="input-panel"><h3>02&nbsp; Statement</h3>',
            unsafe_allow_html=True,
        )
        hypothesis = st.text_area(
            "Hypothesis / Statement",
            key="hypothesis_input",
            height=190,
            label_visibility="collapsed",
            placeholder="Write the statement you want to compare...",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    st.write("")
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
            render_label_result(label, prediction["score"])
            confidence = prediction["score"]
            if confidence >= 0.75:
                st.success("The model is relatively confident in this relationship.")
            elif confidence >= 0.50:
                st.warning("The model sees a leading relationship, but the result is ambiguous.")
            else:
                st.warning("The scores are close; treat this prediction as uncertain.")
            word_count = len((premise + " " + hypothesis).split())
            st.caption(f"Analyzed {word_count} words across the two inputs.")
            scores = pd.DataFrame(
                {
                    "Relationship": [LABEL_META[name]["title"] for name in prediction["scores"]],
                    "Prediction score": list(prediction["scores"].values()),
                }
            )
            st.subheader("Relationship score distribution")
            chart = px.bar(
                scores,
                x="Relationship",
                y="Prediction score",
                color="Relationship",
                color_discrete_map={
                    "Entailment": "#16856b",
                    "Neutral": "#d99b18",
                    "Contradiction": "#c4414b",
                },
                range_y=[0, 1],
                text=scores["Prediction score"].map(lambda value: f"{value:.1%}"),
            )
            chart.update_layout(showlegend=False, yaxis_title="Softmax prediction score")
            st.plotly_chart(chart, use_container_width=True)
            with st.expander("How should I read this?"):
                st.write(
                    "The model compares the meaning of the two supplied texts. "
                    "The scores are softmax outputs from the classification head, "
                    "not guarantees of truth or factual correctness."
                )
            st.caption("All three softmax scores sum to 1; they are not factual-certainty estimates.")
        except FileNotFoundError as error:
            st.error(str(error))
        except (ValueError, RuntimeError, OSError) as error:
            st.error(f"Could not analyze this pair: {error}")
