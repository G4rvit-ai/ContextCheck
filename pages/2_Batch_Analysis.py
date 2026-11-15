"""CSV upload and batch NLI prediction."""

import pandas as pd
import plotly.express as px
import streamlit as st

from src.inference import NliPredictor
from src.ui import inject_styles, render_sidebar
from src.utils import MODEL_DIR, batch_predict_dataframe

st.set_page_config(page_title="Batch Analysis | ContextCheck", page_icon="📄", layout="wide")
inject_styles()
render_sidebar()
st.title("Batch Analysis")
st.write("Upload a CSV with `premise` and `hypothesis` columns.")
st.caption("Blank or missing text rows are retained and marked invalid in the output.")

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
st.download_button(
    "Download CSV template",
    data="premise,hypothesis\n\"A person is reading.\",\"Someone is reading a book.\"\n",
    file_name="contextcheck_template.csv",
    mime="text/csv",
)
if uploaded is not None:
    if uploaded.size > 10 * 1024 * 1024:
        st.error("The CSV is larger than the 10 MB limit.")
    else:
        try:
            frame = pd.read_csv(uploaded)
            if len(frame) > 10_000:
                st.error("The CSV exceeds the 10,000-row limit.")
            elif not {"premise", "hypothesis"}.issubset(frame.columns):
                st.error("CSV must include columns named `premise` and `hypothesis`.")
            else:
                @st.cache_resource
                def load_predictor(model_path):
                    return NliPredictor(model_path)

                with st.spinner(f"Analyzing {len(frame):,} rows..."):
                    predictor = load_predictor(str(MODEL_DIR))
                    chunks = [
                        batch_predict_dataframe(frame.iloc[start : start + 128], predictor)
                        for start in range(0, len(frame), 128)
                    ]
                    output = (
                        pd.concat(chunks, ignore_index=True)
                        if chunks
                        else batch_predict_dataframe(frame, predictor)
                    )
                st.dataframe(output, use_container_width=True)
                valid_output = output[output["status"] == "ok"]
                invalid_count = len(output) - len(valid_output)
                metric_columns = st.columns(3)
                metric_columns[0].metric("Rows processed", f"{len(output):,}")
                metric_columns[1].metric("Valid predictions", f"{len(valid_output):,}")
                metric_columns[2].metric("Invalid rows", f"{invalid_count:,}")
                if not valid_output.empty:
                    distribution = (
                        valid_output["predicted_label"]
                        .value_counts()
                        .rename_axis("Relationship")
                        .reset_index(name="Rows")
                    )
                    st.subheader("Prediction distribution")
                    st.plotly_chart(
                        px.bar(
                            distribution,
                            x="Relationship",
                            y="Rows",
                            color="Relationship",
                            color_discrete_map={
                                "entailment": "#16856b",
                                "neutral": "#d99b18",
                                "contradiction": "#c4414b",
                            },
                        ),
                        use_container_width=True,
                    )
                st.download_button(
                    "Download results as CSV",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="contextcheck_predictions.csv",
                    mime="text/csv",
                )
        except FileNotFoundError as error:
            st.error(str(error))
        except (ValueError, UnicodeDecodeError, pd.errors.ParserError) as error:
            st.error(f"Could not process this CSV: {error}")
