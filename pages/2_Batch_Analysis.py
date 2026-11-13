"""CSV upload and batch NLI prediction."""

import pandas as pd
import streamlit as st

from src.inference import NliPredictor
from src.utils import MODEL_DIR, batch_predict_dataframe

st.set_page_config(page_title="Batch Analysis | ContextCheck", page_icon="📄", layout="wide")
st.title("Batch Analysis")
st.write("Upload a CSV with `premise` and `hypothesis` columns.")
st.caption("Blank or missing text rows are retained and marked invalid in the output.")

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
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
