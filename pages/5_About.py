"""Conceptual background, dataset facts, and project limitations."""

import streamlit as st

st.set_page_config(page_title="About | ContextCheck", page_icon="📚", layout="wide")
st.title("About ContextCheck")

st.markdown(
    """
### Natural Language Processing
Natural Language Processing (NLP) is the area of computing concerned with
processing and modeling human language.

### Natural Language Inference
NLI compares a premise with a hypothesis. The model predicts **entailment** when
the premise supports the hypothesis, **contradiction** when they conflict, and
**neutral** when the premise does not decide the issue.

### Transformers and self-attention
A Transformer processes token representations using attention. Self-attention
lets each token use information from other tokens in the sequence, helping
represent context and long-range relationships. The model also receives the
premise and hypothesis as a paired input.

### DistilBERT and fine-tuning
DistilBERT is a smaller Transformer derived from BERT using knowledge
distillation. Fine-tuning starts from pretrained language representations and
updates the model's parameters on labeled NLI examples for the downstream
three-class task.

### Dataset
The project loads the official train, validation, and test splits of the
Stanford Natural Language Inference (SNLI) corpus through Hugging Face Datasets.
Rows with the unlabeled class (`-1`) or missing/blank sentence text are excluded.
The test split is reserved for final evaluation.

### Limitations
- The model learns patterns in SNLI; it can make incorrect predictions,
  especially for ambiguity, domain shift, uncommon language, or subtle reasoning.
- Prediction scores are softmax outputs, not calibrated measures of truth.
- The result applies only to the supplied pair and is not fact-checking,
  internet search, or verification against external sources.
- Training and evaluation depend on the exact dataset version, preprocessing,
  random seed, compute resources, and package versions.
"""
)
