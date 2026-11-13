# ContextCheck

ContextCheck is an academic Natural Language Inference (NLI) application. Given
a **premise** and a **hypothesis**, it predicts whether the premise entails,
contradicts, or is neutral toward the hypothesis. It compares only the supplied
texts; it does not search the web or verify claims against external facts.

> **Model artifacts are not included.** The source code and training pipelines
> are complete, but no model has been trained in this workspace. The evaluation
> pages therefore remain empty until you run training. No performance numbers
> are claimed or fabricated.

## Objectives

- Demonstrate paired-text classification with a fine-tuned Transformer.
- Compare contextual DistilBERT representations with a TF-IDF + Logistic
  Regression baseline on the same held-out SNLI test split.
- Provide interactive and CSV batch inference, evaluation visualizations, and
  reproducible training suitable for Google Colab.

## Dataset

The application loads `stanfordnlp/snli` from the Hugging Face Hub using the
`datasets` library. Its official train, validation, and test splits are kept
separate. Expected label IDs are **0 = entailment, 1 = neutral, 2 =
contradiction**. Examples labeled `-1`, missing text, or blank text are removed.
The loader prints the dataset feature structure and per-split counts before and
after cleaning; those counts are also written into training result reports.

Dataset source: [Stanford NLI on Hugging Face](https://huggingface.co/datasets/stanfordnlp/snli).
The Hub dataset card identifies the license as **CC BY-SA 4.0**; check the
current dataset card and follow its attribution and share-alike terms when
redistributing derived datasets or content. Original paper:

> Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). *A large
> annotated corpus for learning natural language inference*. Proceedings of
> EMNLP 2015. https://doi.org/10.18653/v1/D15-1075

For the dataset version inspected while developing this project, the original
split sizes were 550,152 train, 10,000 validation, and 10,000 test examples.
Cleaning retained 549,367 train, 9,842 validation, and 9,824 test examples
(removed 785, 158, and 176 rows, respectively). The loader computes these
counts from the live dataset at runtime; they can change if the upstream
dataset revision changes.

## Preprocessing

`src/dataset_loader.py` inspects and loads the official splits and filters
unlabeled/invalid examples without rewriting sentence content. Transformer
tokenization uses the tokenizer for `distilbert/distilbert-base-uncased`,
truncation at 128 tokens by default, and dynamic padding at batch time. The
deterministic `--max-train-samples` option subsets **only the training split**;
validation and test remain separate. The validation split selects the best
Transformer checkpoint. The test split is evaluated only after training.

## Models and methodology

### Transformer

`src/train_transformer.py` loads the pretrained checkpoint through
`AutoTokenizer` and `AutoModelForSequenceClassification`, configures three
classes, and fine-tunes with Hugging Face `Trainer`. Defaults are two epochs,
learning rate `2e-5`, batch size 8, gradient accumulation 2, weight decay 0.01,
and seed 42. Dynamic padding, validation by epoch, best-checkpoint selection,
and CUDA mixed precision when available are enabled. Training writes the model
and tokenizer to `models/contextcheck_distilbert/`, plus measured test results
to `artifacts/evaluation_results.json`.

The script supports CUDA and CPU training. CPU full-dataset training can take a
long time; use Colab GPU for practical full fine-tuning. On Apple Silicon, local
inference uses MPS when PyTorch reports it available. Training uses CUDA mixed
precision only; it does not assume MPS training performance.

### Traditional baseline

`src/train_baseline.py` creates a shared unigram/bigram TF-IDF vocabulary for
premises and hypotheses, then concatenates three sparse feature groups:
premise vectors, hypothesis vectors, and their absolute vector difference.
Multiclass Logistic Regression is trained on the same cleaned training split
and evaluated on the same untouched test split. Its serialized estimator is
saved under `models/baseline/`.

Neither model's results are prefilled. The app displays only metrics written by
the evaluation scripts.

## Project architecture

```text
app.py                         Streamlit dashboard
pages/                         Text, batch, evaluation, comparison, and about pages
src/dataset_loader.py          Hugging Face SNLI loading and cleaning
src/preprocessing.py           Pair validation, tokenization, subset selection
src/train_transformer.py       DistilBERT fine-tuning and test evaluation
src/train_baseline.py          TF-IDF + Logistic Regression baseline
src/inference.py               Saved model loading and single/batch predictions
src/evaluation.py              Metrics and results serialization
notebooks/contextcheck_colab_training.ipynb
                               Colab GPU workflow
tests/                         Unit tests
sample_batch.csv               Example batch input
```

## Installation

Python 3.10+ is recommended. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch wheels vary by operating system and accelerator. If needed, install the
appropriate PyTorch build first using the official
[PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Training

Inspect and clean SNLI, then train both models on their default full training
split (the validation split is used for Transformer model selection):

```bash
python -m src.train_baseline
python -m src.train_transformer
```

To run a deterministic training subset for a quick experiment:

```bash
python -m src.train_baseline --max-train-samples 20000
python -m src.train_transformer --max-train-samples 20000 --epochs 1
```

The same full test split is still used for evaluation. A subset-trained result
is not directly comparable to a full-training result; the report records the
number of training examples. Do not tune hyperparameters against test results.

### Google Colab GPU training

1. Open `notebooks/contextcheck_colab_training.ipynb` in Colab.
2. Select a GPU runtime.
3. Zip the project source (exclude `.venv`, downloaded data, caches, and model
   weights) and upload that ZIP when prompted.
4. Choose `TRAIN_LIMIT = 20000` for a deterministic pilot, or `None` for all
   valid training examples. The notebook runs the baseline and then
   fine-tunes/evaluates DistilBERT.
5. Optionally mount Drive in the final cell to persist model and result files.

The training scripts download the dataset and the pretrained model on first
use. Plan for substantial disk, network, RAM, and GPU usage for full training.

## Run the application

After training, or after copying trained model artifacts into the expected
`models/` and `artifacts/` directories:

```bash
streamlit run app.py
```

The Text Analysis page takes a premise and hypothesis and displays one of the
three classes and all softmax prediction scores. Batch Analysis accepts CSV
files with `premise` and `hypothesis` columns and offers a results download.
Evaluation and comparison pages explain when measured reports are not yet
available. Model startup does not trigger training. If model files are absent,
inference reports the command needed to train them.

Try `sample_batch.csv` for a batch example.

## Evaluation metrics

The evaluation report records accuracy, macro precision, macro recall, macro
F1, weighted F1, a confusion matrix, and a per-class classification report.
`artifacts/evaluation_results.json` is generated by the training scripts and
ignored by Git. The Streamlit evaluation and comparison pages read this file;
they never synthesize metrics. Model scores in interactive prediction are
softmax outputs and are not guarantees of correctness or calibrated factual
probabilities.

## Tests

Run the focused unit tests with:

```bash
pytest -q
```

Tests cover label mapping, invalid SNLI records, text validation, tokenization
options, missing model behavior, and batch output handling. Dataset download,
full training, and end-to-end UI inference require network access and suitable
model artifacts; they are not silently simulated by these tests.

## Limitations and future work

SNLI is a finite English-language dataset and its annotation patterns do not
cover every domain or reasoning style. A model can be confidently wrong, and
softmax scores may not be calibrated. ContextCheck is not a fact-checker or
web-search tool. Useful future work includes calibration, robustness and
domain-shift analysis, additional NLI benchmarks, and broader accessibility
and multilingual support.

## References

- Hugging Face dataset: https://huggingface.co/datasets/stanfordnlp/snli
- SNLI paper: https://doi.org/10.18653/v1/D15-1075
- DistilBERT model: https://huggingface.co/distilbert/distilbert-base-uncased
- Sanh, V. et al. (2019). *DistilBERT, a distilled version of BERT: smaller,
  faster, cheaper and lighter*. https://arxiv.org/abs/1910.01108
