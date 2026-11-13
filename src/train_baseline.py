"""Fit and evaluate a TF-IDF + Logistic Regression SNLI baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.dataset_loader import load_snli
from src.evaluation import compute_classification_results, save_results
from src.preprocessing import select_train_subset


def make_pair_features(premises, hypotheses, vectorizer, fit: bool):
    """Represent premise, hypothesis, and their sparse TF-IDF difference."""
    if fit:
        vectorizer.fit(list(premises) + list(hypotheses))
    premise_vectors = vectorizer.transform(premises)
    hypothesis_vectors = vectorizer.transform(hypotheses)
    difference = abs(premise_vectors - hypothesis_vectors)
    return hstack(
        (premise_vectors, hypothesis_vectors, difference), format="csr"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("models/baseline"))
    parser.add_argument(
        "--results", type=Path, default=Path("artifacts/evaluation_results.json")
    )
    args = parser.parse_args()

    dataset, preprocessing_stats = load_snli()
    train = select_train_subset(dataset["train"], args.max_train_samples, args.seed)
    test = dataset["test"]

    vectorizer = TfidfVectorizer(
        max_features=50_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    x_train = make_pair_features(
        train["premise"], train["hypothesis"], vectorizer, fit=True
    )
    x_test = make_pair_features(
        test["premise"], test["hypothesis"], vectorizer, fit=False
    )
    classifier = LogisticRegression(
        max_iter=1000, class_weight=None, random_state=args.seed
    )
    classifier.fit(x_train, train["label"])
    predictions = classifier.predict(x_test)

    result = compute_classification_results(test["label"], predictions)
    result.update(
        {
            "model": "tfidf_logistic_regression",
            "test_examples": len(test),
            "train_examples": len(train),
            "preprocessing": preprocessing_stats,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "classifier": classifier},
        args.output_dir / "baseline.joblib",
    )
    metadata_path = args.output_dir / "results.json"
    save_results(metadata_path, result)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    if args.results.exists():
        combined = json.loads(args.results.read_text(encoding="utf-8"))
    else:
        combined = {}
    combined["baseline"] = result
    save_results(args.results, combined)
    print(f"Baseline test accuracy: {result['accuracy']:.4f}")
    print(f"Saved baseline artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
