"""Fine-tune DistilBERT on SNLI and evaluate once on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.dataset_loader import load_snli
from src.evaluation import compute_classification_results, save_results
from src.labels import ID_TO_LABEL, LABEL_NAMES
from src.preprocessing import select_train_subset, tokenize_pairs


MODEL_NAME = "distilbert/distilbert-base-uncased"


def positive_int(value: str) -> int:
    """Parse a positive integer CLI option."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    """Parse a positive floating-point CLI option."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train-samples", type=positive_int, default=None)
    parser.add_argument("--epochs", type=positive_float, default=2)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps", type=positive_int, default=2
    )
    parser.add_argument("--max-length", type=positive_int, default=128)
    parser.add_argument("--learning-rate", type=positive_float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("models/contextcheck_distilbert"))
    parser.add_argument(
        "--results", type=Path, default=Path("artifacts/evaluation_results.json")
    )
    args = parser.parse_args()

    set_seed(args.seed)
    dataset, preprocessing_stats = load_snli()
    train = select_train_subset(dataset["train"], args.max_train_samples, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_train = train.map(
        lambda batch: tokenize_pairs(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=["premise", "hypothesis"],
        desc="Tokenizing training split",
    )
    tokenized_validation = dataset["validation"].map(
        lambda batch: tokenize_pairs(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=["premise", "hypothesis"],
        desc="Tokenizing validation split",
    )
    tokenized_test = dataset["test"].map(
        lambda batch: tokenize_pairs(batch, tokenizer, args.max_length),
        batched=True,
        remove_columns=["premise", "hypothesis"],
        desc="Tokenizing test split",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={index: name.upper() for index, name in ID_TO_LABEL.items()},
        label2id={name.upper(): index for index, name in ID_TO_LABEL.items()},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    def validation_metrics(evaluation):
        logits, gold = evaluation
        predictions = np.argmax(logits, axis=-1)
        measured = compute_classification_results(gold, predictions)
        return {
            "accuracy": measured["accuracy"],
            "precision_macro": measured["precision_macro"],
            "recall_macro": measured["recall_macro"],
            "f1_macro": measured["f1_macro"],
            "f1_weighted": measured["f1_weighted"],
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=validation_metrics,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # Test data is used only after training and best-checkpoint selection finish.
    prediction_output = trainer.predict(tokenized_test, metric_key_prefix="test")
    predictions = np.argmax(prediction_output.predictions, axis=-1)
    result = compute_classification_results(
        prediction_output.label_ids, predictions
    )
    result.update(
        {
            "model": MODEL_NAME,
            "test_examples": len(dataset["test"]),
            "train_examples": len(train),
            "preprocessing": preprocessing_stats,
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_length": args.max_length,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "label_names": list(LABEL_NAMES),
            },
        }
    )
    save_results(args.output_dir / "test_results.json", result)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    if args.results.exists():
        combined = json.loads(args.results.read_text(encoding="utf-8"))
    else:
        combined = {}
    combined["transformer"] = result
    save_results(args.results, combined)
    (Path("artifacts") / "training_config.json").write_text(
        json.dumps(result["training"], indent=2), encoding="utf-8"
    )
    print(f"Transformer test accuracy: {result['accuracy']:.4f}")
    print(f"Saved model and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
