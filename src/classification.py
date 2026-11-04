"""Milestone 1: untuned cancellation baseline; the final test is never scored.

Run from the repository root: python -m src.classification train
"""

import argparse
import hashlib
import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate

from src.classification_preprocessing import (
    BookingFeatures, EXCLUDED_FEATURES, FEATURES, build_pipeline,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
THRESHOLD = 0.5
SCORING = {
    "accuracy": "accuracy", "precision": "precision", "recall": "recall",
    "f1": "f1", "roc_auc": "roc_auc", "average_precision": "average_precision",
}


def read_bookings(path, require_target=False):
    # Read IDs as strings so leading zeros, if present, are preserved.
    frame = pd.read_csv(path, dtype={"Agent": "string", "Company": "string"})
    BookingFeatures().transform(frame)
    if require_target:
        required = set(FEATURES) | set(EXCLUDED_FEATURES)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"H2 dataset missing columns: {sorted(missing)}")
        if frame["IsCanceled"].isna().any() or not frame["IsCanceled"].isin([0, 1]).all():
            raise ValueError("IsCanceled must contain only 0 or 1, without missing values.")
    return frame


def grouped_split(frame, seed=SEED):
    # Group by ALL selected inputs, without using labels. Preserve repeated
    # bookings but prevent identical feature vectors from crossing partitions.
    canonical = BookingFeatures().transform(frame)
    groups = pd.util.hash_pandas_object(canonical, index=False).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    development, test = next(splitter.split(canonical, groups=groups))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
    train, validation = next(splitter.split(canonical.iloc[development], groups=groups[development]))
    partitions = {"train": development[train], "validation": development[validation], "test": test}
    if any(len(indices) == 0 for indices in partitions.values()):
        raise ValueError("Not enough independent booking groups to make all three splits.")
    return partitions, groups


def evaluate(y_true, probabilities):
    predictions = (np.asarray(probabilities) >= THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plot_validation(y, probabilities, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(
        y, probabilities >= THRESHOLD, display_labels=["Not canceled", "Canceled"],
        ax=axes[0], colorbar=False, cmap="Blues",
    )
    axes[0].set_title("Validation confusion matrix | threshold 0.50")
    RocCurveDisplay.from_predictions(y, probabilities, ax=axes[1], plot_chance_level=True)
    axes[1].set_title("Validation ROC")
    PrecisionRecallDisplay.from_predictions(y, probabilities, ax=axes[2], plot_chance_level=True)
    axes[2].set_title("Validation precision–recall")
    fig.suptitle("H2 cancellation classification • untuned Logistic Regression", fontweight="bold")
    fig.savefig(output / "validation_plots.png", dpi=170)
    plt.close(fig)


def write_report(output, metadata, results, cv):
    score = results["logistic_regression"]
    dummy = results["dummy_prior"]
    rows = []
    for metric in SCORING:
        rows.append(f"| {metric} | {score[metric]:.4f} | {dummy[metric]:.4f} | "
                    f"{cv[metric].mean():.4f} ± {cv[metric].std(ddof=1):.4f} |")
    counts = metadata["split_rows"]
    audit = "\n".join(f"| {name} | {reason} |" for name, reason in EXCLUDED_FEATURES.items())
    report = f"""# Milestone 1 — Cancellation classification

Contributor: **Omar Momen Ahmed — 24p0036**. Scope: the team's classification baseline only.
This is an automatically generated experiment report, not the complete team submission.

## Task and prediction point

Predict `IsCanceled` (1 = eventually canceled, 0 = not canceled) when the initial
reservation is recorded. Inputs represent initial reservation details and prior
guest history. H2 is a historical extract, not a versioned booking-time snapshot:
retained reservation details are assumed available at creation and may have been
amended subsequently. This residual timing risk needs team review; these results
are not a claim of production performance.

## Dataset and reproducible split

- Source: `data/H2.csv`, {metadata['dataset_rows']:,} rows, {metadata['dataset_columns']} columns.
- SHA-256: `{metadata['dataset_sha256']}`.
- Exact duplicate source rows: {metadata['exact_duplicate_rows']:,}.
- Distinct selected-input groups: {metadata['input_groups']:,}.
- Seed: {metadata['seed']}. Two GroupShuffleSplit steps reserve 20% of groups for
  the final test, then 25% of remaining groups for validation (about 60/20/20 overall).
- Actual row counts: train **{counts['train']:,}**, validation **{counts['validation']:,}**,
  reserved test **{counts['test']:,}**. Group sizes differ, so row percentages need not be exact.
- Cancellation fraction: training {metadata['train_cancellation_fraction']:.2%},
  validation {metadata['validation_cancellation_fraction']:.2%}.
- Duplicate-looking reservations are retained because separate group bookings can
  legitimately match. Identical normalized input vectors stay in one partition;
  the label is never used to construct these groups. Groups are also respected in CV.
- Three-fold GroupKFold uses **training rows only**. Each fold fits its own complete
  preprocessing pipeline. The model evaluated on validation is fitted only on train.
- The final test is reserved, never passed to model fitting, tuning, prediction,
  evaluation, or result plots. Its membership/count is recorded for Milestone 2.
- This is a random grouped split, not a chronological evaluation. Without customer
  identifiers it cannot guarantee separation of all bookings by the same guest.

## Preprocessing and baseline

21 inputs: 12 numeric and 9 categorical. Numeric gaps use training medians followed
by StandardScaler. Strings are trimmed; NULL/empty markers become missing.
Categorical gaps receive a Missing category; one-hot encoding ignores unseen values.
Agent and Company are categorical IDs, not numeric measurements. No global outlier
clipping, resampling, target encoding, PCA, LDA, or hyperparameter search is performed.
Unusual but valid values are retained for this baseline; train_numeric_summary.csv
supports later review without consulting test data.

Logistic Regression: L2, C=1.0, lbfgs, no class weighting, fixed 0.50 threshold,
max_iter=3000 as a convergence budget. Despite its name, this is a binary classifier.
A prior-probability DummyClassifier is a sanity comparison, not a second learned model family.
The saved pipeline contains normalization, fitted preprocessing and the classifier.
Convergence warnings fail the run rather than silently accepting an unfinished fit.

## Preliminary results — validation only

| Metric | Logistic Regression | Dummy prior | Training CV mean ± sample SD |
|---|---:|---:|---:|
{chr(10).join(rows)}

Positive class is cancellation. `average_precision` summarizes the precision–recall
curve (AP); it is not the trapezoidal PR area. The dummy uses the training cancellation
fraction for every booking. No model or threshold was selected using these results.

Confusion matrix: rows = actual [0, 1], columns = predicted [0, 1]:
`{score['confusion_matrix']}`.

The model's validation F1 is {score['f1']:.3f} with recall {score['recall']:.3f}:
it identifies {score['recall']:.1%} of canceled bookings at the fixed threshold.
Precision {score['precision']:.3f} means {score['precision']:.1%} of its cancellation
alerts are correct on validation. ROC-AUC {score['roc_auc']:.3f} measures ranking,
not the percentage of correct predictions. Compare against the dummy and CV spread;
this preliminary score does not estimate future-time or final-test performance.

![Validation results](validation_plots.png)

## Leakage audit — excluded source columns

| Column | Reason |
|---|---|
{audit}

Retained features: {', '.join(f'`{x}`' for x in FEATURES)}.
Calendar and planned-stay fields describe the intended arrival; guest counts, meal,
room and sales-channel fields describe the reservation; repeat-guest and previous-
booking counts describe prior history. The initial-snapshot assumption above applies.
`feature_audit.csv` records a decision for every source column.

## Files, contribution and next milestone

- Implemented classification CLI, train-only preprocessing, duplicate-aware split,
  training CV, validation metrics/plots, schema checks and reusable prediction artifact.
- Experiment outputs are in this directory. `artifacts/classification/` stores the
  model, metadata and split manifest locally and is excluded from Git/submission ZIP.
- Reproduce with `python -m src.classification train` from the repository root.
- Predict without retraining with `python -m src.classification predict --input examples/booking.csv`.
- Milestone 2: compare additional families, tune only inside development data, review
  temporal validation and snapshot availability, assess PCA/LDA appropriately,
  integrate inference in the team application, then evaluate the locked test once.
- Coordinate split policy with regression before the final team submission. This
  change does not certify the other tasks or complete the team's progress report.

## References

- Course project description, Milestone 1 checklist (page 10).
- [Original dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/).
- [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).
- [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
"""
    (output / "report.md").write_text(report, encoding="utf-8")


def train(data_path, output, artifacts, seed=SEED):
    frame = read_bookings(data_path, require_target=True)
    splits, groups = grouped_split(frame, seed)
    train_rows, validation_rows = splits["train"], splits["validation"]
    X_train = frame.iloc[train_rows][FEATURES]
    X_validation = frame.iloc[validation_rows][FEATURES]
    y_train = frame.iloc[train_rows]["IsCanceled"].astype(int)
    y_validation = frame.iloc[validation_rows]["IsCanceled"].astype(int)
    if y_train.nunique() != 2 or y_validation.nunique() != 2:
        raise ValueError("Training and validation must each contain both classes.")
    cv_splitter = GroupKFold(n_splits=3)
    for fit_rows, score_rows in cv_splitter.split(X_train, y_train, groups[train_rows]):
        if y_train.iloc[fit_rows].nunique() != 2 or y_train.iloc[score_rows].nunique() != 2:
            raise ValueError("Each training CV fold must contain both classes.")
    model = build_pipeline(seed)
    print("Running 3-fold grouped cross-validation on training rows...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        cv_results = cross_validate(
            model, X_train, y_train, groups=groups[train_rows], cv=cv_splitter,
            scoring=SCORING, n_jobs=1, error_score="raise",
        )
        print("Fitting baseline on training rows and evaluating validation...", flush=True)
        model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_validation)[:, 1]
    dummy = DummyClassifier(strategy="prior").fit(np.zeros((len(y_train), 1)), y_train)
    results = {
        "logistic_regression": evaluate(y_validation, probabilities),
        "dummy_prior": evaluate(y_validation, dummy.predict_proba(np.zeros((len(y_validation), 1)))[:, 1]),
    }
    output.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    cv = pd.DataFrame({name: cv_results[f"test_{name}"] for name in SCORING})
    cv.index = pd.RangeIndex(1, 4, name="fold")
    cv.to_csv(output / "cross_validation.csv")
    write_json(output / "validation_metrics.json", results)
    pd.DataFrame({name: {k: v for k, v in scores.items() if k != "confusion_matrix"}
                  for name, scores in results.items()}).T.to_csv(output / "validation_metrics.csv", index_label="model")
    metadata = {
        "seed": seed, "threshold": THRESHOLD, "target": "IsCanceled", "positive_class": 1,
        "prediction_point": "initial reservation creation (historical-snapshot caveat in report)",
        "features": FEATURES, "excluded_features": EXCLUDED_FEATURES,
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "dataset_rows": len(frame), "dataset_columns": len(frame.columns),
        "exact_duplicate_rows": int(frame.duplicated().sum()), "input_groups": int(len(np.unique(groups))),
        "split_rows": {name: len(rows) for name, rows in splits.items()},
        "split_groups": {name: len(np.unique(groups[rows])) for name, rows in splits.items()},
        "train_cancellation_fraction": float(y_train.mean()),
        "validation_cancellation_fraction": float(y_validation.mean()),
        "test_evaluated": False, "fit_partition": "train", "cv": "3-fold GroupKFold on train only",
        "model": "LogisticRegression", "C": 1.0, "solver": "lbfgs", "max_iter": 3000,
        "iterations_used": int(model.named_steps["model"].n_iter_[0]),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }
    write_json(output / "experiment.json", metadata)
    write_json(artifacts / "metadata.json", metadata)
    manifest = pd.DataFrame({"row_index": np.arange(len(frame)), "input_group": groups, "partition": ""})
    for name, rows in splits.items():
        manifest.loc[rows, "partition"] = name
    manifest.to_csv(artifacts / "split_manifest.csv", index=False)
    joblib.dump(model, artifacts / "logistic_regression.joblib")
    normalized_train = BookingFeatures().transform(X_train)
    normalized_train.describe().T.to_csv(output / "train_numeric_summary.csv")
    normalized_train.isna().sum().rename("missing_train_rows").to_csv(output / "train_missing_values.csv")
    pd.DataFrame([
        {"column": column, "decision": "include" if column in FEATURES else "exclude",
         "reason": EXCLUDED_FEATURES.get(column, "Assumed initial reservation input or prior guest history; see snapshot caveat.")}
        for column in frame.columns
    ]).to_csv(output / "feature_audit.csv", index=False)
    plot_validation(y_validation, probabilities, output)
    write_report(output, metadata, results, cv)
    print(json.dumps({"validation": results["logistic_regression"], "test_evaluated": False,
                      "report": str(output / "report.md")}, indent=2), flush=True)
    return metadata, results


def predict(input_path, model_path, output_path):
    # Only load trusted local artifacts: joblib/pickle can execute code.
    frame = read_bookings(input_path)
    model = joblib.load(model_path)
    probabilities = model.predict_proba(frame)[:, 1]
    predictions = pd.DataFrame({
        "row_index": np.arange(len(frame)), "cancellation_probability": probabilities,
        "predicted_IsCanceled": (probabilities >= THRESHOLD).astype(int),
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(predictions.to_string(index=False))
    print(f"Saved predictions: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Untuned training CV and validation; never scores test")
    training.add_argument("--data", type=Path, default=ROOT / "data/H2.csv")
    training.add_argument("--output", type=Path, default=ROOT / "outputs/classification")
    training.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/classification")
    inference = commands.add_parser("predict", help="Predict using a saved pipeline, without retraining")
    inference.add_argument("--input", type=Path, required=True)
    inference.add_argument("--model", type=Path, default=ROOT / "artifacts/classification/logistic_regression.joblib")
    inference.add_argument("--output", type=Path, default=ROOT / "outputs/classification/predictions.csv")
    args = parser.parse_args()
    try:
        if args.command == "train":
            train(args.data, args.output, args.artifacts)
        else:
            predict(args.input, args.model, args.output)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
