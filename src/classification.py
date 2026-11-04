"""Milestone 1: run with python src/classification.py from the project root."""

import hashlib
import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline

from PreProcessing import make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_COLUMNS = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
]
ONE_HOT_COLUMNS = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "Agent", "Company", "CustomerType",
]
BINARY_COLUMNS = ["IsRepeatedGuest"]
FEATURES = NUMERICAL_COLUMNS + ONE_HOT_COLUMNS
SCORING = {
    "accuracy": "accuracy", "precision": "precision", "recall": "recall",
    "f1": "f1", "roc_auc": "roc_auc", "average_precision": "average_precision",
}


def load_data():
    df = pd.read_csv(ROOT / "data/H2.csv", dtype={"Agent": "string", "Company": "string"})
    # Only trim raw input and recognize missing markers here. All learned
    # imputation, encoding and scaling use the team's existing preprocessor.
    for column in ONE_HOT_COLUMNS:
        values = df[column].astype("string").str.strip()
        values = values.mask(values.str.upper().isin(["", "NULL", "NAN", "NONE", "N/A"]))
        if column in ("Agent", "Company"):
            values = values.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
        df[column] = values.astype(object).where(values.notna(), np.nan)
    if df["IsCanceled"].isna().any() or not df["IsCanceled"].isin([0, 1]).all():
        raise ValueError("IsCanceled must contain only 0 or 1.")
    return df


def build_model():
    return Pipeline([
        ("preprocessor", make_preprocessor(
            numerical_columns=[c for c in NUMERICAL_COLUMNS if c not in BINARY_COLUMNS],
            one_hot_columns=ONE_HOT_COLUMNS,
            binary_columns=BINARY_COLUMNS,
        )),
        # Untuned default C=1, L2 and lbfgs; extra iterations allow convergence.
        ("model", LogisticRegression(max_iter=3000, random_state=42)),
    ])


def main():
    df = load_data()
    X, y = df[FEATURES], df["IsCanceled"]
    # Keep the original feature order and seeds so the reserved test is unchanged.
    # Identical inputs stay together, even if their targets differ.
    groups = pd.util.hash_pandas_object(X, index=False).to_numpy()
    development, test = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42,
    ).split(X, groups=groups))
    train, validation = next(GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=43,
    ).split(X.iloc[development], groups=groups[development]))
    train, validation = development[train], development[validation]
    X_train, y_train = X.iloc[train], y.iloc[train]
    X_validation, y_validation = X.iloc[validation], y.iloc[validation]

    model = build_model()
    print("Running training-only 3-fold CV and fitting Logistic Regression...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        cv = cross_validate(
            model, X_train, y_train, groups=groups[train], cv=GroupKFold(3),
            scoring=SCORING, error_score="raise",
        )
        model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_validation, predictions),
        "precision": precision_score(y_validation, predictions, zero_division=0),
        "recall": recall_score(y_validation, predictions, zero_division=0),
        "f1": f1_score(y_validation, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_validation, probabilities),
        "average_precision": average_precision_score(y_validation, probabilities),
    }
    results = pd.DataFrame({
        "validation": metrics,
        "training_cv_mean": {name: cv[f"test_{name}"].mean() for name in SCORING},
        "training_cv_std": {name: cv[f"test_{name}"].std(ddof=1) for name in SCORING},
    })
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    results.to_csv(output / "classification_metrics.csv", index_label="metric")
    ConfusionMatrixDisplay.from_predictions(
        y_validation, predictions, display_labels=["Not canceled", "Canceled"],
        cmap="Blues", colorbar=False,
    )
    plt.title("Logistic Regression — Validation")
    plt.tight_layout()
    plt.savefig(output / "classification_confusion_matrix.png", dpi=160)
    plt.close()

    # Local artifacts only: retain split membership for Milestone 2; never score test.
    artifacts = ROOT / "artifacts/classification"
    artifacts.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts / "logistic_regression.joblib")
    manifest = pd.DataFrame({"row_index": np.arange(len(df)), "input_group": groups, "partition": ""})
    for name, rows in {"train": train, "validation": validation, "test": test}.items():
        manifest.loc[rows, "partition"] = name
    manifest.to_csv(artifacts / "split_manifest.csv", index=False)
    metadata = {
        "dataset_sha256": hashlib.sha256((ROOT / "data/H2.csv").read_bytes()).hexdigest(),
        "features": FEATURES, "seed": 42, "threshold": 0.5, "test_evaluated": False,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "preprocessing": "src/PreProcessing.py; call load_data() for raw CSV cleanup",
    }
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(results.round(4))
    print(f"Train: {len(train)} | Validation: {len(validation)} | Reserved test: {len(test)}")
    print("Final test not evaluated. Results saved in outputs/.")


if __name__ == "__main__":
    main()
