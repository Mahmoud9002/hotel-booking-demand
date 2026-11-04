"""Behavioral checks for split leakage, fitted preprocessing and saved inference."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import classification
from src.classification_preprocessing import (
    BookingFeatures, CATEGORICAL_FEATURES, EXCLUDED_FEATURES, FEATURES,
    NUMERIC_FEATURES, build_pipeline,
)


def bookings(n=180):
    rng = np.random.default_rng(17)
    frame = pd.DataFrame({column: rng.integers(0, 10, n) for column in NUMERIC_FEATURES})
    for column in CATEGORICAL_FEATURES:
        frame[column] = rng.choice(["A", "B", " NULL "], n)
    frame["IsCanceled"] = (frame["LeadTime"] > 4).astype(int)
    for column in EXCLUDED_FEATURES:
        if column != "IsCanceled":
            frame[column] = "irrelevant outcome"
    return frame


class ClassificationTests(unittest.TestCase):
    def test_duplicate_groups_never_cross_splits_or_cv(self):
        original = bookings()
        # Include identical inputs with opposite targets; target must not define groups.
        repeated = original.iloc[:30].copy()
        repeated["IsCanceled"] = 1 - repeated["IsCanceled"]
        frame = pd.concat([original, repeated], ignore_index=True)
        splits, groups = classification.grouped_split(frame)
        again, _ = classification.grouped_split(frame)
        all_rows = np.concatenate(list(splits.values()))
        self.assertEqual(sorted(all_rows.tolist()), list(range(len(frame))))
        self.assertEqual(len(set(all_rows)), len(frame))
        for name in splits:
            np.testing.assert_array_equal(splits[name], again[name])
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            self.assertFalse(set(groups[splits[a]]) & set(groups[splits[b]]))
        np.testing.assert_array_equal(groups[:30], groups[-30:])
        train = splits["train"]
        for fit, score in GroupKFold(3).split(frame.iloc[train], groups=groups[train]):
            self.assertFalse(set(groups[train[fit]]) & set(groups[train[score]]))

    def test_normalizes_padded_ids_and_nulls(self):
        frame = bookings(3)
        frame["Agent"] = [" 9 ", 9.0, " NULL "]
        result = BookingFeatures().transform(frame)
        self.assertEqual(result["Agent"].tolist()[:2], ["9", "9"])
        self.assertTrue(pd.isna(result["Agent"].iloc[2]))
        self.assertEqual(list(result.columns), FEATURES)

    def test_preprocessing_learns_only_training_statistics(self):
        frame = bookings()
        frame["LeadTime"] = np.arange(len(frame), dtype=float)
        frame.loc[0, "LeadTime"] = np.nan
        model = build_pipeline().fit(frame, frame["IsCanceled"])
        numeric = model.named_steps["preprocess"].named_transformers_["numeric"]
        imputer = numeric.named_steps["imputer"]
        self.assertEqual(imputer.statistics_[0], frame["LeadTime"].median())
        old_means = numeric.named_steps["scaler"].mean_.copy()
        old_medians = imputer.statistics_.copy()
        unseen = frame.iloc[:2].copy()
        unseen["LeadTime"] = [1e9, np.nan]
        unseen["Country"] = "NEVER_SEEN_IN_TRAINING"
        probabilities = model.predict_proba(unseen)
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_array_equal(old_means, numeric.named_steps["scaler"].mean_)
        np.testing.assert_array_equal(old_medians, imputer.statistics_)

    def test_outcomes_and_target_cannot_change_predictions(self):
        frame = bookings()
        model = build_pipeline().fit(frame, frame["IsCanceled"])
        baseline = model.predict_proba(frame)
        altered = frame.copy()
        for column in EXCLUDED_FEATURES:
            altered[column] = "tampered"
        np.testing.assert_array_equal(baseline, model.predict_proba(altered))

    def test_schema_and_invalid_numeric_values_fail_clearly(self):
        frame = bookings()
        with self.assertRaisesRegex(ValueError, "Missing booking columns: LeadTime"):
            BookingFeatures().transform(frame.drop(columns="LeadTime"))
        frame["LeadTime"] = "invalid"
        with self.assertRaisesRegex(ValueError, "Non-numeric value in LeadTime"):
            BookingFeatures().transform(frame)

    def test_saved_model_reuses_same_preprocessing(self):
        frame = bookings()
        model = build_pipeline().fit(frame, frame["IsCanceled"])
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            input_path = Path(directory) / "booking.csv"
            output_path = Path(directory) / "predictions.csv"
            joblib.dump(model, model_path)
            frame[FEATURES].to_csv(input_path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                classification.predict(input_path, model_path, output_path)
            predictions = pd.read_csv(output_path)
            np.testing.assert_allclose(predictions["cancellation_probability"], model.predict_proba(frame)[:, 1])

    def test_training_workflow_only_fits_train_and_scores_validation(self):
        frame = bookings()
        splits, _ = classification.grouped_split(frame)
        model = build_pipeline()
        original_fit, original_predict = model.fit, model.predict_proba
        original_cv, original_evaluate = classification.cross_validate, classification.evaluate

        def fit(X, y):
            self.assertEqual(X.index.tolist(), splits["train"].tolist())
            return original_fit(X, y)

        def predict(X):
            self.assertEqual(X.index.tolist(), splits["validation"].tolist())
            return original_predict(X)

        def cv(estimator, X, y, **kwargs):
            self.assertEqual(X.index.tolist(), splits["train"].tolist())
            return original_cv(estimator, X, y, **kwargs)

        def evaluate(y, probabilities):
            self.assertEqual(y.index.tolist(), splits["validation"].tolist())
            return original_evaluate(y, probabilities)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            data_path = directory / "H2.csv"
            frame.to_csv(data_path, index=False)
            with (
                patch.object(classification, "build_pipeline", return_value=model),
                patch.object(model, "fit", side_effect=fit),
                patch.object(model, "predict_proba", side_effect=predict),
                patch.object(classification, "cross_validate", side_effect=cv),
                patch.object(classification, "evaluate", side_effect=evaluate),
                patch.object(classification, "plot_validation"),
                patch.object(classification.joblib, "dump"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                metadata, _ = classification.train(data_path, directory / "output", directory / "artifacts")
            self.assertFalse(metadata["test_evaluated"])
            manifest = pd.read_csv(directory / "artifacts/split_manifest.csv")
            self.assertEqual(set(manifest.partition), {"train", "validation", "test"})
            self.assertNotIn("IsCanceled", manifest.columns)


if __name__ == "__main__":
    unittest.main()
