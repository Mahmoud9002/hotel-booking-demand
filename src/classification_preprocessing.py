"""Booking-input selection and normalization shared by training and inference."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber", "ArrivalDateDayOfMonth",
    "StaysInWeekendNights", "StaysInWeekNights", "Adults", "Children", "Babies",
    "IsRepeatedGuest", "PreviousCancellations", "PreviousBookingsNotCanceled",
]
CATEGORICAL_FEATURES = [
    "ArrivalDateMonth", "Meal", "Country", "MarketSegment", "DistributionChannel",
    "ReservedRoomType", "Agent", "Company", "CustomerType",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXCLUDED_FEATURES = {
    "IsCanceled": "Target, never an input.",
    "ReservationStatus": "Final booking outcome directly reveals the target.",
    "ReservationStatusDate": "Date of a later status update, including cancellation.",
    "AssignedRoomType": "Actual allocation may be determined after booking.",
    "BookingChanges": "Cumulative amendments after the reservation was created.",
    "DaysInWaitingList": "Elapsed waiting time is not known at initial booking.",
    "DepositType": "Conservative exclusion: collected-deposit status lacks a verified initial snapshot.",
    "ADR": "Recorded average rate is not a verified initial price quote.",
    "RequiredCarParkingSpaces": "Conservative exclusion: requests can change after booking.",
    "TotalOfSpecialRequests": "Conservative exclusion: requests can accumulate after booking.",
}


class BookingFeatures(TransformerMixin, BaseEstimator):
    """Stateless normalization; no statistics are learned from held-out rows.

    All selected input columns are required, but missing cell values are allowed.
    Unselected columns (including targets and outcomes) are always ignored.
    """

    def fit(self, X, y=None):
        self.transform(X)  # Fail early on malformed input.
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Booking input must be a pandas DataFrame with named columns.")
        missing = sorted(set(FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"Missing booking columns: {', '.join(missing)}")
        result = X.loc[:, FEATURES].copy()
        for column in NUMERIC_FEATURES:
            original = result[column].replace(r"^\s*$", np.nan, regex=True)
            numeric = pd.to_numeric(original, errors="coerce")
            if (original.notna() & numeric.isna()).any():
                raise ValueError(f"Non-numeric value in {column}")
            if np.isinf(numeric.to_numpy(dtype=float)).any():
                raise ValueError(f"Infinite value in {column}")
            result[column] = numeric
        for column in CATEGORICAL_FEATURES:
            values = result[column].astype("string").str.strip()
            values = values.mask(values.str.upper().isin(["", "NULL", "NAN", "NONE", "N/A"]))
            if column in ("Agent", "Company"):
                # CSV readers can infer 9 as 9.0 when some identifiers are missing.
                values = values.str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
            result[column] = values.astype(object).where(values.notna(), np.nan)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)


def build_pipeline(seed=42):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return Pipeline([
        ("booking", BookingFeatures()),
        ("preprocess", ColumnTransformer([
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ], sparse_threshold=1.0)),
        # C=1, L2 penalty, no class weighting or parameter search. max_iter is
        # a numerical convergence budget, not a validation-selected setting.
        ("model", LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000, random_state=seed)),
    ])
