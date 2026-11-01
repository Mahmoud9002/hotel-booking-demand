
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder


class IQRCapper(BaseEstimator, TransformerMixin):

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):

        X = pd.DataFrame(X)

        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)

        IQR = Q3 - Q1

        self.lower_bounds_ = Q1 - (self.factor * IQR)
        self.upper_bounds_ = Q3 + (self.factor * IQR)

        return self

    def transform(self, X):

        X = pd.DataFrame(X).copy()

        X = X.clip(
            lower=self.lower_bounds_,
            upper=self.upper_bounds_,
            axis=1
        )

        return X


def make_preprocessor(
    numerical_columns,
    one_hot_columns,
    binary_columns,
    outlier_columns=None
):

    if outlier_columns is None:
        outlier_columns = []


    numerical_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    # Apply IQR capping only to selected columns
    if outlier_columns:
        numerical_steps.append(
            ("outlier_handler", IQRCapper())
        )

    numerical_steps.append(
        ("scaler", RobustScaler())
    )

    numerical_pipeline = Pipeline(numerical_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("one_hot", categorical_pipeline, one_hot_columns),
        ("binary", binary_pipeline, binary_columns)
    ])

    return preprocessor
