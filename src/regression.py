import joblib
from PreProcessing import make_preprocessor
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("../data/H2.csv")

col_drop = ['ADR', 'Hotel', 'ReservationStatus', 'ReservationStatusDate', 'Agent', 'Country', 'ArrivalDateYear']
features = df.drop(columns=col_drop, errors="ignore")
target = df["ADR"]

numerical_columns = features.select_dtypes(include="number").columns.tolist()
categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)
preprocessor = make_preprocessor(numerical_columns, categorical_columns)
linear_regression = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
linear_regression.fit(features_train, target_train)

target_pred = linear_regression.predict(features_test)

meanSquared = mean_squared_error(target_test, target_pred)
meanAbsolute = mean_absolute_error(target_test, target_pred)
r2 = r2_score(target_test, target_pred)
print("Mean squared error: " , meanSquared)
print("Mean absolute error: " , meanAbsolute)
print("R2 score: " , r2)

#to save the trained model to the Artifact folder
joblib.dump(linear_regression, "../artifacts/regression_pipeline.joblib")

metrics_df = pd.DataFrame({
    "Metric": ["MSE", "MAE", "R2"],
    "Score": [meanSquared, meanAbsolute, r2]
})
metrics_df.to_csv("../outputs/regression_metrics.csv", index=False)