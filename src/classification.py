import joblib
from PreProcessing import *
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("../data/H2.csv")

col_drop = ['IsCanceled', 'Hotel', 'ReservationStatus', 'ReservationStatusDate', 'Agent', 'Country', 'Company', 'ADR'
            ,'ArrivalDateYear']
features = df.drop(columns=col_drop, errors="ignore")
target = df["IsCanceled"]

numerical_columns = features.select_dtypes(include="number").columns.tolist()
categorical_columns = features.select_dtypes(exclude="number").columns.tolist()
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=42)
preprocessor = make_preprocessor(numerical_columns, categorical_columns)
logistic_regression = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
logistic_regression.fit(features_train, target_train)

target_pred = logistic_regression.predict(features_test)

acc = accuracy_score(target_test, target_pred)
precision = precision_score(target_test, target_pred)
recall = recall_score(target_test, target_pred)
f1 = f1_score(target_test, target_pred)

print("Accuracy: ",acc)
print("Precision: ",precision)
print("Recall: ",recall)
print("F1: ",f1)

joblib.dump(logistic_regression, "../artifacts/classification_pipeline.joblib")

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1"],
    "Score": [acc, precision, recall, f1]
})
metrics_df.to_csv("../outputs/classification_metrics.csv", index=False)