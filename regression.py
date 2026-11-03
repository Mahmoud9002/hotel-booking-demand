
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import make_preprocessor


df = pd.read_csv("../data/H2.csv")

numerical_columns = [
    "LeadTime", "ArrivalDateYear", "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth", "StaysInWeekendNights",
    "StaysInWeekNights", "Adults", "Children", "Babies",
    "PreviousCancellations", "PreviousBookingsNotCanceled",
    "RequiredCarParkingSpaces", "TotalOfSpecialRequests"
]

one_hot_columns = [
    "Hotel", "ArrivalDateMonth", "Meal", "Country",
    "MarketSegment", "DistributionChannel", "ReservedRoomType",
    "DepositType", "CustomerType"
]

binary_columns = ["IsRepeatedGuest"]

features = numerical_columns + one_hot_columns + binary_columns

X = df[features]
y = df["ADR"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    )
}

results = []

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", make_preprocessor(
            numerical_columns,
            one_hot_columns,
            binary_columns
        )),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append([name, mae, mse, rmse, r2])

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "MSE", "RMSE", "R2"]
)

print(results_df)

results_df.to_csv("../reports/regression_results.csv", index=False)
