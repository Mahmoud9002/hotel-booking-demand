import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/H2.csv")

print(df.shape)

print(df.head())

df.info()

print(df.describe())

print(df.isnull().sum())

#agent feature cant accept duplicates

print(df.duplicated().sum())

#print(df["IsCanceled"].value_counts())
numeric_columns = df.select_dtypes(include='number').columns

for column in numeric_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[
        (df[column] > upper_bound) |
        (df[column] < lower_bound)
        ]

    print(f"\nColumn: {column}")
    print("Lower bound:", lower_bound)
    print("Upper bound:", upper_bound)
    print("Number of potential outliers:", len(outliers))