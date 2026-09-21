import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    r"D:\ML Git\Regression\placement_predict_50k.csv"
)

print("Dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nSalary Package statistics:")
print(df["Salary Package"].describe())