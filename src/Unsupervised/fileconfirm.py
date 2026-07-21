import pandas as pd

df = pd.read_csv("../../data/processed/train.csv")

print(df["HasFailure"].value_counts())

print(df.groupby("HasFailure").mean())