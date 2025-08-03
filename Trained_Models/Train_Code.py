import pandas as pd

# Load only the first few rows to get column names
df = pd.read_csv("train.csv", nrows=5)
print("🧾 Column Names:")
print(df.columns.tolist())

# Use iterator mode for memory-efficient row counting
row_count = sum(1 for row in open("train.csv", 'r')) - 1  # subtract header
print(f"📊 Total Rows: {row_count}")

grouped = df.groupby("sequence_id")
print(len(grouped))  # Number of unique gesture sequences

import pandas as pd

df = pd.read_csv("train.csv", usecols=["sequence_id"])  # only load this column
print("✅ Unique sequences:", df["sequence_id"].nunique())


