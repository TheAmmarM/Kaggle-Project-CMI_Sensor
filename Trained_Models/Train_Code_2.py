import pandas as pd

# Load full dataset (if your RAM can handle it)
df = pd.read_csv("train.csv")

# Drop non-numeric or metadata columns that can't be averaged
non_numeric_cols = ['row_id', 'sequence_type', 'sequence_counter', 'subject', 'phase', 'gesture']
numeric_cols = df.drop(columns=non_numeric_cols).select_dtypes(include='number').columns.tolist()

# Group by sequence_id and compute statistical features
aggregated = df.groupby("sequence_id")[numeric_cols].agg(['mean', 'std', 'min', 'max'])
aggregated.columns = ['_'.join(col) for col in aggregated.columns]
aggregated.reset_index(inplace=True)

# Add gesture labels for each sequence_id
labels = df.groupby("sequence_id")["gesture"].first().reset_index()
processed_df = pd.merge(aggregated, labels, on="sequence_id")

print(processed_df.head())
print("✅ Shape:", processed_df.shape)
processed_df.to_parquet("processed_data.parquet")

