from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import pandas as pd

# Load dataset
df = pd.read_csv("train.csv")

# Drop non-numeric columns
non_numeric_cols = ['row_id', 'sequence_type', 'sequence_counter', 'subject', 'phase', 'gesture']
numeric_cols = df.drop(columns=non_numeric_cols).select_dtypes(include='number').columns.tolist()

# Aggregate
aggregated = df.groupby("sequence_id")[numeric_cols].agg(['mean', 'std', 'min', 'max'])
aggregated.columns = ['_'.join(col) for col in aggregated.columns]
aggregated.reset_index(inplace=True)

# Add labels
labels = df.groupby("sequence_id")["gesture"].first().reset_index()
processed_df = pd.merge(aggregated, labels, on="sequence_id")

# 1. Label encode gesture classes
le = LabelEncoder()
processed_df['gesture_encoded'] = le.fit_transform(processed_df['gesture'])

# 2. Prepare features and labels
X = processed_df.drop(columns=['sequence_id', 'gesture', 'gesture_encoded'])
y = processed_df['gesture_encoded']

# 3. Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Train the model
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_val)
f1 = f1_score(y_val, y_pred, average='macro')
print(f"🎯 Validation Macro F1 Score: {f1:.4f}")
