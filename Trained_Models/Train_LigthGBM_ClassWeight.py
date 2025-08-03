import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_parquet("processed_data.parquet")

# Encode gestures
le = LabelEncoder()
df['gesture_encoded'] = le.fit_transform(df['gesture'])

# Features and target
X = df.drop(columns=['sequence_id', 'gesture', 'gesture_encoded'])
y = df['gesture_encoded']

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize model
model = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y_train)),
    class_weight='balanced',
    n_estimators=200,
    random_state=42
)

# Train with early stopping using callbacks
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    callbacks=[early_stopping(stopping_rounds=10)]
)

# Predict
y_pred = model.predict(X_val)

# Macro F1 Score
f1_macro = f1_score(y_val, y_pred, average='macro')
print(f"\n🎯 LightGBM Gesture Macro F1 Score: {f1_macro:.4f}")

# Classification report
print("\n📄 Classification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, xticklabels=le.classes_, yticklabels=le.classes_,
            annot=False, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix - LightGBM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
