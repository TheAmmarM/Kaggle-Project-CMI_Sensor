# ----------------------
# Baseline Evaluation & Diagnosis
# ----------------------
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load preprocessed data
processed_df = pd.read_parquet("processed_data.parquet")

# ✅ 1. Create LabelEncoder FIRST
le = LabelEncoder()

# ✅ 2. Include 'no_gesture' explicitly in fitting
all_possible_labels = processed_df['gesture'].unique().tolist()
if 'no_gesture' not in all_possible_labels:
    all_possible_labels.append('no_gesture')
le.fit(all_possible_labels)

# ✅ 3. Encode gestures (multiclass labels)
processed_df['gesture_encoded'] = le.transform(processed_df['gesture'])

# ✅ 4. Create binary labels (BFRB vs non-BFRB)
processed_df['binary_target'] = processed_df['gesture'].apply(lambda x: 0 if x == 'no_gesture' else 1)

# ✅ 5. Train-test split
X = processed_df.drop(columns=['sequence_id', 'gesture', 'gesture_encoded', 'binary_target'])
y_multi = processed_df['gesture_encoded']
y_binary = processed_df['binary_target']
X_train, X_val, y_train_multi, y_val_multi, y_train_bin, y_val_bin = train_test_split(
    X, y_multi, y_binary, test_size=0.2, random_state=42, stratify=y_multi)

# ✅ 6. Train Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_multi)

# ✅ 7. Predict
y_pred_multi = rf.predict(X_val)

# ✅ 8. Convert to binary: compare with label index of 'no_gesture'
no_gesture_index = le.transform(['no_gesture'])[0]
y_pred_bin = np.where(y_pred_multi == no_gesture_index, 0, 1)

# ✅ 9. F1 Scores
f1_macro = f1_score(y_val_multi, y_pred_multi, average='macro')
f1_binary = f1_score(y_val_bin, y_pred_bin, average='binary')
print(f"\n🎯 Gesture Macro F1 Score: {f1_macro:.4f}")
print(f"🎯 Binary F1 Score (BFRB vs non-BFRB): {f1_binary:.4f}")

# ✅ 10. Confusion Matrix
cm = confusion_matrix(y_val_multi, y_pred_multi)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, xticklabels=le.classes_, yticklabels=le.classes_, annot=False, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Multi-Class Gesture")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ✅ 11. Classification Report
print("\nClassification Report:")
print(classification_report(y_val_multi, y_pred_multi, target_names=le.classes_))
