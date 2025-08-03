import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and enhance data
print("Loading data...")
df = pd.read_parquet("processed_data.parquet")

# 2. Add specialized features for confused pairs
print("Engineering specialized features...")
# Eye region features (eyebrow vs eyelash)
tof_sensor = 'tof_1_'  # Sensor near eye region
# Create column names for aggregated features
upper_eye_cols = [col for col in df.columns if col.startswith(tof_sensor) and 'v' in col and int(col.split('_v')[1].split('_')[0]) < 32]
lower_eye_cols = [col for col in df.columns if col.startswith(tof_sensor) and 'v' in col and int(col.split('_v')[1].split('_')[0]) >= 32]

if upper_eye_cols and lower_eye_cols:
    df['eye_tof_gradient'] = df[upper_eye_cols].mean(axis=1) - df[lower_eye_cols].mean(axis=1)
else:
    print("⚠️ Warning: TOF eye columns not found. Skipping eye_tof_gradient")
    df['eye_tof_gradient'] = 0

# Thermal features
if 'thm_2_mean' in df.columns and 'thm_1_mean' in df.columns:
    df['eye_thermal_asymmetry'] = df['thm_2_mean'] - df['thm_1_mean']
else:
    print("⚠️ Warning: Thermal columns not found. Skipping eye_thermal_asymmetry")
    df['eye_thermal_asymmetry'] = 0

# Neck region features (pinch vs scratch)
df['neck_motion_range'] = df['acc_z_max'] - df['acc_z_min']  # Motion intensity
tof5_cols = [col for col in df.columns if col.startswith('tof_5_')]
if tof5_cols:
    df['neck_tof_activation'] = df[tof5_cols].mean(axis=1)
else:
    print("⚠️ Warning: TOF5 columns not found. Skipping neck_tof_activation")
    df['neck_tof_activation'] = 0

# 3. Separate features and target BEFORE imputation
print("Preparing features and target...")
non_feature_cols = ['sequence_id', 'gesture']
X = df.drop(columns=non_feature_cols)
y = df['gesture']

# 4. Handle missing values ONLY on numeric features
print("Handling missing values...")
numeric_cols = X.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='median')
X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

# 5. Encode gestures
print("Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 6. Train/Validation split
print("Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 7. Apply SMOTE with dynamic target sizing
print("Applying SMOTE...")
low_sample_gestures = ['Pinch knee/leg skin', 'Write name on leg', 'Scratch knee/leg skin']
low_sample_indices = [i for i, name in enumerate(le.classes_) if name in low_sample_gestures]

# Calculate current class counts
class_counts = np.bincount(y_train)
sampling_strategy = {}

for idx in low_sample_indices:
    current_count = class_counts[idx]
    if current_count < 150:  # Only boost classes with <150 samples
        target_count = current_count + 20  # Add 20 samples
        sampling_strategy[idx] = target_count
        print(f"Class {le.classes_[idx]} ({idx}): {current_count} -> {target_count} samples")

if sampling_strategy:
    print(f"Applying SMOTE to: {sampling_strategy}")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied. New training size: {len(X_train_res)}")
else:
    print("⚠️ No eligible classes for SMOTE. Using original data")
    X_train_res, y_train_res = X_train, y_train

# 8. Initialize optimized LightGBM model
print("Initializing model...")
model = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y_train_res)),
    learning_rate=0.05,
    n_estimators=2000,
    num_leaves=63,
    min_child_samples=45,
    reg_alpha=0.1,
    reg_lambda=0.1,
    feature_fraction=0.8,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# 9. Train with early stopping
print("Training model...")
model.fit(
    X_train_res, y_train_res,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    callbacks=[early_stopping(stopping_rounds=20, verbose=True)]
)

# 10. Predict and evaluate
print("Evaluating model...")
y_pred = model.predict(X_val)

# Macro F1 Score
f1_macro = f1_score(y_val, y_pred, average='macro')
print(f"\n🎯 Optimized LightGBM Gesture Macro F1 Score: {f1_macro:.4f}")

# Classification report
print("\n📄 Classification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# 11. Confusion Matrix for analysis
confused_gestures = ['Eyebrow - pull hair', 'Eyelash - pull hair',
                     'Neck - pinch skin', 'Neck - scratch']
confused_indices = [i for i, name in enumerate(le.classes_) if name in confused_gestures]

# Filter for confused pairs
conf_mask = np.isin(y_val, confused_indices)
y_val_conf = y_val[conf_mask]
y_pred_conf = y_pred[conf_mask]

print("\n🔍 Focused Confusion Matrix for Problem Gestures:")
print(confusion_matrix(y_val_conf, y_pred_conf))
print(classification_report(y_val_conf, y_pred_conf, target_names=confused_gestures))

# 12. Save model and results
print("\n✅ Saving results...")
model.booster_.save_model('optimized_model.txt')
print("Model saved as optimized_model.txt")