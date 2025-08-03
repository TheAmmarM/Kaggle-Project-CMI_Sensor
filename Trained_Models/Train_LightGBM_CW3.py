import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectPercentile, f_classif
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 1. Load and enhance data
print("Loading data...")
df = pd.read_parquet("processed_data.parquet")

# 2. Add specialized features for confused pairs
print("Engineering specialized features...")


# Function to safely get TOF columns
def get_tof_columns(prefix, indices):
    cols = []
    for i in indices:
        col_name = f"{prefix}{i}_mean"
        if col_name in df.columns:
            cols.append(col_name)
    return cols


# Eye region features (eyebrow vs eyelash)
try:
    # Use existing columns instead of regex for safety
    upper_eye_indices = list(range(0, 32))
    lower_eye_indices = list(range(32, 64))
    upper_eye_cols = get_tof_columns('tof_1_v', upper_eye_indices)
    lower_eye_cols = get_tof_columns('tof_1_v', lower_eye_indices)

    if upper_eye_cols and lower_eye_cols:
        df['eye_tof_gradient'] = df[upper_eye_cols].mean(axis=1) - df[lower_eye_cols].mean(axis=1)
        df['eye_tof_vertical_ratio'] = df[upper_eye_cols].mean(axis=1) / (df[lower_eye_cols].mean(axis=1) + 1e-6)
    else:
        print("⚠️ Warning: Missing TOF columns for eye features")
        df['eye_tof_gradient'] = 0
        df['eye_tof_vertical_ratio'] = 1
except Exception as e:
    print(f"⚠️ Error creating eye features: {e}")
    df['eye_tof_gradient'] = 0
    df['eye_tof_vertical_ratio'] = 1

# Thermal features
if 'thm_1_max' in df.columns and 'thm_2_max' in df.columns:
    df['eye_thermal_max_diff'] = df['thm_1_max'] - df['thm_2_max']
else:
    print("⚠️ Warning: Thermal max columns not found")
    df['eye_thermal_max_diff'] = 0

# Neck region features (pinch vs scratch)
try:
    # Center 4x4 pixels (rows 3-6, cols 3-6) in 8x8 grid
    center_indices = []
    for row in [3, 4, 5, 6]:
        for col in [3, 4, 5, 6]:
            center_indices.append(row * 8 + col)

    neck_center_cols = get_tof_columns('tof_5_v', center_indices)
    if neck_center_cols:
        df['neck_tof_center_activation'] = df[neck_center_cols].mean(axis=1)
    else:
        print("⚠️ Warning: Missing TOF5 center columns")
        df['neck_tof_center_activation'] = 0

    # Motion features
    df['neck_motion_range'] = df['acc_z_max'] - df['acc_z_min']
    df['neck_motion_irregularity'] = df['acc_z_std'] / (df['neck_motion_range'] + 1e-6)
except Exception as e:
    print(f"⚠️ Error creating neck features: {e}")
    df['neck_tof_center_activation'] = 0
    df['neck_motion_irregularity'] = 0

# 3. Temporal feature - gesture duration
print("Creating gesture duration feature...")
if 'sequence_counter' in df.columns:
    # Calculate duration from grouped data
    seq_counts = df.groupby('sequence_id')['sequence_counter'].agg(['min', 'max'])
    seq_counts['duration'] = seq_counts['max'] - seq_counts['min']
    df = df.merge(seq_counts[['duration']], how='left', left_on='sequence_id', right_index=True)
    df.rename(columns={'duration': 'gesture_duration'}, inplace=True)
else:
    print("⚠️ Warning: No sequence counter column found. Setting duration to 0")
    df['gesture_duration'] = 0

# 4. Separate features and target BEFORE imputation
print("Preparing features and target...")
non_feature_cols = ['sequence_id', 'gesture']
X = df.drop(columns=non_feature_cols)
y = df['gesture']

# 5. Handle missing values ONLY on numeric features
print("Handling missing values...")
numeric_cols = X.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='median')
X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

# 6. Encode gestures
print("Encoding labels...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 7. Train/Validation split
print("Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 8. Apply SMOTE with dynamic target sizing
print("Applying SMOTE...")
confused_classes = ['Eyebrow - pull hair', 'Eyelash - pull hair',
                    'Neck - pinch skin', 'Neck - scratch']
low_sample_gestures = ['Pinch knee/leg skin', 'Write name on leg',
                       'Scratch knee/leg skin', 'Drink from bottle/cup']
boost_classes = list(set(confused_classes + low_sample_gestures))

class_counts = np.bincount(y_train)
sampling_strategy = {}

for class_name in boost_classes:
    if class_name in le.classes_:
        class_idx = list(le.classes_).index(class_name)
        current_count = class_counts[class_idx]
        target_count = current_count + 50  # Add 50 samples
        sampling_strategy[class_idx] = target_count
        print(f"Class {class_name} ({class_idx}): {current_count} -> {target_count} samples")

if sampling_strategy:
    print(f"Applying SMOTE to: {sampling_strategy}")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied. New training size: {len(X_train_res)}")
else:
    print("⚠️ No eligible classes for SMOTE. Using original data")
    X_train_res, y_train_res = X_train, y_train

# 9. Feature Selection
print("Applying feature selection...")
selector = SelectPercentile(f_classif, percentile=70)
X_train_res = selector.fit_transform(X_train_res, y_train_res)
X_val = selector.transform(X_val)

# 10. Class Weighting
print("Applying class weights...")
class_weights = np.ones(len(le.classes_))
confused_indices = [list(le.classes_).index(c) for c in confused_classes if c in le.classes_]

for idx in confused_indices:
    class_weights[idx] = 2.0  # Double weight for confused classes

sample_weights = class_weights[y_train_res]

# 11. Train LightGBM Model
print("Training LightGBM model...")
lgbm = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y_train_res)),
    learning_rate=0.03,
    n_estimators=3000,
    num_leaves=45,
    min_child_samples=25,
    reg_alpha=0.2,
    reg_lambda=0.2,
    feature_fraction=0.7,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42,
    n_jobs=-1
)

lgbm.fit(
    X_train_res, y_train_res,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    callbacks=[early_stopping(stopping_rounds=30, verbose=True)]
)

# 12. Train XGBoost Model
print("Training XGBoost model...")
xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y_train_res)),
    learning_rate=0.05,
    max_depth=6,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

xgb.fit(X_train_res, y_train_res)

# 13. Ensemble Predictions
print("Making ensemble predictions...")
lgbm_proba = lgbm.predict_proba(X_val)
xgb_proba = xgb.predict_proba(X_val)

# Combine predictions with weights (3:1 for LGBM:XGB)
ensemble_proba = (3 * lgbm_proba + xgb_proba) / 4
y_pred = np.argmax(ensemble_proba, axis=1)

# 14. Evaluate
print("Evaluating model...")
f1_macro = f1_score(y_val, y_pred, average='macro')
print(f"\n🎯 Optimized Ensemble Gesture Macro F1 Score: {f1_macro:.4f}")

# Classification report
print("\n📄 Classification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# 15. Confusion Matrix for analysis
confused_gestures = ['Eyebrow - pull hair', 'Eyelash - pull hair',
                     'Neck - pinch skin', 'Neck - scratch']
confused_indices = [list(le.classes_).index(c) for c in confused_gestures if c in le.classes_]

# Filter for confused pairs
conf_mask = np.isin(y_val, confused_indices)
y_val_conf = y_val[conf_mask]
y_pred_conf = y_pred[conf_mask]

if len(y_val_conf) > 0:
    print("\n🔍 Focused Confusion Matrix for Problem Gestures:")
    conf_matrix = confusion_matrix(y_val_conf, y_pred_conf)
    print(conf_matrix)

    # Get only relevant classes
    present_indices = np.unique(np.concatenate([y_val_conf, y_pred_conf]))
    present_classes = [le.classes_[i] for i in present_indices]

    print("\n🔍 Focused Classification Report:")
    print(classification_report(y_val_conf, y_pred_conf, target_names=present_classes))
else:
    print("⚠️ No samples for focused confusion analysis")

# 16. Save models
print("\n✅ Saving results...")
lgbm.booster_.save_model('lgbm_model.txt')
xgb.save_model('xgb_model.json')
print("Models saved as lgbm_model.txt and xgb_model.json")