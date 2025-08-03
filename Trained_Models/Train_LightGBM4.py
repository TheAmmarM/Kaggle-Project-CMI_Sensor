import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, early_stopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# 1. Load and prepare data
print("Loading data...")
df = pd.read_parquet("processed_data.parquet")

# 2. Feature engineering
print("Engineering features...")
# Eye features
upper_eye_cols = [col for col in df.columns if 'tof_1_v' in col and int(col.split('_v')[1].split('_')[0]) < 32]
lower_eye_cols = [col for col in df.columns if 'tof_1_v' in col and int(col.split('_v')[1].split('_')[0]) >= 32]
if upper_eye_cols and lower_eye_cols:
    df['eye_tof_ratio'] = df[upper_eye_cols].mean(axis=1) / (df[lower_eye_cols].mean(axis=1) + 1e-6)

# Neck features
tof5_cols = [col for col in df.columns if 'tof_5_v' in col]
if tof5_cols:
    df['neck_tof_center'] = df[tof5_cols].mean(axis=1)

# 3. Prepare features/target
non_feature_cols = ['sequence_id', 'gesture']
X = df.drop(columns=non_feature_cols)
y = df['gesture']

# 4. Handle missing values
numeric_cols = X.select_dtypes(include=np.number).columns
imputer = SimpleImputer(strategy='median')
X[numeric_cols] = imputer.fit_transform(X[numeric_cols])

# 5. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 6. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 7. Apply SMOTE
boost_classes = ['Eyebrow - pull hair', 'Eyelash - pull hair',
                 'Neck - pinch skin', 'Neck - scratch',
                 'Pinch knee/leg skin', 'Write name on leg']
class_indices = [list(le.classes_).index(c) for c in boost_classes if c in le.classes_]

sampling_strategy = {}
class_counts = np.bincount(y_train)
for idx in class_indices:
    sampling_strategy[idx] = class_counts[idx] + 50  # Add 50 samples

if sampling_strategy:
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 8. Class weighting
class_weights = np.ones(len(le.classes_))
for idx in class_indices:
    class_weights[idx] = 2.0
sample_weights = class_weights[y_train_res]

# 9. Train LightGBM
lgbm = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y_train_res)),
    learning_rate=0.05,
    n_estimators=1500,
    num_leaves=63,
    min_child_samples=30,
    reg_alpha=0.1,
    reg_lambda=0.1,
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

# 10. Train XGBoost
xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y_train_res)),
    learning_rate=0.07,
    max_depth=6,
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
xgb.fit(X_train_res, y_train_res)

# 11. Ensemble predictions
lgbm_proba = lgbm.predict_proba(X_val)
xgb_proba = xgb.predict_proba(X_val)
ensemble_proba = (0.75 * lgbm_proba) + (0.25 * xgb_proba)
y_pred = np.argmax(ensemble_proba, axis=1)

# 12. Evaluate
f1_macro = f1_score(y_val, y_pred, average='macro')
print(f"🎯 Ensemble Macro F1 Score: {f1_macro:.4f}")

# Binary F1 check (critical!)
binary_pred = (y_pred < 8)  # First 8 classes = BFRB
binary_true = (y_val < 8)  # Same for true labels
binary_f1 = f1_score(binary_true, binary_pred)
print(f"🔒 Binary F1 Score: {binary_f1:.4f}")

# Classification report
print(classification_report(y_val, y_pred, target_names=le.classes_))

# Confusion matrix for problem gestures
confused_indices = [list(le.classes_).index(c) for c in boost_classes[:4]]
conf_mask = np.isin(y_val, confused_indices)
print("Confusion Matrix for Problem Gestures:")
print(confusion_matrix(y_val[conf_mask], y_pred[conf_mask]))