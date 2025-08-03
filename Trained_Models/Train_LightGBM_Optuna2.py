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

# ... [YOUR EXISTING FEATURE ENGINEERING CODE REMAINS UNCHANGED] ...

# 3. Temporal feature - gesture duration
print("Creating gesture duration feature...")
# ... [YOUR EXISTING DURATION CODE] ...

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

# ... [Previous imports and data loading code remains the same until SMOTE section] ...

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

# Initialize X_train_res/y_train_res with original data first
X_train_res, y_train_res = X_train, y_train

if sampling_strategy:
    print(f"Applying SMOTE to: {sampling_strategy}")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"SMOTE applied. New training size: {len(X_train_res)}")
else:
    print("⚠️ No eligible classes for SMOTE. Using original data")

# Now feature selection can safely use X_train_res
print("Applying feature selection...")
selector = SelectPercentile(f_classif, percentile=70)
X_train_res = selector.fit_transform(X_train_res, y_train_res)
X_val = selector.transform(X_val)

# ... [Rest of your code remains unchanged] ...

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

# Initialize sample_weights before using it
sample_weights = class_weights[y_train_res]

# Now train the model with sample_weights
print("Training LightGBM model with Optuna optimized parameters...")
lgbm = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y_train_res)),
    # OPTUNA OPTIMIZED PARAMETERS
    learning_rate=0.01345,
    max_depth=6,
    num_leaves=126,
    feature_fraction=0.6159,
    bagging_fraction=0.9301,
    bagging_freq=2,
    min_child_samples=74,
    reg_alpha=0.6194,
    reg_lambda=0.7749,
    # ADDITIONAL CRITICAL PARAMETERS
    min_split_gain=0.001,
    n_estimators=3000,
    random_state=42,
    n_jobs=-1
)

lgbm.fit(
    X_train_res, y_train_res,
    sample_weight=sample_weights,  # Now properly defined
    eval_set=[(X_val, y_val)],
    eval_metric='multi_logloss',
    callbacks=[early_stopping(stopping_rounds=30, verbose=True)]
)
# 11. UPDATED LIGHTGBM MODEL WITH OPTUNA OPTIMIZED PARAMETERS
# ================================================================
print("Training LightGBM model with Optuna optimized parameters...")
lgbm = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y_train_res)),
    # OPTUNA OPTIMIZED PARAMETERS
    learning_rate=0.01345,            # From Optuna
    max_depth=6,                      # From Optuna
    num_leaves=126,                   # From Optuna
    feature_fraction=0.6159,          # From Optuna
    bagging_fraction=0.9301,          # From Optuna
    bagging_freq=2,                   # From Optuna
    min_child_samples=74,             # From Optuna
    reg_alpha=0.6194,                 # From Optuna
    reg_lambda=0.7749,                # From Optuna
    # ADDITIONAL CRITICAL PARAMETERS
    min_split_gain=0.001,             # Prevents "-inf gain" warnings
    n_estimators=3000,                # Maintain high iteration count
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

# ================================================================
# 12. BINARY CLASSIFIER FOR BFRB DETECTION (TO RECOVER BINARY F1)
# ================================================================
print("\n🔧 Training binary classifier for BFRB detection...")

# Identify BFRB classes
bfrb_classes = ['Eyebrow - pull hair', 'Eyelash - pull hair',
                'Pinch knee/leg skin', 'Scratch knee/leg skin',
                'Neck - pinch skin', 'Neck - scratch',
                'Write name on leg']
bfrb_indices = [list(le.classes_).index(c) for c in bfrb_classes if c in le.classes_]

# Create binary labels
y_train_binary = np.array([1 if y in bfrb_indices else 0 for y in y_train_res])
y_val_binary = np.array([1 if y in bfrb_indices else 0 for y in y_val])

# Train specialized binary model
binary_clf = LGBMClassifier(
    num_leaves=32,
    min_child_samples=10,
    reg_alpha=0.1,
    n_estimators=200,
    random_state=42
)
binary_clf.fit(X_train_res, y_train_binary)

# 13. Train XGBoost Model (UNCHANGED)
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

# ================================================================
# 14. ENHANCED ENSEMBLE WITH BINARY BOOSTING
# ================================================================
print("Making ensemble predictions with binary boosting...")
lgbm_proba = lgbm.predict_proba(X_val)
xgb_proba = xgb.predict_proba(X_val)

# Combine predictions with weights (3:1 for LGBM:XGB)
ensemble_proba = (3 * lgbm_proba + xgb_proba) / 4

# Get binary probabilities
binary_proba = binary_clf.predict_proba(X_val)[:, 1]  # Probability of BFRB

# Apply binary boost to BFRB classes
BINARY_BOOST_FACTOR = 1.25  # 25% confidence boost
for class_idx in bfrb_indices:
    ensemble_proba[:, class_idx] *= (1 + binary_proba * BINARY_BOOST_FACTOR)

# Renormalize probabilities
ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1)[:, np.newaxis]

y_pred = np.argmax(ensemble_proba, axis=1)

# 15. Enhanced Evaluation
print("Evaluating model with dual metrics...")
# Macro F1 (multiclass)
f1_macro = f1_score(y_val, y_pred, average='macro')
print(f"\n🎯 Optimized Ensemble Gesture Macro F1 Score: {f1_macro:.4f}")

# Binary F1 (BFRB detection)
binary_pred = np.array([1 if pred in bfrb_indices else 0 for pred in y_pred])
binary_f1 = f1_score(y_val_binary, binary_pred)
print(f"🎯 Binary F1 (BFRB vs Non-BFRB): {binary_f1:.4f}")

# Classification report
print("\n📄 Classification Report:")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# 16. Confusion Matrix for analysis (UNCHANGED)
# ... [YOUR EXISTING CONFUSION MATRIX CODE] ...

# 17. Save models
print("\n✅ Saving results...")
lgbm.booster_.save_model('lgbm_model.txt')
xgb.save_model('xgb_model.json')
binary_clf.booster_.save_model('binary_model.txt')
print("Models saved as lgbm_model.txt, xgb_model.json, and binary_model.txt")