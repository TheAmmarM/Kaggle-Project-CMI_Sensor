# ⚡ OPTUNA TUNING ADDED TO LIGHTGBM PIPELINE
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
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
import optuna

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 1. Load Data
print("Loading data...")
df = pd.read_parquet("processed_data.parquet")

# (Insert your feature engineering code here)

# 2. Prepare features and handle NaNs
non_feature_cols = ['sequence_id', 'gesture']
X = df.drop(columns=non_feature_cols)
y = df['gesture']
numeric_cols = X.select_dtypes(include=np.number).columns
X[numeric_cols] = SimpleImputer(strategy='median').fit_transform(X[numeric_cols])

# 3. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Apply SMOTE to boost BFRB classes
boost_classes = ['Eyebrow - pull hair', 'Eyelash - pull hair',
                 'Neck - pinch skin', 'Neck - scratch',
                 'Pinch knee/leg skin', 'Write name on leg', 'Scratch knee/leg skin']
sampling_strategy = {}
class_counts = np.bincount(y_train)
for cname in boost_classes:
    if cname in le.classes_:
        idx = list(le.classes_).index(cname)
        sampling_strategy[idx] = class_counts[idx] + 50
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 6. Feature selection
selector = SelectPercentile(f_classif, percentile=70)
X_train_res = selector.fit_transform(X_train_res, y_train_res)
X_val = selector.transform(X_val)

# 7. Class weights
weights = np.ones(len(le.classes_))
for c in boost_classes:
    if c in le.classes_:
        weights[list(le.classes_).index(c)] = 2.0
sample_weights = weights[y_train_res]

# 8. Define BFRB indices for binary F1
bfrb_indices = [list(le.classes_).index(c) for c in boost_classes if c in le.classes_]

# 9. Optuna objective
def objective(trial):
    params = {
        'objective': 'multiclass',
        'num_class': len(le.classes_),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 128),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'n_estimators': 1200,
        'min_split_gain': 0.001,
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    model = LGBMClassifier(**params)
    model.fit(
        X_train_res, y_train_res,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[early_stopping(30), log_evaluation(period=0)]
    )
    preds = model.predict(X_val)
    macro_f1 = f1_score(y_val, preds, average='macro')
    binary_val = [1 if y in bfrb_indices else 0 for y in y_val]
    binary_pred = [1 if y in bfrb_indices else 0 for y in preds]
    trial.set_user_attr("binary_f1", f1_score(binary_val, binary_pred))
    return macro_f1

# 10. Optimize with Optuna
print("\n🔍 Starting Optuna tuning...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20, timeout=1800)

print("\n✅ Optuna Completed")
print("Best Macro F1:", study.best_value)
print("Best Binary F1:", study.best_trial.user_attrs['binary_f1'])
print("Best Params:")
for k, v in study.best_params.items():
    print(f"{k}: {v}")

# (Continue with final training and ensemble as before using best_params)
