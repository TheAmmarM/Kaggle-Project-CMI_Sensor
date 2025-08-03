import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# -----------------------
# Load Data and Preprocess
# -----------------------

# Load preprocessed data
processed_df = pd.read_parquet("processed_data.parquet")

# Label Encode gesture column if not already encoded
if 'gesture_encoded' not in processed_df.columns:
    le = LabelEncoder()
    processed_df['gesture_encoded'] = le.fit_transform(processed_df['gesture'])

# Drop only existing columns safely
drop_cols = ['sequence_id', 'gesture', 'gesture_encoded', 'binary_target']
drop_cols = [col for col in drop_cols if col in processed_df.columns]
X = processed_df.drop(columns=drop_cols)

# Target column
y_multi = processed_df['gesture_encoded']

# -----------------------
# Sensor-Specific Training
# -----------------------

def train_sensor_model(sensor_prefix, X_full, y):
    sensor_cols = [col for col in X_full.columns if col.startswith(sensor_prefix)]
    if not sensor_cols:
        print(f"No columns found for prefix '{sensor_prefix}'")
        return 0.0
    X_sensor = X_full[sensor_cols]
    X_train, X_val, y_train, y_val = train_test_split(X_sensor, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return f1_score(y_val, preds, average='macro')

# Combine IMU features (acc_* + rot_*)
f1_acc = train_sensor_model('acc_', X, y_multi)
f1_rot = train_sensor_model('rot_', X, y_multi)
f1_imu = (f1_acc + f1_rot) / 2

# Thermopile features
f1_thm = train_sensor_model('thm_', X, y_multi)

# Time-of-Flight features
f1_tof = train_sensor_model('tof_', X, y_multi)

# -----------------------
# Print Results
# -----------------------

print("\n📊 Sensor-wise Macro F1 Scores:")
print(f"IMU (acc + rot): {f1_imu:.4f}")
print(f"Thermopile (thm): {f1_thm:.4f}")
print(f"ToF (tof): {f1_tof:.4f}")
