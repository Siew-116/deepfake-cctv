import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import joblib
import os
from datetime import datetime, timedelta
import random

# =========================
# GENERATE SYNTHETIC TRAIN & TEST DATA
# =========================
def generate_logs(num_rows, start_time, contamination=0.08):
    event_types = [
        "Start camera", "Stop camera", "Motion detected",
        "Signal lost", "Door opened", "Door closed", "Low battery"
    ]
    anomaly_events = [
        "Tamper detected", "Unauthorized access", "Camera offline"
    ]
    timestamps = [start_time]
    events = []
    labels = []

    for i in range(num_rows):
        delta = timedelta(seconds=random.randint(1,10))
        ts = timestamps[-1] + delta
        timestamps.append(ts)

        if random.random() < contamination:
            event = random.choice(anomaly_events)
            label = 1
        else:
            event = random.choice(event_types)
            label = 0

        events.append(event)
        labels.append(label)

    df = pd.DataFrame({
        'timestamp': timestamps[1:],
        'event': events,
        'label': labels
    })
    return df

# Generate datasets
start_train = datetime(2026,1,1,0,0,0)
start_test = datetime(2026,2,1,0,0,0)

train_df = generate_logs(5000, start_train, contamination=0.08)
test_df  = generate_logs(2000, start_test, contamination=0.08)

# =========================
# PREPARE FEATURES
# =========================
for df in [train_df, test_df]:
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

X_train = train_df[['event', 'hour', 'minute', 'second', 'time_diff']]
y_train = train_df['label']

X_test = test_df[['event', 'hour', 'minute', 'second', 'time_diff']]
y_test = test_df['label']

# =========================
# PIPELINE
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(analyzer='char', ngram_range=(3,5)), 'event'),
        ('num', StandardScaler(), ['hour','minute','second','time_diff'])
    ]
)

model = IsolationForest(
    n_estimators=200,
    contamination=0.08,
    random_state=42
)

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', model)
])

# =========================
# TRAIN
# =========================
pipeline.fit(X_train)

# =========================
# TEST PREDICTION
# =========================
y_pred_test = pipeline.predict(X_test)
y_pred_test = np.where(y_pred_test == -1, 1, 0)

print("=== TEST SET PERFORMANCE ===")
print(classification_report(y_test, y_pred_test))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================
# ANOMALY SCORE DISTRIBUTION
# =========================
scores = pipeline.decision_function(X_test)
plt.figure(figsize=(6,4))
sns.histplot(scores, bins=50)
plt.title("Anomaly Score Distribution")
plt.xlabel("Isolation Forest Score")
plt.ylabel("Count")
plt.show()

# =========================
# SAVE PIPELINE
# =========================
folder_path = 'data'
os.makedirs(folder_path, exist_ok=True)
model_path = os.path.join(folder_path, 'log_anomaly_isolation_forest.pkl')

if not os.path.exists(model_path):
    joblib.dump(pipeline, model_path)
    print(f"Pipeline model saved successfully at '{model_path}'!")
else:
    print(f"File '{model_path}' already exists. Skipping save.")