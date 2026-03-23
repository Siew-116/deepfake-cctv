from flask import Flask, request, jsonify, send_from_directory
import os
import joblib
import pandas as pd
import cv2
import numpy as np
import psutil

from tensorflow.keras.models import load_model
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import traceback

app = Flask(__name__)
CORS(app, origins=["https://foreneye.netlify.app"])

# ==============================
# CONFIG
# ==============================
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# ==============================
# Load Models
# ==============================
PIPELINE_PATH = 'model/log_anomaly_isolation_forest_v1.pkl'
MODEL_PATH = "model/VGG_2_FINAL.h5"

deepfake_model = None

if os.path.exists(MODEL_PATH):
    deepfake_model = load_model(MODEL_PATH)
    print("Deepfake model loaded (Keras)")
else:
    print("Warning: Deepfake model not found!")

if os.path.exists(PIPELINE_PATH):
    log_pipeline = joblib.load(PIPELINE_PATH)
    print("Log anomaly model loaded")
else:
    log_pipeline = None
    print("Warning: Log anomaly model not found!")

# ==============================
# Deepfake Detection
# ==============================
def detect_deepfake(video_path, max_frames=10, threshold=0.5):

    if deepfake_model is None:
        return None, None, "Deepfake model not loaded"

    cap = cv2.VideoCapture(video_path)
    frames = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        return None, None, "Cannot read video frames"

    indices = np.linspace(0, total_frames - 1, max_frames).astype(int)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None, None, "No frames found in video"

    frame_preds = []

    for frame in frames:
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = deepfake_model.predict(img, verbose=0)[0][0]

        print("Frame prediction:", pred)

        # assume: 1 = FAKE, 0 = REAL
        fake_score = float(pred)

        frame_preds.append(fake_score)

    visual_score = float(np.mean(frame_preds))
    print("FINAL SCORE:", visual_score)

    video_label = "Fake" if visual_score > threshold else "Real"

    return visual_score, video_label, None

# ==============================
# Log Analysis
# ==============================
def analyze_log(log_path):

    if log_pipeline is None:
        return None, None, "Log model not loaded"

    try:
        df = pd.read_csv(log_path)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')

        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['second'] = df['timestamp'].dt.second
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)

        X = df[['event','hour','minute','second','time_diff']]

        y_pred = log_pipeline.predict(X)
        y_pred = pd.Series(y_pred).replace({-1:1, 1:0})

        anomalies = df[y_pred==1]
        suspicious_logs = anomalies.to_dict(orient='records')

        return len(anomalies), suspicious_logs, None

    except Exception as e:
        return None, None, str(e)

# ==============================
# Error Handler
# ==============================
@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({'status':'error','message':'File too large'}), 413

# ==============================
# Upload Endpoint
# ==============================
@app.route('/upload_files', methods=['POST'])
def upload_files():
    process = psutil.Process(os.getpid())
    print(f"Memory before: {process.memory_info().rss / 1024 / 1024:.1f} MB")
    try:
        if 'video_file' not in request.files:
            return jsonify({'status':'error','message':'video_file missing'}), 400

        video_file = request.files.get('video_file')
        log_file = request.files.get('log_file')

        if video_file.filename == "":
            return jsonify({'status':'error','message':'No video selected'}), 400

        video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
        video_file.save(video_path)

        visual_score, result, error = detect_deepfake(video_path)

        if error:
            return jsonify({'status':'error','message': error}), 400

        response = {
            'status':'success',
            'video_file': video_file.filename,
            'deepfake_score': visual_score,
            'video_label': result
        }

        if log_file and log_file.filename != "":
            log_path = os.path.join(UPLOAD_FOLDER, log_file.filename)
            log_file.save(log_path)

            anomaly_count, suspicious_logs, error = analyze_log(log_path)

            if error:
                response['log_status'] = 'error'
                response['log_error'] = error
            else:
                response['log_status'] = 'analyzed'
                response['anomaly_count'] = anomaly_count
                response['suspicious_logs'] = suspicious_logs
        else:
            response['log_status'] = 'not_provided'
        print(f"Memory after: {process.memory_info().rss / 1024 / 1024:.1f} MB")
        return jsonify(response)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status':'error','message':str(e)}), 500
    

# ==============================
# Run
# ==============================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)