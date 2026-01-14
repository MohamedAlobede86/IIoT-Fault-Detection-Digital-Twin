import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory, make_response
import joblib
import numpy as np
import os
import shap
import csv
from datetime import datetime
# تم حذف StandardScaler() اليدوي لأنه يجب تحميله من الملف لضمان دقة النتائج

app = Flask(__name__)

# 1. Setup Directories (تعديل المسارات لتتوافق مع المجلد الرئيسي)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODEL_DIR = os.path.join(BASE_DIR, 'model') # مسار مجلد الموديلات
LOG_FILE_CSV = os.path.join(RESULTS_DIR, 'prediction_history.csv')

for d in [STATIC_DIR, RESULTS_DIR]:
    if not os.path.exists(d): 
        os.makedirs(d)

# 2. Load Model and Explainer (تعديل ديناميكي لتحميل الملفات من مجلد model)
try:
    # تحميل الملفات باستخدام os.path.join لضمان عملها على أي جهاز
    model = joblib.load(os.path.join(MODEL_DIR, 'fault_classifier_xgboost_digital_twin.pkl'))
    encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    
    # محاولة تحميل السكيلر الأصلي لضمان دقة النتائج بدلاً من تعريف واحد جديد
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler() #fallback
        
    explainer = shap.TreeExplainer(model)
    print("✅ System Core Active: Connected to model/ folder successfully.")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

# English Recommendations Dictionary
RECOMMENDATIONS = {
    "0": "System status is optimal. Continue routine monitoring.",
    "21": "CRITICAL: Sensor 21 high temperature. Inspect cooling water valves immediately.",
    "22": "WARNING: Feedstock oscillation (Sensor 22). Check input pumps.",
    "Mixed": "Unstable indicators. Please review the overall system pressure.",
    "Extreme": "Extreme Outlier detected! Shutdown suggested for electrical inspection.",
    "Incipient": "Early signs of instability. Schedule preventive maintenance."
}

def log_to_csv(data):
    """Function to save each prediction result into a CSV file."""
    file_exists = os.path.isfile(LOG_FILE_CSV)
    with open(LOG_FILE_CSV, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'filename', 'status', 'fault_id', 'confidence', 'recommendation'])
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        samples_raw = data.get('samples', [])
        dataset_name = data.get('dataset_name', 'test_file.csv').lower()

        if not samples_raw:
            return jsonify({'error': 'No data provided'}), 400

        # 3. Data Processing (Raw and Scaled)
        X = np.zeros((len(samples_raw), 55))
        extreme_flag = False
        for i, row in enumerate(samples_raw):
            vals = list(row.values())
            for j in range(min(len(vals), 55)):
                val = float(vals[j])
                X[i, j] = val
                if abs(val) > 150: extreme_flag = True

        # Use the raw probabilities to maintain sensitivity
        probs = model.predict_proba(X)
        avg_probs = np.mean(probs, axis=0)
        top_indices = np.argsort(avg_probs)[::-1]
        
        p_idx = top_indices[0]
        s_idx = top_indices[1]
        
        p_fault = str(encoder.inverse_transform([p_idx])[0])
        s_fault = str(encoder.inverse_transform([s_idx])[0])
        
        # 4. Decision Engine (Sensitivity Logic)
        display_fault = p_fault
        system_status = "Stable"
        confidence = avg_probs[p_idx]

        # Trigger on secondary fault if primary is '0' but secondary is significant (> 5%)
        if p_fault == "0" and avg_probs[s_idx] > 0.05:
            system_status = "Warning"
            display_fault = f"Incipient: Fault {s_fault}"
            confidence = avg_probs[s_idx]
        
        elif extreme_flag:
            system_status = "Critical"
            display_fault = "Extreme Outlier (F21)"
            confidence = 1.0
            
        elif p_fault != "0":
            system_status = "Critical"

        # Map English Recommendation
        key = "Incipient" if "Incipient" in display_fault else (display_fault.split()[0])
        recommendation = RECOMMENDATIONS.get(key, "Anomaly detected. Consult system logs.")

        # 5. Generate SHAP Visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_idx = s_idx if system_status == "Warning" else p_idx
        
        plt.figure(figsize=(10, 6))
        shap_values = explainer.shap_values(X)
        current_shap = shap_values[target_idx] if isinstance(shap_values, list) else shap_values
        
        shap.summary_plot(current_shap, X, plot_type="bar", show=False, max_display=10)
        plt.title(f"Status: {system_status} | Diagnostic: {display_fault}", fontsize=10, color='navy')
        
        # Save unique version in results folder
        plt.savefig(os.path.join(RESULTS_DIR, f'shap_{timestamp}.png'), bbox_inches='tight', dpi=110)
        # Save/Overwrite 'current' version in static for UI display
        plt.savefig(os.path.join(STATIC_DIR, 'shap_result.png'), bbox_inches='tight', dpi=110)
        plt.close()

        # 6. Logging to CSV
        log_to_csv({
            'timestamp': timestamp,
            'filename': dataset_name,
            'status': system_status,
            'fault_id': display_fault,
            'confidence': f"{confidence:.2%}",
            'recommendation': recommendation
        })

        return jsonify({
            'fault_id': display_fault,
            'confidence': f"{confidence:.2%}",
            'system_status': system_status,
            'recommendation': recommendation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<filename>')
def serve_static(filename):
    response = make_response(send_from_directory(STATIC_DIR, filename))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return response

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)