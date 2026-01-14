import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder

# 1) مسارات
os.makedirs('model', exist_ok=True)
os.makedirs('data/digital_twin_samples', exist_ok=True)

print("=== تدريب XGBoost مع التوأم الرقمي (نسخة مصححة) ===")

# 2) تعريف الأعمدة (55 عمودًا)
feature_cols = [f"xmeas_{i}" for i in range(1, 42)] + \
               [f"xmv_{i}"   for i in range(1, 12)] + \
               ["noise_amplitude", "drift_severity", "mixed_flag"]

# 3) دالة توليد صناعي
def generate_digital_twin_samples(n_samples, fault_type, max_features=55, seed=42):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        # 41 قياس + 11 متغير تحكم = 52
        sample = rng.uniform(0.2, 0.8, size=52)

        noise_amp = 0.0; drift_sev = 0.0; mixed_flag = 0.0

        if fault_type == 0:  # Normal
            sample[10] += rng.normal(0, 0.01)
            sample[29] += rng.normal(0, 0.01)
            sample[38] += rng.normal(0, 0.01)

        elif fault_type == 21:  # Drift
            sample[10] += rng.uniform(0.2, 0.5)
            sample[29] += rng.uniform(0.1, 0.3)
            sample[38] += rng.uniform(0.1, 0.25)
            drift_sev = rng.uniform(0.8, 1.0)

        elif fault_type == 22:  # Sensor Noise
            sample[10] += rng.normal(0.25, 0.1)
            sample[29] += rng.normal(0.0, 0.25)
            sample[38] += rng.normal(0.0, 0.2)
            noise_amp = rng.uniform(0.1, 0.3)

        elif fault_type == 23:  # Mixed
            sample[10] += rng.normal(0.9, 0.08)
            sample[29] += rng.normal(0.33, 0.2)
            sample[38] += rng.normal(0.33, 0.18)
            drift_sev = 0.95
            noise_amp = rng.uniform(0.2, 0.4)
            mixed_flag = 1.0

        elif fault_type == 24:  # Hybrid (Drift + Noise)
            sample[10] += rng.normal(0.5, 0.15)
            sample[29] += rng.normal(0.2, 0.2)
            sample[38] += rng.normal(0.25, 0.18)
            drift_sev = rng.uniform(0.7, 0.9)
            noise_amp = rng.uniform(0.1, 0.25)
            mixed_flag = 1.0

        # دمج مع ميزات التوأم الرقمي
        full_sample = np.concatenate([sample, np.array([noise_amp, drift_sev, mixed_flag])])

        # تأكد أن العدد = 55
        if len(full_sample) < max_features:
            full_sample = np.concatenate([full_sample, np.zeros(max_features - len(full_sample))])
        elif len(full_sample) > max_features:
            full_sample = full_sample[:max_features]

        samples.append(full_sample)
    return np.array(samples)

# 4) توليد صناعي متوازن
SYN_PER_CLASS = 600
X_synth = []
y_synth = []
for ft in [0, 21, 22, 23, 24]:
    arr = generate_digital_twin_samples(SYN_PER_CLASS, ft)
    X_synth.append(arr)
    y_synth += [ft] * SYN_PER_CLASS
X_synth = np.vstack(X_synth)
y_synth = np.array(y_synth)

# 5) تحويل إلى DataFrame صناعي
df_synth = pd.DataFrame(X_synth, columns=feature_cols)
df_synth["fault_id"] = y_synth
print("✅ حجم البيانات الصناعية:", df_synth.shape)

# 6) قراءة الملفات الحقيقية (إن وجدت)
try:
    df_norm   = pd.read_csv("data/digital_twin_samples/normal.csv"); df_norm["fault_id"] = 0
    df_driftR = pd.read_csv("data/digital_twin_samples/drift_real.csv"); df_driftR["fault_id"] = 21
    df_driftS = pd.read_csv("data/digital_twin_samples/drift_synthetic.csv"); df_driftS["fault_id"] = 21
    df_hybrid = pd.read_csv("data/digital_twin_samples/hybrid_fault.csv"); df_hybrid["fault_id"] = 24

    df_real_all = pd.concat([df_norm, df_driftR, df_driftS, df_hybrid], ignore_index=True)
    print("✅ حجم البيانات الحقيقية:", df_real_all.shape)

    # دمج مع الصناعي
    df_total = pd.concat([df_synth, df_real_all], ignore_index=True)
except Exception as e:
    print("⚠️ لم يتم العثور على بعض الملفات الحقيقية، سيتم الاعتماد على البيانات الصناعية فقط.")
    df_total = df_synth

# 7) انفصال الميزات والهدف
X_all = df_total[feature_cols].values
y_all = df_total["fault_id"].values

# 8) ترميز الفئات وتقسيم البيانات
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y_all)
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print("توزيع الفئات في التدريب:", dict(zip(*np.unique(y_train, return_counts=True))))

# 9) أوزان موازنة
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 10) إعداد وتدريب النموذج
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    tree_method='hist',
    eval_metric='mlogloss'
)
model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=True
)

# 11) تقييم سريع
y_pred = model.predict(X_val)
print("\n=== تقييم مبدئي ===")
print(classification_report(
    y_val, y_pred,
    target_names=[str(c) for c in encoder.classes_]
))

# 12) حفظ النموذج والـ encoder
joblib.dump(model, 'model/fault_classifier_xgboost_digital_twin.pkl')
joblib.dump(encoder, 'model/label_encoder.pkl')
print("\n✅ تم حفظ النموذج والـ LabelEncoder في مجلد model")
