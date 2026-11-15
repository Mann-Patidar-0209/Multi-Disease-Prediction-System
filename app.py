import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedPredict — Disease Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@300;400;600&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --navy:    #0d1b2a;
    --teal:    #1a6b72;
    --teal-lt: #2a9d8f;
    --slate:   #4a5568;
    --danger:  #c0392b;
    --success: #1e8449;
    --body-bg: #f0f4f8;
}
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--body-bg);
    color: var(--navy);
}
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] { background: var(--navy) !important; border-right: 3px solid var(--teal); }
[data-testid="stSidebar"] * { color: #d0dce8 !important; }

.page-header {
    background: linear-gradient(135deg, var(--navy) 0%, #1a3a5c 60%, var(--teal) 100%);
    border-radius: 12px; padding: 2rem 2.5rem 1.6rem;
    margin-bottom: 1.8rem; position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(13,27,42,0.18);
}
.page-header h1 { font-family: 'Source Serif 4', serif; font-size: 1.9rem;
    font-weight: 600; color: #ffffff !important; margin: 0 0 0.3rem; }
.page-header p  { color: #a8c5d8 !important; font-size: 0.92rem; margin: 0; }

.section-label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--teal); margin: 1.4rem 0 0.6rem;
    border-left: 3px solid var(--teal-lt); padding-left: 0.6rem;
}
.result-box {
    border-radius: 10px; padding: 1.4rem 1.8rem; margin-top: 1.2rem;
    font-size: 1.05rem; font-weight: 500; display: flex; align-items: center;
    gap: 0.8rem; border: 1.5px solid; box-shadow: 0 4px 18px rgba(0,0,0,0.07);
}
.result-positive { background: #fdf3f2; border-color: var(--danger); color: var(--danger); }
.result-negative { background: #f0faf4; border-color: var(--success); color: var(--success); }

.stButton > button {
    background: linear-gradient(135deg, var(--teal) 0%, var(--teal-lt) 100%) !important;
    color: #ffffff !important; border: none !important; border-radius: 8px !important;
    padding: 0.6rem 2.4rem !important; font-size: 0.95rem !important;
    font-weight: 600 !important; letter-spacing: 0.04em !important;
    box-shadow: 0 4px 14px rgba(26,107,114,0.35) !important;
    transition: all 0.2s ease !important; margin-top: 1rem !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }

.disclaimer {
    font-size: 0.78rem; color: #8a9bb0; background: #f7f9fc;
    border: 1px dashed #c5d0dc; border-radius: 8px;
    padding: 0.75rem 1rem; margin-top: 1.8rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    models = {}
    errors = []
    files = {
        "heart":     "heart_disease_model.sav",
        "diabetes":  "diabetes_model.sav",
        "cancer":    "breast_cancer_model.sav",   # tuple: (scaler, model)
        "parkinsons":"parkinsons_model.sav",       # tuple: (scaler, model)
    }
    for key, fname in files.items():
        try:
            with open(fname, "rb") as f:
                models[key] = pickle.load(f)
        except FileNotFoundError:
            errors.append(fname)
    return models, errors

models, missing = load_all_models()

# ── Helpers ────────────────────────────────────────────────────────────────────
def result_box(positive, pos_msg, neg_msg):
    if positive:
        st.markdown(f'<div class="result-box result-positive">⚠️ &nbsp; {pos_msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box result-negative">✅ &nbsp; {neg_msg}</div>', unsafe_allow_html=True)

def section(label):
    st.markdown(f'<div class="section-label">{label}</div>', unsafe_allow_html=True)

def predict(key, features):
    """Run prediction; handles plain model or (scaler, model) tuple."""
    obj = models.get(key)
    if obj is None:
        st.error(f"Model '{key}' not loaded. Make sure the .sav file is present.")
        return None
    if isinstance(obj, tuple):
        scaler, mdl = obj
        features = scaler.transform(features)
        return mdl.predict(features)[0]
    return obj.predict(features)[0]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.5rem;">
        <div style="font-size:2.4rem;">🏥</div>
        <div style="font-family:'Source Serif 4',serif;font-size:1.1rem;
                    color:#e0eaf3;font-weight:600;margin-top:0.3rem;">MedPredict</div>
        <div style="font-size:0.72rem;color:#6a8faa;letter-spacing:0.08em;
                    text-transform:uppercase;">Clinical AI System</div>
    </div>
    <hr style="border-color:#1e3a52;margin:1rem 0;">
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Home", "Heart Disease", "Breast Cancer", "Parkinson's", "Diabetes"],
        icons=["house-heart", "heart-pulse", "gender-female", "activity", "droplet"],
        default_index=0,
        styles={
            "container":         {"background-color": "transparent", "padding": "0"},
            "icon":              {"color": "#2a9d8f", "font-size": "1rem"},
            "nav-link":          {"font-size": "0.88rem", "color": "#a0b8cc",
                                  "padding": "0.55rem 1rem", "border-radius": "6px"},
            "nav-link-selected": {"background-color": "#1a3a5c", "color": "#ffffff",
                                  "font-weight": "600"},
        },
    )

    st.markdown("""
    <hr style="border-color:#1e3a52;margin:1rem 0;">
    <div style="font-size:0.72rem;color:#4a6a82;padding:0 0.5rem;">
        ⚠️ For research &amp; educational use only.<br>Not a substitute for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Home":
    st.markdown("""
    <div class="page-header">
        <h1>🏥 MedPredict — Multi-Disease Prediction System</h1>
        <p>AI-assisted screening for four major conditions using clinically validated models</p>
    </div>
    """, unsafe_allow_html=True)

    if missing:
        st.warning(f"⚠️ Missing model files — place these in the same folder as `app.py`:\n\n" +
                   "\n".join(f"- `{f}`" for f in missing))

    col1, col2 = st.columns(2)
    cards = [
        ("❤️",  "Heart Disease",       "88.5% accuracy", "13 clinical parameters including ECG, cholesterol & chest pain type. (Logistic Regression)"),
        ("🩺",  "Breast Cancer",       "97.4% accuracy", "30 cell-nucleus features from digitised FNA images — malignant vs benign. (Logistic Regression + Scaler)"),
        ("🧠",  "Parkinson's Disease", "89.7% accuracy", "22 biomedical voice measurements including jitter, shimmer & NHR. (SVM + Scaler)"),
        ("💉",  "Diabetes",            "76.6% accuracy", "8 diagnostic measurements from the PIMA Indian Diabetes dataset. (SVM)"),
    ]
    for i, (icon, title, acc, desc) in enumerate(cards):
        col = col1 if i % 2 == 0 else col2
        col.markdown(f"""
        <div style="background:#fff;border:1px solid #d6dee8;border-radius:10px;
                    padding:1.4rem 1.6rem;margin-bottom:1rem;
                    box-shadow:0 2px 12px rgba(0,0,0,0.05);">
            <div style="font-size:1.7rem;margin-bottom:0.4rem;">{icon}</div>
            <div style="font-family:'Source Serif 4',serif;font-size:1.05rem;
                        font-weight:600;color:#0d1b2a;margin-bottom:0.2rem;">{title}</div>
            <div style="font-size:0.78rem;color:#2a9d8f;font-weight:600;
                        margin-bottom:0.4rem;">{acc}</div>
            <div style="font-size:0.83rem;color:#4a5568;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="disclaimer">📋 <strong>Disclaimer:</strong> Models trained on publicly available benchmark datasets. Results are indicative only and must not replace professional medical evaluation.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEART DISEASE
# Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
#           oldpeak, slope, ca, thal
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Heart Disease":
    st.markdown("""
    <div class="page-header">
        <h1>❤️ Heart Disease Prediction</h1>
        <p>UCI Cleveland Heart Disease Dataset — 13 clinical features — Logistic Regression (88.5% accuracy)</p>
    </div>
    """, unsafe_allow_html=True)

    section("Patient Demographics")
    c1, c2 = st.columns(2)
    age   = c1.number_input("Age (years)", 1, 120, 50)
    sex   = c2.selectbox("Sex", ["Male (1)", "Female (0)"])
    sex_v = 1 if "Male" in sex else 0

    section("Symptoms")
    cp = st.selectbox("Chest Pain Type",
                      ["0 — Typical Angina", "1 — Atypical Angina",
                       "2 — Non-Anginal Pain", "3 — Asymptomatic"])
    cp_v = int(cp[0])

    section("Clinical Readings")
    c1, c2, c3 = st.columns(3)
    trestbps = c1.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 120)
    chol     = c2.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs      = c3.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
    fbs_v    = 1 if "Yes" in fbs else 0

    c1, c2, c3 = st.columns(3)
    restecg  = c1.selectbox("Resting ECG Result",
                             ["0 — Normal", "1 — ST-T Abnormality", "2 — LV Hypertrophy"])
    restecg_v = int(restecg[0])
    thalach  = c2.number_input("Max Heart Rate Achieved", 60, 220, 150)
    exang    = c3.selectbox("Exercise-Induced Angina", ["No (0)", "Yes (1)"])
    exang_v  = 1 if "Yes" in exang else 0

    c1, c2, c3 = st.columns(3)
    oldpeak  = c1.number_input("ST Depression (Oldpeak)", 0.0, 10.0, 1.0, step=0.1)
    slope    = c2.selectbox("Slope of Peak ST Segment",
                             ["0 — Upsloping", "1 — Flat", "2 — Downsloping"])
    slope_v  = int(slope[0])
    ca       = c3.selectbox("Major Vessels Coloured by Fluoroscopy", [0, 1, 2, 3])

    thal = st.selectbox("Thalassemia",
                        ["1 — Normal", "2 — Fixed Defect", "3 — Reversible Defect"])
    thal_v = int(thal[0])

    if st.button("🔍  Run Prediction", key="heart"):
        feat = np.array([[age, sex_v, cp_v, trestbps, chol, fbs_v,
                          restecg_v, thalach, exang_v, oldpeak, slope_v, ca, thal_v]])
        pred = predict("heart", feat)
        if pred is not None:
            result_box(pred == 1,
                       "High likelihood of heart disease detected. Please consult a cardiologist.",
                       "No significant indicators of heart disease found.")

    st.markdown('<div class="disclaimer">Based on UCI Cleveland Heart Disease dataset. Model accuracy may vary across populations.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BREAST CANCER
# Features: 30 nucleus measurements (mean, SE, worst)
# Model stored as (StandardScaler, LogisticRegression)
# Label: M=1 (Malignant), B=0 (Benign)
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Breast Cancer":
    st.markdown("""
    <div class="page-header">
        <h1>🩺 Breast Cancer Prediction</h1>
        <p>Wisconsin Diagnostic Breast Cancer Dataset — 30 nucleus features — Logistic Regression (97.4% accuracy)</p>
    </div>
    """, unsafe_allow_html=True)

    section("Mean Cell Nucleus Features")
    c1, c2, c3 = st.columns(3)
    radius_mean          = c1.number_input("Radius Mean",           0.0, 30.0,  14.13, step=0.01)
    texture_mean         = c2.number_input("Texture Mean",          0.0, 40.0,  19.29, step=0.01)
    perimeter_mean       = c3.number_input("Perimeter Mean",        0.0, 200.0, 91.97, step=0.1)
    c1, c2, c3 = st.columns(3)
    area_mean            = c1.number_input("Area Mean",             0.0, 2600.0,654.89,step=1.0)
    smoothness_mean      = c2.number_input("Smoothness Mean",       0.0, 0.2,   0.0964,step=0.0001,format="%.4f")
    compactness_mean     = c3.number_input("Compactness Mean",      0.0, 0.4,   0.1043,step=0.0001,format="%.4f")
    c1, c2, c3 = st.columns(3)
    concavity_mean       = c1.number_input("Concavity Mean",        0.0, 0.5,   0.0888,step=0.0001,format="%.4f")
    concave_points_mean  = c2.number_input("Concave Points Mean",   0.0, 0.2,   0.0489,step=0.0001,format="%.4f")
    symmetry_mean        = c3.number_input("Symmetry Mean",         0.0, 0.4,   0.1812,step=0.0001,format="%.4f")
    fractal_dim_mean     = st.number_input("Fractal Dimension Mean",0.0, 0.1,   0.0628,step=0.0001,format="%.4f")

    section("Standard Error Features")
    c1, c2, c3 = st.columns(3)
    radius_se            = c1.number_input("Radius SE",             0.0, 3.0,   0.405, step=0.001, format="%.4f")
    texture_se           = c2.number_input("Texture SE",            0.0, 5.0,   1.217, step=0.001, format="%.4f")
    perimeter_se         = c3.number_input("Perimeter SE",          0.0, 22.0,  2.866, step=0.01)
    c1, c2, c3 = st.columns(3)
    area_se              = c1.number_input("Area SE",               0.0, 550.0, 40.34, step=0.1)
    smoothness_se        = c2.number_input("Smoothness SE",         0.0, 0.03,  0.007, step=0.0001,format="%.5f")
    compactness_se       = c3.number_input("Compactness SE",        0.0, 0.15,  0.025, step=0.001, format="%.4f")
    c1, c2, c3 = st.columns(3)
    concavity_se         = c1.number_input("Concavity SE",          0.0, 0.4,   0.032, step=0.001, format="%.4f")
    concave_points_se    = c2.number_input("Concave Points SE",     0.0, 0.05,  0.012, step=0.0001,format="%.5f")
    symmetry_se          = c3.number_input("Symmetry SE",           0.0, 0.08,  0.021, step=0.001, format="%.4f")
    fractal_dim_se       = st.number_input("Fractal Dimension SE",  0.0, 0.03,  0.004, step=0.0001,format="%.5f")

    section("Worst (Largest) Features")
    c1, c2, c3 = st.columns(3)
    radius_worst         = c1.number_input("Radius Worst",          0.0, 40.0,  16.27, step=0.01)
    texture_worst        = c2.number_input("Texture Worst",         0.0, 50.0,  25.68, step=0.01)
    perimeter_worst      = c3.number_input("Perimeter Worst",       0.0, 260.0, 107.26,step=0.1)
    c1, c2, c3 = st.columns(3)
    area_worst           = c1.number_input("Area Worst",            0.0, 4300.0,880.58,step=1.0)
    smoothness_worst     = c2.number_input("Smoothness Worst",      0.0, 0.3,   0.1324,step=0.001, format="%.4f")
    compactness_worst    = c3.number_input("Compactness Worst",     0.0, 1.1,   0.2543,step=0.001, format="%.4f")
    c1, c2, c3 = st.columns(3)
    concavity_worst      = c1.number_input("Concavity Worst",       0.0, 1.3,   0.2722,step=0.001, format="%.4f")
    concave_points_worst = c2.number_input("Concave Points Worst",  0.0, 0.3,   0.1147,step=0.001, format="%.4f")
    symmetry_worst       = c3.number_input("Symmetry Worst",        0.0, 0.7,   0.2901,step=0.001, format="%.4f")
    fractal_dim_worst    = st.number_input("Fractal Dimension Worst",0.0,0.22,  0.0839,step=0.001, format="%.4f")

    if st.button("🔍  Run Prediction", key="cancer"):
        feat = np.array([[
            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
            compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dim_mean,
            radius_se, texture_se, perimeter_se, area_se, smoothness_se,
            compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dim_se,
            radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
            compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dim_worst
        ]])
        pred = predict("cancer", feat)
        if pred is not None:
            result_box(pred == 1,
                       "Tumour classified as Malignant. Urgent oncological review recommended.",
                       "Tumour classified as Benign. Routine monitoring advised.")

    st.markdown('<div class="disclaimer">Wisconsin Diagnostic Breast Cancer (WDBC) dataset. Output is a classification aid, not a pathological diagnosis.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PARKINSON'S
# Features: 22 voice measures (name & status columns dropped)
# Model stored as (StandardScaler, SVC)
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Parkinson's":
    st.markdown("""
    <div class="page-header">
        <h1>🧠 Parkinson's Disease Prediction</h1>
        <p>UCI Parkinson's Dataset — 22 biomedical voice measurements — SVM (89.7% accuracy)</p>
    </div>
    """, unsafe_allow_html=True)

    section("Fundamental Frequency")
    c1, c2, c3 = st.columns(3)
    fo   = c1.number_input("MDVP:Fo(Hz)",  80.0,  270.0, 154.23, step=0.01)
    fhi  = c2.number_input("MDVP:Fhi(Hz)", 100.0, 600.0, 197.10, step=0.01)
    flo  = c3.number_input("MDVP:Flo(Hz)", 60.0,  240.0, 116.32, step=0.01)

    section("Jitter Measures")
    c1, c2, c3 = st.columns(3)
    jitter_pct = c1.number_input("MDVP:Jitter(%)",    0.0, 2.0,    0.00622, step=0.00001, format="%.5f")
    jitter_abs = c2.number_input("MDVP:Jitter(Abs)",  0.0, 0.0002, 0.0000440,step=0.0000001,format="%.7f")
    rap        = c3.number_input("MDVP:RAP",           0.0, 0.02,  0.00330, step=0.00001, format="%.5f")
    c1, c2 = st.columns(2)
    ppq = c1.number_input("MDVP:PPQ",     0.0, 0.02, 0.00345, step=0.00001, format="%.5f")
    ddp = c2.number_input("Jitter:DDP",   0.0, 0.07, 0.00990, step=0.00001, format="%.5f")

    section("Shimmer Measures")
    c1, c2, c3 = st.columns(3)
    shimmer    = c1.number_input("MDVP:Shimmer",      0.0, 0.12, 0.02971, step=0.00001, format="%.5f")
    shimmer_db = c2.number_input("MDVP:Shimmer(dB)",  0.0, 1.1,  0.28228, step=0.001,   format="%.4f")
    apq3       = c3.number_input("Shimmer:APQ3",      0.0, 0.06, 0.01568, step=0.00001, format="%.5f")
    c1, c2, c3 = st.columns(3)
    apq5       = c1.number_input("Shimmer:APQ5",      0.0, 0.08, 0.02008, step=0.00001, format="%.5f")
    apq        = c2.number_input("MDVP:APQ",          0.0, 0.14, 0.02440, step=0.00001, format="%.5f")
    dda        = c3.number_input("Shimmer:DDA",       0.0, 0.17, 0.04704, step=0.00001, format="%.5f")

    section("Noise & Nonlinear Dynamics")
    c1, c2 = st.columns(2)
    nhr  = c1.number_input("NHR",     0.0, 0.35, 0.02488, step=0.0001, format="%.5f")
    hnr  = c2.number_input("HNR",     8.0, 35.0, 21.886,  step=0.001,  format="%.3f")
    c1, c2, c3 = st.columns(3)
    rpde    = c1.number_input("RPDE",    0.2, 0.7,  0.4986, step=0.001, format="%.4f")
    dfa     = c2.number_input("DFA",     0.5, 0.9,  0.7180, step=0.001, format="%.4f")
    spread1 = c3.number_input("spread1",-8.0,-2.0, -5.6840, step=0.001, format="%.4f")
    c1, c2, c3 = st.columns(3)
    spread2 = c1.number_input("spread2", 0.0, 0.5,  0.2265, step=0.001, format="%.4f")
    d2      = c2.number_input("D2",      1.5, 3.7,  2.3816, step=0.001, format="%.4f")
    ppe     = c3.number_input("PPE",     0.04,0.53, 0.2065, step=0.001, format="%.4f")

    if st.button("🔍  Run Prediction", key="park"):
        feat = np.array([[fo, fhi, flo, jitter_pct, jitter_abs, rap, ppq, ddp,
                          shimmer, shimmer_db, apq3, apq5, apq, dda,
                          nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
        pred = predict("parkinsons", feat)
        if pred is not None:
            result_box(pred == 1,
                       "Voice patterns suggest possible Parkinson's disease. Neurological consultation advised.",
                       "No significant Parkinson's-related voice biomarkers detected.")

    st.markdown('<div class="disclaimer">UCI Parkinson&#39;s dataset. This is a screening aid only; clinical diagnosis requires comprehensive neurological assessment.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DIABETES
# Features: Pregnancies, Glucose, BloodPressure, SkinThickness,
#           Insulin, BMI, DiabetesPedigreeFunction, Age
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "Diabetes":
    st.markdown("""
    <div class="page-header">
        <h1>💉 Diabetes Prediction</h1>
        <p>PIMA Indian Diabetes Dataset — 8 diagnostic measurements — SVM (76.6% accuracy)</p>
    </div>
    """, unsafe_allow_html=True)

    section("Obstetric History")
    pregnancies = st.number_input("Number of Pregnancies", 0, 20, 1)

    section("Blood & Metabolic Readings")
    c1, c2, c3 = st.columns(3)
    glucose    = c1.number_input("Plasma Glucose (mg/dl)", 0, 300, 117)
    blood_pres = c2.number_input("Diastolic Blood Pressure (mm Hg)", 0, 140, 72)
    skin_thick = c3.number_input("Triceps Skin Fold Thickness (mm)", 0, 100, 23)

    section("Lab Values & Anthropometrics")
    c1, c2, c3 = st.columns(3)
    insulin = c1.number_input("2-Hour Serum Insulin (μU/ml)", 0, 900, 30)
    bmi     = c2.number_input("BMI (kg/m²)", 0.0, 70.0, 32.0, step=0.1)
    dpf     = c3.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.372, step=0.001, format="%.3f")

    section("Demographics")
    age_d = st.number_input("Age (years)", 1, 120, 29)

    if st.button("🔍  Run Prediction", key="diab"):
        feat = np.array([[pregnancies, glucose, blood_pres, skin_thick,
                          insulin, bmi, dpf, age_d]])
        pred = predict("diabetes", feat)
        if pred is not None:
            result_box(pred == 1,
                       "High likelihood of diabetes detected. Endocrinology review and HbA1c testing recommended.",
                       "No significant diabetes indicators found. Maintain a healthy lifestyle.")

    st.markdown('<div class="disclaimer">PIMA Indian Diabetes dataset. Fasting glucose and HbA1c lab tests are required for clinical diagnosis.</div>', unsafe_allow_html=True)
