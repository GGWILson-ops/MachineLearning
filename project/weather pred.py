import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Weather Predictor",
    page_icon="🌦️",
    layout="centered"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f1b2d 0%, #1a2f4a 40%, #0d2137 100%);
    min-height: 100vh;
}

/* Title */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.6rem !important;
    background: linear-gradient(90deg, #7ec8e3, #a8edea, #ffffff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
    margin-bottom: 0 !important;
}

h3 {
    font-family: 'Syne', sans-serif !important;
    color: #a8edea !important;
    font-weight: 600 !important;
}

/* Subtitle */
.subtitle {
    color: #7ec8e3;
    font-size: 1rem;
    font-weight: 300;
    margin-top: -8px;
    margin-bottom: 2rem;
    letter-spacing: 0.5px;
}

/* Card container */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(126, 200, 227, 0.2);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}

/* Slider labels */
label {
    color: #a8edea !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
}

/* Slider track */
.stSlider > div > div > div {
    background: rgba(126, 200, 227, 0.2) !important;
}
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #7ec8e3, #a8edea) !important;
}

/* Predict button */
.stButton > button {
    background: linear-gradient(90deg, #7ec8e3, #a8edea) !important;
    color: #0f1b2d !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(126, 200, 227, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(126, 200, 227, 0.5) !important;
}

/* Result box */
.result-box {
    background: rgba(126, 200, 227, 0.08);
    border: 2px solid rgba(126, 200, 227, 0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}
.result-label {
    color: #7ec8e3;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 500;
}
.result-weather {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0.5rem 0;
}
.result-emoji {
    font-size: 4rem;
    margin-bottom: 0.5rem;
}
.result-confidence {
    color: #a8edea;
    font-size: 0.9rem;
    font-weight: 300;
}

/* Confidence bar */
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 8px;
    margin-top: 0.8rem;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #7ec8e3, #a8edea);
    transition: width 0.6s ease;
}

/* Info pills */
.info-pill {
    display: inline-block;
    background: rgba(126, 200, 227, 0.12);
    border: 1px solid rgba(126, 200, 227, 0.3);
    color: #a8edea;
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.78rem;
    margin: 3px;
    font-weight: 500;
}

/* Divider */
hr {
    border-color: rgba(126, 200, 227, 0.15) !important;
    margin: 1.5rem 0 !important;
}

/* Number display */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}
.metric-box {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(126,200,227,0.15);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #ffffff;
}
.metric-lbl {
    color: #7ec8e3;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ── TRAIN & CACHE MODEL ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv('seattle-weather.csv')
    df = df.drop(['date'], axis=1)

    encoder = LabelEncoder()
    df['weather'] = encoder.fit_transform(df['weather'])

    df = df.drop(['temp_min', 'wind'], axis=1)

    X = df.drop(columns=['weather'])
    y = df['weather']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=2)
    model.fit(X_train, y_train)

    return model, encoder


model, encoder = load_model()

WEATHER_META = {
    "drizzle": {"emoji": "🌦️", "desc": "Light drizzle expected. Carry an umbrella!"},
    "fog": {"emoji": "🌫️", "desc": "Foggy conditions. Drive carefully."},
    "rain": {"emoji": "🌧️", "desc": "Rainy day ahead. Stay dry!"},
    "snow": {"emoji": "❄️", "desc": "Snow expected. Bundle up!"},
    "sun": {"emoji": "☀️", "desc": "Sunny skies! Great day to go outside."},
}

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("<h1>🌦 Weather Predictor </h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Machine learning powered weather prediction</div>", unsafe_allow_html=True)

st.markdown("""
<div style='margin-bottom:1.5rem'>
    <span class='info-pill'>📍 Seattle, WA</span>
    <span class='info-pill'>🤖 Gradient Boosting</span>
    <span class='info-pill'>📊 Trained on Historical Data</span>
    <span class='info-pill'>3 Features</span>
</div>
""", unsafe_allow_html=True)

# ── INPUT SECTION ──────────────────────────────────────────────────────────────
st.markdown("<h3>Enter Weather Conditions</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    precipitation = st.slider(
        "💧 Precipitation (mm)",
        min_value=0.0, max_value=60.0,
        value=10.0, step=0.5,
        help="Total rainfall or snowfall in millimeters"
    )

with col2:
    temp_max = st.slider(
        "🌡️ Max Temperature (°C)",
        min_value=-5.0, max_value=45.0,
        value=15.0, step=0.5,
        help="Maximum temperature of the day"
    )

temp_mean = st.slider(
    "🌤️ Mean Temperature (°C)",
    min_value=-5.0, max_value=40.0,
    value=round((temp_max * 0.75), 1), step=0.5,
    help="Average temperature of the day"
)

# Live input summary
st.markdown(f"""
<div class='metric-row'>
    <div class='metric-box'>
        <div class='metric-val'>{precipitation} mm</div>
        <div class='metric-lbl'>Precipitation</div>
    </div>
    <div class='metric-box'>
        <div class='metric-val'>{temp_max}°C</div>
        <div class='metric-lbl'>Max Temp</div>
    </div>
    <div class='metric-box'>
        <div class='metric-val'>{temp_mean}°C</div>
        <div class='metric-lbl'>Mean Temp</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── PREDICT ────────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Weather"):
    input_df = pd.DataFrame([[precipitation, temp_max, temp_mean]],
                            columns=['precipitation', 'temp_max', 'temp_mean'])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = round(probabilities[prediction] * 100, 1)
    weather_label = encoder.classes_[prediction]
    meta = WEATHER_META.get(weather_label, {"emoji": "🌡️", "desc": ""})

    st.markdown(f"""
    <div class='result-box'>
        <div class='result-emoji'>{meta['emoji']}</div>
        <div class='result-label'>Predicted Weather</div>
        <div class='result-weather'>{weather_label.upper()}</div>
        <div class='result-confidence'>{meta['desc']}</div>
        <div class='conf-bar-bg'>
            <div class='conf-bar-fill' style='width:{confidence}%'></div>
        </div>
        <div style='color:#7ec8e3; font-size:0.8rem; margin-top:0.5rem;'>
            Model confidence: <strong style='color:#fff'>{confidence}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability breakdown
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h3>Probability Breakdown</h3>", unsafe_allow_html=True)
    prob_df = pd.DataFrame({
        "Weather": encoder.classes_,
        "Probability (%)": [round(p * 100, 1) for p in probabilities]
    }).sort_values("Probability (%)", ascending=False)

    for _, row in prob_df.iterrows():
        m = WEATHER_META.get(row["Weather"], {"emoji": "🌡️"})
        st.markdown(f"""
        <div style='margin-bottom:0.6rem'>
            <div style='display:flex; justify-content:space-between; color:#a8edea; font-size:0.85rem; margin-bottom:3px'>
                <span>{m['emoji']} {row['Weather'].capitalize()}</span>
                <span style='color:#fff; font-weight:600'>{row['Probability (%)']:.1f}%</span>
            </div>
            <div class='conf-bar-bg'>
                <div class='conf-bar-fill' style='width:{row["Probability (%)"]}%; opacity:0.7'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:rgba(126,200,227,0.4); font-size:0.75rem; padding-bottom:1rem'>
    Trained on Seattle Weather Dataset · Gradient Boosting Classifier · Features: Precipitation, Temp Max, Temp Mean
</div>
""", unsafe_allow_html=True)