import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import soundfile as sf

# =============================
# Page Config
# =============================
st.set_page_config(
    page_title="SynthDetect — AI Audio Detection",
    page_icon="🎵",
    layout="centered"
)

# =============================
# Custom CSS
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ---- Base ---- */
html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #080c14;
    color: #e2e8f0;
}

/* ---- Hide Streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 860px;
}

/* ---- Hero title ---- */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.1;
    background: linear-gradient(135deg, #00f5c4 0%, #0ea5e9 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #475569;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ---- Upload zone ---- */
.upload-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00f5c4;
    margin-bottom: 0.4rem;
}

[data-testid="stFileUploader"] {
    background: #0f1729;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 0.5rem;
    transition: border-color 0.3s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #00f5c4;
}

/* ---- Model cards ---- */
.model-card {
    background: #0f1729;
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.4rem 1.4rem 1rem 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}

.model-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00f5c4, #0ea5e9);
}

.model-card-ai::before {
    background: linear-gradient(90deg, #f43f5e, #fb923c);
}

.model-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.3rem;
}

.model-label-real {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #00f5c4;
    line-height: 1;
    margin-bottom: 0.2rem;
}

.model-label-ai {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #f43f5e;
    line-height: 1;
    margin-bottom: 0.2rem;
}

.model-confidence {
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
    margin-bottom: 1rem;
}

.tier-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.8rem;
}

.tier-real { background: #022c22; color: #00f5c4; border: 1px solid #00f5c4; }
.tier-ai   { background: #2d0a14; color: #f43f5e; border: 1px solid #f43f5e; }

.desc-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.73rem;
    color: #64748b;
    line-height: 1.6;
    margin-bottom: 1rem;
    padding: 0.8rem;
    background: #080c14;
    border-radius: 8px;
    border-left: 2px solid #1e293b;
}

/* ---- Indicators ---- */
.indicator-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 0.5rem;
}

.indicator-row {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #94a3b8;
    padding: 0.4rem 0;
    border-bottom: 1px solid #0f1729;
    line-height: 1.5;
}

.ind-real { color: #00f5c4; }
.ind-ai   { color: #f43f5e; }

/* ---- Section header ---- */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #334155;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e293b;
}

/* ---- Divider ---- */
hr { border-color: #1e293b !important; }

/* ---- Progress bars ---- */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00f5c4, #0ea5e9) !important;
    border-radius: 999px !important;
}

</style>
""", unsafe_allow_html=True)

# =============================
# Load Models
# =============================
@st.cache_resource
def load_models():
    model_mp3 = tf.keras.models.load_model("model_mp3.keras")
    model_wav = tf.keras.models.load_model("model_wav.keras")
    return model_mp3, model_wav

model_mp3, model_wav = load_models()

# =============================
# Constants
# =============================
SAMPLE_RATE      = 22050
N_MFCC           = 20
MAX_DURATION     = 15
FIXED_LENGTH     = SAMPLE_RATE * MAX_DURATION
SPEC_TIME_FRAMES = 646
TEMP_TIME_FRAMES = 646
THRESHOLD        = 0.50

# =============================
# Audio Loader
# =============================
def load_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    if len(audio) <= FIXED_LENGTH:
        return np.pad(audio, (0, FIXED_LENGTH - len(audio)))

    frame_length = int(0.05 * SAMPLE_RATE)
    hop_length   = int(0.01 * SAMPLE_RATE)

    energy = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    center = np.argmax(energy) * hop_length
    start  = max(0, center - FIXED_LENGTH // 2)
    end    = start + FIXED_LENGTH

    if end > len(audio):
        end   = len(audio)
        start = end - FIXED_LENGTH

    segment = audio[start:end]
    if len(segment) < FIXED_LENGTH:
        segment = np.pad(segment, (0, FIXED_LENGTH - len(segment)))

    return segment

# =============================
# Feature Extraction
# =============================
def extract_features(path):
    audio = load_audio(path)

    mel    = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = mel_db[:, :SPEC_TIME_FRAMES]
    if mel_db.shape[1] < SPEC_TIME_FRAMES:
        mel_db = np.pad(mel_db, ((0,0),(0, SPEC_TIME_FRAMES - mel_db.shape[1])), mode="constant")

    mel_db = mel_db[..., np.newaxis]
    X_spec = np.expand_dims(mel_db, axis=0)

    mfcc   = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    rms    = librosa.feature.rms(y=audio)
    zcr    = librosa.feature.zero_crossing_rate(y=audio)

    f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=SAMPLE_RATE
    )

    min_len = min(mfcc.shape[1], len(f0))
    mfcc   = mfcc[:, :min_len]
    delta  = delta[:, :min_len]
    delta2 = delta2[:, :min_len]
    rms    = rms[:, :min_len]
    zcr    = zcr[:, :min_len]
    f0     = f0[:min_len]

    voiced_flag = (~np.isnan(f0)).astype(float)
    f0          = np.nan_to_num(f0)

    temp = np.vstack([mfcc, delta, delta2, rms, zcr, f0[np.newaxis,:], voiced_flag[np.newaxis,:]])
    temp = temp.T[:TEMP_TIME_FRAMES]
    if temp.shape[0] < TEMP_TIME_FRAMES:
        temp = np.pad(temp, ((0, TEMP_TIME_FRAMES - temp.shape[0]),(0,0)), mode="constant")

    X_temp = np.expand_dims(temp, axis=0)

    feature_values = {
        "mfcc_std":     float(np.std(mfcc)),
        "delta_std":    float(np.std(delta)),
        "delta2_std":   float(np.std(delta2)),
        "rms_std":      float(np.std(rms)),
        "rms_mean":     float(np.mean(rms)),
        "zcr_mean":     float(np.mean(zcr)),
        "f0_std":       float(np.std(f0[f0 > 0])) if np.any(f0 > 0) else 0.0,
        "voiced_ratio": float(np.mean(voiced_flag)),
        "spectral_std": float(np.std(mel_db))
    }

    return X_spec, X_temp, feature_values

# =============================
# Feature Analysis
# =============================
def analyze_features(feature_values):
    indicators = []

    indicators.append({
        "feature": "Pitch (F0)",
        "observation": "Unnaturally smooth and consistent pitch" if feature_values["f0_std"] < 15.0 else "Natural pitch variation detected",
        "lean": "AI-generated" if feature_values["f0_std"] < 15.0 else "Real"
    })
    indicators.append({
        "feature": "Voiced Segments",
        "observation": "Overly continuous voicing, little natural breaks" if feature_values["voiced_ratio"] > 0.85 else "Natural voicing pattern with breaks",
        "lean": "AI-generated" if feature_values["voiced_ratio"] > 0.85 else "Real"
    })
    indicators.append({
        "feature": "RMS Energy",
        "observation": "Overly uniform loudness across the track" if feature_values["rms_std"] < 0.02 else "Natural loudness variation detected",
        "lean": "AI-generated" if feature_values["rms_std"] < 0.02 else "Real"
    })
    indicators.append({
        "feature": "MFCC (Timbre)",
        "observation": "Limited timbral variation, unnaturally consistent" if feature_values["mfcc_std"] < 20.0 else "Natural timbral variation detected",
        "lean": "AI-generated" if feature_values["mfcc_std"] < 20.0 else "Real"
    })
    indicators.append({
        "feature": "Delta MFCC (Transitions)",
        "observation": "Overly smooth spectral transitions" if feature_values["delta_std"] < 8.0 else "Natural spectral transition dynamics",
        "lean": "AI-generated" if feature_values["delta_std"] < 8.0 else "Real"
    })
    indicators.append({
        "feature": "Zero-Crossing Rate",
        "observation": "Low noise texture, unusually clean signal" if feature_values["zcr_mean"] < 0.05 else "Natural noise texture present",
        "lean": "AI-generated" if feature_values["zcr_mean"] < 0.05 else "Real"
    })
    indicators.append({
        "feature": "Mel-Spectrogram",
        "observation": "Unnaturally uniform spectral energy distribution" if feature_values["spectral_std"] < 18.0 else "Natural spectral energy variation detected",
        "lean": "AI-generated" if feature_values["spectral_std"] < 18.0 else "Real"
    })

    return indicators

# =============================
# Confidence Interpretation
# =============================
def interpret_prediction(label, ai_prob):
    confidence = ai_prob if label == "AI-generated" else 1 - ai_prob

    if label == "AI-generated":
        if confidence >= 0.90:
            tier = "Very Strong"
            desc = "The model detected strongly synthetic patterns across both spectral and temporal dimensions — including unnaturally smooth pitch contours, uniform spectral energy, and limited timbral variation consistent with AI voice synthesis."
        elif confidence >= 0.75:
            tier = "Strong"
            desc = "The model identified notable synthetic characteristics — spectral uniformity and reduced expressive dynamics in the vocal performance are consistent with AI-generated audio."
        elif confidence >= 0.60:
            tier = "Moderate"
            desc = "Some synthetic patterns were detected — mild spectral smoothness and limited pitch variation suggest possible AI involvement, though the audio retains some natural characteristics."
        else:
            tier = "Borderline"
            desc = "The model marginally classified this as AI-generated. The audio exhibits subtle synthetic traits but closely resembles a real human performance. Manual review is recommended."
    else:
        if confidence >= 0.90:
            tier = "Very Strong"
            desc = "The model detected strongly human characteristics — natural pitch variation, expressive timbral dynamics, and organic energy fluctuations are consistent with an authentic human vocal performance."
        elif confidence >= 0.75:
            tier = "Strong"
            desc = "The model identified clear human vocal characteristics — natural spectral variation and expressive dynamics in the performance are consistent with a real human recording."
        elif confidence >= 0.60:
            tier = "Moderate"
            desc = "The audio shows mostly natural characteristics — organic pitch movement and timbral variation suggest a real performance, though some production processing may be affecting the signal."
        else:
            tier = "Borderline"
            desc = "The model marginally classified this as Real. The audio shares characteristics with both human and AI-generated audio. Heavy studio processing or vocal effects may be influencing the result. Manual review is recommended."

    return tier, desc

# =============================
# Gauge Chart
# =============================
def make_gauge(ai_prob, label):
    is_ai = label == "AI-generated"
    color = "#f43f5e" if is_ai else "#00f5c4"
    bg = "#1e293b"

    val = ai_prob if is_ai else 1 - ai_prob

    fig, ax = plt.subplots(figsize=(4.2, 2.8))

    fig.patch.set_facecolor("#0f1729")
    ax.set_facecolor("#0f1729")

    # Background arc
    arc_bg = mpatches.Wedge(
        center=(0, 0),
        r=1.0,
        theta1=180,
        theta2=0,
        width=0.22,
        facecolor=bg,
        edgecolor="none"
    )
    ax.add_patch(arc_bg)

    # Filled arc
    filled_angle = 180 * val

    arc_fill = mpatches.Wedge(
        center=(0, 0),
        r=1.0,
        theta1=180 - filled_angle,
        theta2=180,
        width=0.22,
        facecolor=color,
        edgecolor="none"
    )
    ax.add_patch(arc_fill)

    # Percentage text
    ax.text(
        0,
        -0.10,
        f"{val*100:.1f}%",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=color,
        fontfamily="monospace"
    )

    # Label text
    ax.text(
        0,
        -0.38,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color="#475569",
        fontfamily="monospace"
    )

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.05, 1.05)

    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout(pad=0.6)

    return fig

# =============================
# Comparison Bar Chart
# =============================
def make_comparison_chart(ai_mp3, ai_wav):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#080c14")

    categories = ["MP3 Model", "WAV Model"]
    ai_vals    = [ai_mp3, ai_wav]
    real_vals  = [1 - ai_mp3, 1 - ai_wav]
    x          = np.arange(len(categories))
    width      = 0.35

    bars_ai   = ax.bar(x - width/2, ai_vals,   width, label="AI Score",   color="#f43f5e", alpha=0.85, zorder=3)
    bars_real = ax.bar(x + width/2, real_vals, width, label="Real Score", color="#00f5c4", alpha=0.85, zorder=3)

    ax.set_ylim(0, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, color="#64748b", fontsize=9, fontfamily="monospace")
    ax.set_ylabel("Probability", color="#475569", fontsize=8, fontfamily="monospace")
    ax.tick_params(colors="#334155")
    ax.set_facecolor("#080c14")
    ax.spines[:].set_color("#1e293b")
    ax.yaxis.label.set_color("#475569")
    ax.tick_params(axis="y", colors="#334155")

    ax.axhline(0.5, color="#1e293b", linewidth=1, linestyle="--", zorder=2)
    ax.text(1.55, 0.51, "threshold", color="#334155", fontsize=7, fontfamily="monospace")

    for bar in bars_ai:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f"{h*100:.1f}%", ha="center", va="bottom",
                fontsize=8, color="#f43f5e", fontfamily="monospace")

    for bar in bars_real:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                f"{h*100:.1f}%", ha="center", va="bottom",
                fontsize=8, color="#00f5c4", fontfamily="monospace")

    ax.legend(
        facecolor="#0f1729", edgecolor="#1e293b",
        labelcolor="#64748b", fontsize=8
    )

    plt.tight_layout()
    return fig

# =============================
# Render Model Card
# =============================
def render_card(label, ai_prob, indicators, model_type):
    confidence  = ai_prob if label == "AI-generated" else 1 - ai_prob
    tier, desc  = interpret_prediction(label, ai_prob)
    is_ai       = label == "AI-generated"
    card_class  = "model-card model-card-ai" if is_ai else "model-card"
    label_class = "model-label-ai" if is_ai else "model-label-real"
    tier_class  = "tier-badge tier-ai" if is_ai else "tier-badge tier-real"
    icon        = "🔴" if is_ai else "🟢"

    matching = [i for i in indicators if i["lean"] == label]

    # Card header — static HTML (no dynamic indicator injection)
    st.markdown(f"""
    <div class="{card_class}">
        <div class="model-tag">// {model_type} Model</div>
        <div class="{label_class}">{icon} {label}</div>
        <div class="model-confidence">
    <div><strong>Confidence:</strong> {confidence*100:.2f}%</div>
    <div><strong>AI:</strong> {ai_prob*100:.2f}%</div>
    <div><strong>Real:</strong> {(1-ai_prob)*100:.2f}%</div>
</div>
        <span class="{tier_class}">{tier} Confidence</span>
        <div class="desc-text">{desc}</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
<div class="section-header">// Supporting Feature Indicators</div>
""", unsafe_allow_html=True)

    # Indicators — each rendered separately
    if matching:
        for ind in matching:
            ind_icon  = "🔴" if ind["lean"] == "AI-generated" else "🟢"
            ind_class = "ind-ai" if ind["lean"] == "AI-generated" else "ind-real"
            st.markdown(f"""
            <div class="indicator-row">
                <span class="{ind_class}">{ind_icon} <strong>{ind['feature']}</strong></span>
                <span style="color:#475569"> — {ind['observation']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="indicator-row" style="color:#334155">
            No strongly matching indicators for this prediction.
        </div>
        """, unsafe_allow_html=True)
# =============================
# Prediction
# =============================
def predict(model, audio_path):
    X_spec, X_temp, feature_values = extract_features(audio_path)
    prob    = float(model.predict([X_spec, X_temp], verbose=0)[0][0])
    ai_prob = prob
    label   = "AI-generated" if ai_prob >= THRESHOLD else "Real"
    confidence = ai_prob if label == "AI-generated" else 1 - ai_prob
    indicators = analyze_features(feature_values)
    return label, confidence, ai_prob, indicators

# =============================
# UI
# =============================
st.markdown('<div class="hero-title">SynthDetect</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">// AI-Generated Audio Detection System</div>', unsafe_allow_html=True)

st.markdown('<div class="upload-label">// Upload Audio File</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="",
    type=["mp3", "wav"],
    label_visibility="collapsed"
)

if uploaded_file is not None:

    ext            = os.path.splitext(uploaded_file.name)[1].lower()
    temp_audio_path = f"temp_audio{ext}"

    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Analyzing audio..."):
        label_mp3, conf_mp3, ai_mp3, indicators_mp3 = predict(model_mp3, temp_audio_path)
        label_wav, conf_wav, ai_wav, indicators_wav = predict(model_wav, temp_audio_path)

    # ---- Gauges ----
    st.markdown('<div class="section-header">// Detection Results</div>', unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown('<div style="text-align:center; font-family:monospace; font-size:0.65rem; color:#334155; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.3rem;">MP3 Model</div>', unsafe_allow_html=True)
        st.pyplot(make_gauge(ai_mp3, label_mp3), use_container_width=True)
    with g2:
        st.markdown('<div style="text-align:center; font-family:monospace; font-size:0.65rem; color:#334155; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:0.3rem;">WAV Model</div>', unsafe_allow_html=True)
        st.pyplot(make_gauge(ai_wav, label_wav), use_container_width=True)

    # ---- Model Cards ----
    st.markdown('<div class="section-header">// Model Analysis</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        render_card(label_mp3, ai_mp3, indicators_mp3, "MP3")
    with c2:
        render_card(label_wav, ai_wav, indicators_wav, "WAV")

    # ---- Comparison Chart ----
    st.markdown('<div class="section-header">// Probability Comparison</div>', unsafe_allow_html=True)
    st.pyplot(make_comparison_chart(ai_mp3, ai_wav), use_container_width=True)

    os.remove(temp_audio_path)