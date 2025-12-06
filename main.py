import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import json
import re
import random
import matplotlib.pyplot as plt

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

# -----------------------------------------------------
# MODERN UI STYLING (Glassmorphism + Smooth UI)
# -----------------------------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #eef2ff, #f8fafc);
}

/* Header */
.title {
    font-size: 3rem;
    text-align: center;
    font-weight: 700;
    color: #1e293b;
    margin-top: -1rem;
}
.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #475569;
    margin-bottom: 2rem;
}

/* Card */
.card {
    background: rgba(255,255,255,0.55);
    padding: 1.7rem;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.4);
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 1.5rem;
    transition: 0.3s;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.12);
}

/* Text area */
textarea {
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
    font-size: 1rem !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    padding: 10px 22px;
    border: none;
    background: #3b82f6;
    color: white;
    font-weight: 600;
    transition: 0.25s;
}
.stButton > button:hover {
    background: #1d4ed8;
    box-shadow: 0 4px 14px rgba(30,64,175,0.35);
}

/* Sentiment Colors */
.positive { color:#10b981!important; font-weight:600; }
.negative { color:#ef4444!important; font-weight:600; }
.neutral { color:#64748b!important; font-weight:600; }

.progress {
    width: 100%;
    height: 10px;
    border-radius: 12px;
    background: #e2e8f0;
    margin-top: .5rem;
}
.bar {
    height: 100%;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# SESSION STATE
# -----------------------------------------------------
if "input_text" not in st.session_state:
    st.session_state.input_text = "I love this amazing product!"

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    show_rnn = st.checkbox("Enable RNN Model", True)
    show_lstm = st.checkbox("Enable LSTM Model", True)
    show_gru = st.checkbox("Enable GRU Model", True)

    st.markdown("---")

    st.subheader("Examples")
    examples = [
        "I love this amazing product!",
        "Terrible service, very disappointed.",
        "Exceeded all my expectations!",
        "Not worth the money at all.",
    ]
    for e in examples:
        if st.button(e[:25] + "..."):
            st.session_state.input_text = e
            st.rerun()

    st.markdown("---")
    st.info("This project uses RNN, LSTM, and GRU models trained on Sentiment140 dataset.")

# -----------------------------------------------------
# HEADER
# -----------------------------------------------------
st.markdown('<div class="title">💬 Sentiment Analysis</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze emotions using deep learning models</p>', unsafe_allow_html=True)

# -----------------------------------------------------
# TEXT INPUT
# -----------------------------------------------------
text = st.text_area(
    "Enter your text:",
    value=st.session_state.input_text,
    height=120,
    key="text_area"
)

colA, colB, colC = st.columns(3)
with colA:
    analyze = st.button("🔍 Analyze")
with colB:
    if st.button("🗑 Clear"):
        st.session_state.input_text = ""
        st.rerun()
with colC:
    if st.button("🎲 Random"):
        random_examples = [
            "The sunset was breathtaking!",
            "Worst purchase ever.",
            "I’m extremely satisfied!",
            "This made me very upset.",
        ]
        st.session_state.input_text = random.choice(random_examples)
        st.rerun()


# -----------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------
@st.cache_resource
def load_all_models():
    base = "saved_models"
    models = {}

    try:
        # tokenizer
        with open(os.path.join(base, "tokenizer.pickle"), "rb") as f:
            tokenizer = pickle.load(f)

        # parameters
        with open(os.path.join(base, "params.json"), "r") as f:
            params = json.load(f)

        # model files
        names = {
            "RNN": "rnn_model.h5",
            "LSTM": "lstm_model.h5",
            "GRU": "gru_model.h5"
        }

        for name, file in names.items():
            path = os.path.join(base, file)
            models[name] = keras.models.load_model(path) if os.path.exists(path) else None

        return models, tokenizer, params

    except Exception as e:
        st.error("❌ Error loading models: " + str(e))
        return None, None, None


# -----------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------
def clean_text(t):
    return re.sub(r'@\S+|https?:\S+|[^A-Za-z0-9 ]+', ' ', t.lower()).strip()


# -----------------------------------------------------
# PREDICT
# -----------------------------------------------------
def predict(text, models, tokenizer, max_len):
    seq = tokenizer.texts_to_sequences([clean_text(text)])
    padded = keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding="post")
    results = {}
    for name, model in models.items():
        if model:
            raw = float(model.predict(padded, verbose=0)[0][0])
            sentiment = "positive" if raw > .5 else "negative"
            conf = raw if sentiment == "positive" else 1 - raw
            results[name] = (sentiment, raw, conf)
    return results


# -----------------------------------------------------
# PROCESSING
# -----------------------------------------------------
if analyze and text.strip():
    models, tokenizer, params = load_all_models()

    if models:
        selected_models = {
            "RNN": show_rnn,
            "LSTM": show_lstm,
            "GRU": show_gru
        }

        usable = {k: v for k, v in models.items() if selected_models[k] and v}

        if len(usable) == 0:
            st.warning("⚠️ No model selected!")
        else:
            preds = predict(text, usable, tokenizer, params["MAX_SEQ_LENGTH"])

            st.subheader("📊 Model Outputs")

            cols = st.columns(len(usable))
            for i, (name, (sent, score, conf)) in enumerate(preds.items()):
                color = "#10b981" if sent == "positive" else "#ef4444"

                with cols[i]:
                    st.markdown(
                        f"""
                        <div class="card">
                            <h3 style='margin-bottom:4px'>{name}</h3>
                            <p class="{sent}">{sent.upper()}</p>
                            <div class="progress">
                                <div class="bar" style="width:{conf*100}%; background:{color}"></div>
                            </div>
                            <p style="margin-top:8px; font-size:13px;">Score: {score:.4f}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.subheader("🎯 Overall Sentiment")
            positive_votes = sum(1 for v in preds.values() if v[0] == "positive")

            if positive_votes > len(preds)/2:
                overall = ("Positive 😊", "#10b981")
            elif positive_votes < len(preds)/2:
                overall = ("Negative 😞", "#ef4444")
            else:
                overall = ("Neutral 😐", "#64748b")

            st.markdown(
                f"""
                <div class="card" style="text-align:center;">
                    <h2 style="color:{overall[1]}">{overall[0]}</h2>
                    <p>{positive_votes} out of {len(preds)} models voted positive</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# -----------------------------------------------------
# END
# -----------------------------------------------------
