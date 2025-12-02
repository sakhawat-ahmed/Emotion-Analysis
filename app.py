# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import pickle
import io
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Demo",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .positive {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
    }
    .negative {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
    }
    .model-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: #f8f9fa;
        box-shadow: 0 2px 4px rgba(0,0,0,.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🧠 Sentiment Analysis Demo</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; margin-bottom: 2rem;">
    Real-time sentiment analysis using RNN, LSTM, and GRU models<br>
    <small>Built for classroom presentation with live demonstration</small>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103655.png", width=100)
    st.title("Navigation")
    page = st.radio(
        "Choose a section:",
        ["🏠 Home", "📊 Live Demo", "📈 Model Comparison", "📚 About"]
    )
    
    st.markdown("---")
    st.markdown("### Model Settings")
    show_details = st.checkbox("Show technical details", True)
    
    st.markdown("---")
    st.markdown("### Sample Sentences")
    sample_sentences = [
        "I like reading.",
        "This is absolutely wonderful!",
        "I hate this product, it's terrible.",
        "The movie was okay, not great but not bad either.",
        "Excellent service and amazing quality!",
        "Worst experience of my life."
    ]
    
    st.markdown("Try these examples:")
    for sentence in sample_sentences:
        if st.button(f"📝 {sentence[:30]}...", key=f"btn_{sentence[:10]}"):
            st.session_state.demo_text = sentence

# Initialize session state
if 'demo_text' not in st.session_state:
    st.session_state.demo_text = "I like reading."

# Text preprocessing function
def preprocess_text(text, stem=False):
    """Clean and preprocess text data"""
    text = str(text).lower()
    text_cleaning_re = r'@\S+|https?:\S+|http?:\S|[^A-Za-z0-9\s]+'
    text = re.sub(text_cleaning_re, ' ', text)
    text = ' '.join(text.split())
    return text

# Load models (with caching to avoid reloading)
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        rnn_model = tf.keras.models.load_model('sentiment_rnn_final.h5')
        lstm_model = tf.keras.models.load_model('sentiment_lstm_final.h5')
        gru_model = tf.keras.models.load_model('sentiment_gru_final.h5')
        
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        return rnn_model, lstm_model, gru_model, tokenizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Prediction function
def predict_sentiment(text, models, tokenizer, max_length=30):
    """Predict sentiment for given text"""
    # Preprocess
    clean_text = preprocess_text(text)
    
    # Tokenize
    sequence = tokenizer.texts_to_sequences([clean_text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
        sequence, maxlen=max_length, padding='post', truncating='post'
    )
    
    # Get predictions
    rnn_pred = float(models[0].predict(padded_sequence, verbose=0)[0][0])
    lstm_pred = float(models[1].predict(padded_sequence, verbose=0)[0][0])
    gru_pred = float(models[2].predict(padded_sequence, verbose=0)[0][0])
    
    return {
        'text': text,
        'clean_text': clean_text,
        'rnn': {'score': rnn_pred, 'sentiment': 'positive' if rnn_pred > 0.5 else 'negative'},
        'lstm': {'score': lstm_pred, 'sentiment': 'positive' if lstm_pred > 0.5 else 'negative'},
        'gru': {'score': gru_pred, 'sentiment': 'positive' if gru_pred > 0.5 else 'negative'}
    }

# Load models
rnn_model, lstm_model, gru_model, tokenizer = load_models()
models_loaded = all([rnn_model, lstm_model, gru_model, tokenizer])

# Home Page
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Project Goal")
        st.info("""
        Build a classifier using recurrent neural networks 
        to recognize positive and negative emotions in text.
        """)
    
    with col2:
        st.markdown("### 🤖 Models Used")
        st.success("""
        • **RNN** - Simple Recurrent Network
        • **LSTM** - Long Short-Term Memory  
        • **GRU** - Gated Recurrent Unit
        """)
    
    with col3:
        st.markdown("### 📊 Dataset")
        st.warning("""
        **Sentiment140 Dataset**
        - 1.6 million labeled tweets
        - Binary classification
        - Positive/Negative sentiment
        """)
    
    st.markdown("---")
    
    # Quick start section
    st.markdown("### 🚀 Quick Start")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        1. Go to **Live Demo** tab
        2. Enter a sentence
        3. Click **Analyze Sentiment**
        4. See predictions from all 3 models
        """)
        
        if st.button("🚀 Go to Live Demo", use_container_width=True):
            st.session_state.page = "📊 Live Demo"
            st.rerun()
    
    with col2:
        # Display sample prediction
        if models_loaded:
            sample_result = predict_sentiment("I like reading.", 
                                            (rnn_model, lstm_model, gru_model), 
                                            tokenizer)
            
            st.metric("Sample Prediction", 
                     sample_result['rnn']['sentiment'].title(),
                     delta=f"Score: {sample_result['rnn']['score']:.3f}")
    
    # Architecture comparison
    st.markdown("---")
    st.markdown("### 🏗️ Model Architectures")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("RNN Architecture", expanded=True):
            st.markdown("""
            **Simple Recurrent Network**
            - Single hidden state
            - Simple but effective
            - Suffers from vanishing gradient
            """)
            st.code("""
            Input → Embedding
                  → SimpleRNN
                  → Dense → Output
            """, language="python")
    
    with col2:
        with st.expander("LSTM Architecture", expanded=True):
            st.markdown("""
            **Long Short-Term Memory**
            - Memory cells with gates
            - Handles long sequences
            - Input/Forget/Output gates
            """)
            st.code("""
            Input → Embedding
                  → LSTM
                  → Dense → Output
            """, language="python")
    
    with col3:
        with st.expander("GRU Architecture", expanded=True):
            st.markdown("""
            **Gated Recurrent Unit**
            - Simplified LSTM
            - Update/Reset gates
            - Faster training
            """)
            st.code("""
            Input → Embedding
                  → GRU
                  → Dense → Output
            """, language="python")

# Live Demo Page
elif page == "📊 Live Demo":
    st.markdown('<h2 class="sub-header">🎤 Live Sentiment Analysis</h2>', unsafe_allow_html=True)
    
    if not models_loaded:
        st.error("⚠️ Models not loaded. Please ensure model files exist in the current directory.")
        st.info("""
        Required files:
        1. `sentiment_rnn_final.h5`
        2. `sentiment_lstm_final.h5`
        3. `sentiment_gru_final.h5`
        4. `tokenizer.pickle`
        
        Run the training script first to generate these files.
        """)
        st.stop()
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        input_text = st.text_area(
            "📝 Enter text to analyze:",
            value=st.session_state.demo_text,
            height=100,
            placeholder="Type a sentence here...",
            key="input_text"
        )
    
    with col2:
        st.markdown("### ")
        st.markdown("### ")
        analyze_btn = st.button("🔍 Analyze Sentiment", 
                              type="primary", 
                              use_container_width=True)
        
        clear_btn = st.button("🗑️ Clear", 
                            use_container_width=True)
        
        if clear_btn:
            input_text = ""
            st.session_state.demo_text = ""
            st.rerun()
    
    if analyze_btn and input_text.strip():
        with st.spinner("🔮 Analyzing sentiment..."):
            # Add progress bar for visual effect
            progress_bar = st.progress(0)
            
            # Simulate processing steps
            for i in range(3):
                time.sleep(0.2)
                progress_bar.progress((i + 1) * 33)
            
            # Get predictions
            result = predict_sentiment(input_text, 
                                     (rnn_model, lstm_model, gru_model), 
                                     tokenizer)
            
            progress_bar.progress(100)
            time.sleep(0.2)
            progress_bar.empty()
        
        # Display results in a nice layout
        st.markdown("---")
        st.markdown(f"### 📊 Analysis Results")
        st.markdown(f"**Original text:** `{result['text']}`")
        st.markdown(f"**Cleaned text:** `{result['clean_text']}`")
        
        # Create columns for each model
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rnn_sentiment = result['rnn']['sentiment']
            rnn_score = result['rnn']['score']
            
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.markdown("### 🔄 RNN Model")
            
            # Color-coded sentiment display
            if rnn_sentiment == 'positive':
                st.markdown(f'<div class="prediction-box positive">', unsafe_allow_html=True)
                st.metric("Sentiment", "😊 Positive", 
                         delta=f"{rnn_score:.3f}")
            else:
                st.markdown(f'<div class="prediction-box negative">', unsafe_allow_html=True)
                st.metric("Sentiment", "😞 Negative", 
                         delta=f"{rnn_score:.3f}")
            
            st.markdown(f"**Confidence:** {rnn_score:.4f}")
            
            # Progress bar for score visualization
            st.progress(rnn_score if rnn_sentiment == 'positive' else 1 - rnn_score)
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        with col2:
            lstm_sentiment = result['lstm']['sentiment']
            lstm_score = result['lstm']['score']
            
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.markdown("### 🔐 LSTM Model")
            
            if lstm_sentiment == 'positive':
                st.markdown(f'<div class="prediction-box positive">', unsafe_allow_html=True)
                st.metric("Sentiment", "😊 Positive", 
                         delta=f"{lstm_score:.3f}")
            else:
                st.markdown(f'<div class="prediction-box negative">', unsafe_allow_html=True)
                st.metric("Sentiment", "😞 Negative", 
                         delta=f"{lstm_score:.3f}")
            
            st.markdown(f"**Confidence:** {lstm_score:.4f}")
            st.progress(lstm_score if lstm_sentiment == 'positive' else 1 - lstm_score)
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        with col3:
            gru_sentiment = result['gru']['sentiment']
            gru_score = result['gru']['score']
            
            st.markdown('<div class="model-card">', unsafe_allow_html=True)
            st.markdown("### ⚙️ GRU Model")
            
            if gru_sentiment == 'positive':
                st.markdown(f'<div class="prediction-box positive">', unsafe_allow_html=True)
                st.metric("Sentiment", "😊 Positive", 
                         delta=f"{gru_score:.3f}")
            else:
                st.markdown(f'<div class="prediction-box negative">', unsafe_allow_html=True)
                st.metric("Sentiment", "😞 Negative", 
                         delta=f"{gru_score:.3f}")
            
            st.markdown(f"**Confidence:** {gru_score:.4f}")
            st.progress(gru_score if gru_sentiment == 'positive' else 1 - gru_score)
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Comparison visualization
        st.markdown("---")
        st.markdown("### 📈 Model Comparison")
        
        # Create comparison chart
        fig, ax = plt.subplots(figsize=(10, 4))
        models = ['RNN', 'LSTM', 'GRU']
        scores = [rnn_score, lstm_score, gru_score]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = ax.bar(models, scores, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add threshold line at 0.5
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Decision Boundary')
        
        ax.set_ylabel('Sentiment Score', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.set_title('Model Prediction Scores Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        st.pyplot(fig)
        
        # Technical details (optional)
        if show_details:
            with st.expander("🔧 Technical Details"):
                st.markdown("**Prediction Details:**")
                st.json({
                    'text': result['text'],
                    'clean_text': result['clean_text'],
                    'predictions': {
                        'RNN': {'score': float(rnn_score), 'sentiment': rnn_sentiment},
                        'LSTM': {'score': float(lstm_score), 'sentiment': lstm_sentiment},
                        'GRU': {'score': float(gru_score), 'sentiment': gru_sentiment}
                    }
                })
        
        # History tracking
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        st.session_state.history.append({
            'text': input_text,
            'rnn': rnn_sentiment,
            'lstm': lstm_sentiment,
            'gru': gru_sentiment,
            'timestamp': time.strftime("%H:%M:%S")
        })
        
        # Keep only last 5 entries
        st.session_state.history = st.session_state.history[-5:]
        
        # Show history
        if st.session_state.history:
            st.markdown("---")
            st.markdown("### 📜 Recent Predictions")
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True)

# Model Comparison Page
elif page == "📈 Model Comparison":
    st.markdown('<h2 class="sub-header">📊 Model Performance Comparison</h2>', unsafe_allow_html=True)
    
    # Load comparison data
    try:
        comparison_df = pd.read_csv('model_comparison_results.csv')
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
            st.metric("🏆 Best Model", best_model)
        
        with col2:
            best_acc = comparison_df['Accuracy'].max()
            st.metric("📈 Best Accuracy", f"{best_acc:.3f}")
        
        with col3:
            best_f1 = comparison_df['F1-Score'].max()
            st.metric("🎯 Best F1-Score", f"{best_f1:.3f}")
        
        with col4:
            fastest = comparison_df.loc[comparison_df['Training Time (s)'].idxmin(), 'Model']
            st.metric("⚡ Fastest", fastest)
        
        # Comparison charts
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📊 Performance Metrics", "⏱️ Training Time", "🧮 Model Complexity"])
        
        with tab1:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx//2, idx%2]
                bars = ax.bar(comparison_df['Model'], comparison_df[metric], color=colors)
                ax.set_title(f'{metric} Comparison', fontweight='bold')
                ax.set_ylabel(metric)
                ax.set_ylim([0.7, 1.0])
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(comparison_df['Model'], comparison_df['Training Time (s)'], 
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title('Training Time Comparison', fontweight='bold')
            ax.set_ylabel('Time (seconds)')
            ax.set_xlabel('Model')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
        
        with tab3:
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(comparison_df['Model'], comparison_df['Parameters'], 
                         color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
            ax.set_title('Model Complexity (Number of Parameters)', fontweight='bold')
            ax.set_ylabel('Parameters')
            ax.set_xlabel('Model')
            ax.grid(True, alpha=0.3, axis='y')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5000,
                       f'{height:,}', ha='center', va='bottom', fontsize=9)
            
            st.pyplot(fig)
        
        # Technical analysis
        st.markdown("---")
        with st.expander("🔍 Technical Analysis", expanded=True):
            st.markdown("""
            ### Model Architecture Analysis
            
            **RNN (Simple Recurrent Network):**
            - ✅ Simple architecture with fewer parameters
            - ✅ Faster training time
            - ❌ Suffers from vanishing gradient problem
            - ❌ Struggles with long-term dependencies
            
            **LSTM (Long Short-Term Memory):**
            - ✅ Handles long-term dependencies well
            - ✅ Memory cells with gating mechanisms
            - ❌ More complex architecture
            - ❌ Higher computational cost
            
            **GRU (Gated Recurrent Unit):**
            - ✅ Simplified LSTM with fewer gates
            - ✅ Faster training than LSTM
            - ✅ Good balance of performance and efficiency
            - ❌ May not capture very long dependencies as well as LSTM
            
            **Conclusion:**
            For sentiment analysis tasks, GRU often provides the best trade-off between 
            accuracy and computational efficiency.
            """)
            
            # Show the dataframe
            st.dataframe(comparison_df, use_container_width=True)
    
    except FileNotFoundError:
        st.warning("Comparison data not found. Run the training script to generate comparison results.")

# About Page
elif page == "📚 About":
    st.markdown('<h2 class="sub-header">📖 About This Project</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Overview
        
        This project implements three different Recurrent Neural Network architectures
        for sentiment analysis on the Sentiment140 dataset.
        
        ### 📊 Dataset Information
        
        **Sentiment140 Dataset:**
        - Contains 1.6 million tweets
        - Binary sentiment labels (positive/negative)
        - Collected using Twitter API
        - Widely used for sentiment analysis research
        
        ### 🏗️ Technical Implementation
        
        **Preprocessing Steps:**
        1. Text cleaning and normalization
        2. Tokenization and sequence padding
        3. Word embeddings using GloVe
        
        **Model Architectures:**
        1. **RNN:** Simple recurrent network
        2. **LSTM:** Long short-term memory
        3. **GRU:** Gated recurrent unit
        
        **Evaluation Metrics:**
        - Accuracy
        - Precision
        - Recall
        - F1-Score
        - Training time
        - Model complexity
        
        ### 🎓 Educational Purpose
        
        This project was developed for academic presentation and demonstration
        of different RNN architectures for natural language processing tasks.
        """)
    
    with col2:
        st.markdown("### 🛠️ Technologies Used")
        
        tech_list = {
            "Python": "https://www.python.org/",
            "TensorFlow": "https://www.tensorflow.org/",
            "Streamlit": "https://streamlit.io/",
            "Scikit-learn": "https://scikit-learn.org/",
            "Pandas": "https://pandas.pydata.org/",
            "NumPy": "https://numpy.org/",
            "Matplotlib": "https://matplotlib.org/",
            "NLTK": "https://www.nltk.org/"
        }
        
        for tech, url in tech_list.items():
            st.markdown(f"• [{tech}]({url})")
        
        st.markdown("---")
        st.markdown("### 📁 Project Structure")
        
        structure = """
        project/
        ├── app.py                # This Streamlit app
        ├── train.py              # Training script
        ├── sentiment_*.h5        # Trained models
        ├── tokenizer.pickle      # Tokenizer object
        ├── requirements.txt      # Dependencies
        └── README.md             # Documentation
        """
        
        st.code(structure, language="bash")
    
    st.markdown("---")
    st.markdown("### 👨‍💻 How to Run This Project")
    
    with st.expander("Setup Instructions", expanded=False):
        st.markdown("""
        1. **Install dependencies:**
        ```bash
        pip install streamlit tensorflow pandas numpy scikit-learn matplotlib seaborn
        ```
        
        2. **Run the training script (optional):**
        ```bash
        python train.py
        ```
        
        3. **Launch the web app:**
        ```bash
        streamlit run app.py
        ```
        
        4. **Open browser and navigate to:**
        ```
        http://localhost:8501
        ```
        """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>🎓 Academic Project | Sentiment Analysis Demo</p>
            <p>For presentation and educational purposes</p>
        </div>
        """, unsafe_allow_html=True)

# Footer for all pages
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.8rem;">
    <p>Sentiment Analysis Demo • RNN/LSTM/GRU Models • Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)