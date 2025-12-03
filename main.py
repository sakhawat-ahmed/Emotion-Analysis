import os
import json
import pickle
import numpy as np
import pandas as pd
import re
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .model-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .positive {
        color: #10B981;
        font-weight: bold;
    }
    .negative {
        color: #EF4444;
        font-weight: bold;
    }
    .confidence-bar {
        height: 20px;
        background-color: #E5E7EB;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .overall-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class SentimentAnalyzer:
    """Loads and uses saved sentiment analysis models"""
    
    def __init__(self, models_dir="saved_models"):
        self.models_dir = models_dir
        self.models = {}
        self.tokenizer = None
        self.params = {}
        self.load_resources()
        
        # Define stopwords for preprocessing
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
            'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
            'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
            'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
            't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 
            're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 
            'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 
            'wasn', 'weren', 'won', 'wouldn'
        }
    
    def load_resources(self):
        """Load all necessary resources"""
        try:
            # Load parameters
            params_path = os.path.join(self.models_dir, 'params.json')
            with open(params_path, 'r') as f:
                self.params = json.load(f)
            
            # Load tokenizer
            tokenizer_path = os.path.join(self.models_dir, 'tokenizer.pickle')
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Load models
            model_files = {
                'RNN': 'rnn_model.h5',
                'LSTM': 'lstm_model.h5',
                'GRU': 'gru_model.h5'
            }
            
            for model_name, filename in model_files.items():
                model_path = os.path.join(self.models_dir, filename)
                if os.path.exists(model_path):
                    st.sidebar.success(f"✓ Loaded {model_name} model")
                    self.models[model_name] = keras.models.load_model(model_path)
                else:
                    st.sidebar.warning(f"⚠ {model_name} model not found")
            
        except Exception as e:
            st.error(f"Error loading resources: {e}")
            raise
    
    def preprocess_text(self, text):
        """Preprocess input text"""
        if not isinstance(text, str):
            return ""
        
        text_cleaning_re = r'@\S+|https?:\S+|http?:\S|[^A-Za-z0-9\s]+'
        text = re.sub(text_cleaning_re, ' ', text.lower()).strip()
        tokens = [word for word in text.split() if word not in self.stop_words]
        return ' '.join(tokens)
    
    def predict_sentiment(self, text):
        """Predict sentiment using all loaded models"""
        # Preprocess text
        cleaned_text = self.preprocess_text(text)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = keras.preprocessing.sequence.pad_sequences(
            sequence,
            maxlen=self.params['MAX_SEQ_LENGTH'],
            padding='post',
            truncating='post'
        )
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                prediction = model.predict(padded_sequence, verbose=0)[0][0]
                sentiment = "positive" if prediction > 0.5 else "negative"
                confidence = prediction if sentiment == "positive" else 1 - prediction
                
                predictions[model_name] = {
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'score': float(prediction),
                    'color': '#10B981' if sentiment == 'positive' else '#EF4444'
                }
                
            except Exception as e:
                predictions[model_name] = {
                    'sentiment': 'error',
                    'confidence': 0.0,
                    'score': 0.0,
                    'color': '#6B7280',
                    'error': str(e)
                }
        
        return predictions

def create_confidence_bar(confidence, color):
    """Create HTML for confidence bar"""
    return f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {confidence*100}%; background-color: {color};"></div>
    </div>
    """

def create_model_card(model_name, prediction):
    """Create HTML for model card"""
    if prediction['sentiment'] == 'error':
        return f"""
        <div class="model-card">
            <h4>❌ {model_name} Model</h4>
            <p style="color: #6B7280;">Error: {prediction.get('error', 'Unknown error')}</p>
        </div>
        """
    
    sentiment_class = "positive" if prediction['sentiment'] == 'positive' else "negative"
    return f"""
    <div class="model-card">
        <h4>🧠 {model_name} Model</h4>
        <p>Prediction: <span class="{sentiment_class}">{prediction['sentiment'].upper()}</span></p>
        <p>Score: <code>{prediction['score']:.8f}</code></p>
        <p>Confidence: {prediction['confidence']:.2%}</p>
        {create_confidence_bar(prediction['confidence'], prediction['color'])}
    </div>
    """

def create_comparison_chart(predictions):
    """Create comparison chart using Plotly"""
    models = list(predictions.keys())
    scores = [predictions[m]['score'] for m in models]
    sentiments = [predictions[m]['sentiment'] for m in models]
    colors = ['#10B981' if s == 'positive' else '#EF4444' for s in sentiments]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=scores,
            text=[f"{s:.4f}" for s in scores],
            textposition='auto',
            marker_color=colors,
            hoverinfo='text',
            hovertext=[f"Model: {m}<br>Score: {s:.6f}<br>Sentiment: {sent}" 
                      for m, s, sent in zip(models, scores, sentiments)]
        )
    ])
    
    fig.update_layout(
        title="Model Predictions Comparison",
        xaxis_title="Model",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                  annotation_text="Decision Boundary", 
                  annotation_position="bottom right")
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("<h1 class='main-header'>🧠 Neural Network Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Analyze text sentiment using RNN, LSTM, and GRU models</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
        st.title("Settings")
        
        # Model selection
        st.subheader("📊 Models")
        show_rnn = st.checkbox("RNN Model", value=True)
        show_lstm = st.checkbox("LSTM Model", value=True)
        show_gru = st.checkbox("GRU Model", value=True)
        
        st.subheader("⚙️ Options")
        show_details = st.checkbox("Show Technical Details", value=False)
        
        # Example sentences
        st.subheader("📝 Examples")
        examples = [
            "I like reading.",
            "This movie was absolutely fantastic!",
            "The service was terrible and disappointing.",
            "Great product, highly recommend it.",
            "I'm not satisfied with the quality.",
            "Amazing experience, will definitely come back."
        ]
        
        selected_example = st.selectbox("Try an example:", examples)
        
        if st.button("Load Example"):
            st.session_state.input_text = selected_example
        
        st.markdown("---")
        st.markdown("### 📊 Project Info")
        st.markdown("""
        **Level 1:** RNN Model  
        **Level 2:** LSTM Model  
        **Level 3:** GRU Model  
        
        For demonstration:  
        Enter "I like reading." and click Analyze
        """)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.analyzer = SentimentAnalyzer()
    
    # Initialize session state for text input
    if 'input_text' not in st.session_state:
        st.session_state.input_text = "I like reading."
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Input section
        st.subheader("📝 Enter Text for Analysis")
        input_text = st.text_area(
            "Type or paste your text here:",
            value=st.session_state.input_text,
            height=150,
            placeholder="Enter text to analyze sentiment..."
        )
        
        col1_1, col1_2, col1_3 = st.columns(3)
        with col1_1:
            analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)
        with col1_2:
            clear_btn = st.button("🗑️ Clear Text", use_container_width=True)
        with col1_3:
            if st.button("📋 Random Example", use_container_width=True):
                import random
                random_examples = [
                    "The customer support was exceptional!",
                    "I hate waiting in long lines.",
                    "Product quality exceeds expectations.",
                    "Very disappointed with the delivery.",
                    "Absolutely love this new feature!"
                ]
                st.session_state.input_text = random.choice(random_examples)
                st.rerun()
    
    with col2:
        # Quick info panel
        st.subheader("ℹ️ Quick Info")
        st.info("""
        **Expected Output Format:**
        - RNN: positive, score: 0.61676633
        - LSTM: positive, score: 0.7692368  
        - GRU: positive, score: 0.7972771
        """)
        
        st.success("✅ All models loaded successfully!")
        st.caption(f"Vocabulary size: {st.session_state.analyzer.params.get('MAX_WORDS', 'N/A')}")
        st.caption(f"Sequence length: {st.session_state.analyzer.params.get('MAX_SEQ_LENGTH', 'N/A')}")
    
    # Handle buttons
    if clear_btn:
        st.session_state.input_text = ""
        st.rerun()
    
    if analyze_btn and input_text.strip():
        with st.spinner("Analyzing sentiment..."):
            # Get predictions
            predictions = st.session_state.analyzer.predict_sentiment(input_text)
            
            # Filter based on sidebar selection
            filtered_models = []
            if show_rnn and 'RNN' in predictions:
                filtered_models.append('RNN')
            if show_lstm and 'LSTM' in predictions:
                filtered_models.append('LSTM')
            if show_gru and 'GRU' in predictions:
                filtered_models.append('GRU')
            
            filtered_predictions = {m: predictions[m] for m in filtered_models}
            
            # Display results in columns
            st.markdown("---")
            st.subheader("📊 Analysis Results")
            
            if filtered_predictions:
                # Create columns for model cards
                cols = st.columns(len(filtered_models))
                for idx, (col, model_name) in enumerate(zip(cols, filtered_models)):
                    with col:
                        st.markdown(create_model_card(model_name, filtered_predictions[model_name]), 
                                   unsafe_allow_html=True)
                
                # Overall sentiment
                st.markdown("---")
                st.subheader("🎯 Overall Sentiment")
                
                # Calculate overall sentiment
                positive_count = sum(1 for p in filtered_predictions.values() 
                                   if p['sentiment'] == 'positive')
                total = len(filtered_predictions)
                
                if positive_count > total / 2:
                    overall = "POSITIVE"
                    overall_color = "#10B981"
                    emoji = "😊"
                elif positive_count < total / 2:
                    overall = "NEGATIVE"
                    overall_color = "#EF4444"
                    emoji = "😞"
                else:
                    overall = "NEUTRAL"
                    overall_color = "#6B7280"
                    emoji = "😐"
                
                # Display overall sentiment card
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {overall_color}20 0%, {overall_color}40 100%);
                        padding: 2rem;
                        border-radius: 15px;
                        border: 2px solid {overall_color};
                        text-align: center;
                        margin: 1rem 0;
                    ">
                        <h1 style="font-size: 4rem; margin: 0;">{emoji}</h1>
                        <h2 style="color: {overall_color}; margin: 0.5rem 0;">{overall}</h2>
                        <p style="color: #6B7280; margin: 0;">
                            {positive_count} out of {total} models predict positive
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Visualization
                st.markdown("---")
                st.subheader("📈 Visual Comparison")
                
                tab1, tab2 = st.tabs(["Bar Chart", "Detailed View"])
                
                with tab1:
                    fig = create_comparison_chart(filtered_predictions)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Create detailed dataframe
                    data = []
                    for model_name, pred in filtered_predictions.items():
                        data.append({
                            'Model': model_name,
                            'Score': pred['score'],
                            'Sentiment': pred['sentiment'].upper(),
                            'Confidence': pred['confidence']
                        })
                    
                    df = pd.DataFrame(data)
                    st.dataframe(
                        df.style.format({
                            'Score': '{:.8f}',
                            'Confidence': '{:.2%}'
                        }).apply(lambda x: ['background-color: #10B98120' if v == 'POSITIVE' 
                                          else 'background-color: #EF444420' 
                                          for v in x], axis=1, subset=['Sentiment']),
                        use_container_width=True
                    )
                
                # Technical details (if enabled)
                if show_details:
                    st.markdown("---")
                    st.subheader("🔧 Technical Details")
                    
                    with st.expander("View preprocessing details"):
                        st.code(f"""
                        Original text: {input_text}
                        Preprocessed: {st.session_state.analyzer.preprocess_text(input_text)}
                        Sequence length: {st.session_state.analyzer.params['MAX_SEQ_LENGTH']}
                        Vocabulary size: {st.session_state.analyzer.params['MAX_WORDS']}
                        """, language="python")
                    
                    with st.expander("View raw predictions"):
                        st.json(predictions)
            else:
                st.warning("No models selected for analysis. Please enable at least one model in the sidebar.")
    
    elif analyze_btn and not input_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #6B7280;">
            <p>Sentiment Analysis Demo • For Teacher's Presentation</p>
            <p>Expected output for "I like reading.":</p>
            <p>RNN: positive, score: 0.61676633<br>
            LSTM: positive, score: 0.7692368<br>
            GRU: positive, score: 0.7972771</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()