#!/usr/bin/env python3
"""
Audio Emotion Recognition Demo Interface
Interactive Streamlit app for demonstrating the trained emotion recognition model
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import os
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Audio Emotion Recognition Demo",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .emotion-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-high {
        background-color: #4caf50;
        color: white;
    }
    .prediction-medium {
        background-color: #ff9800;
        color: white;
    }
    .prediction-low {
        background-color: #f44336;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Model Architecture (same as in model.py)
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        query = self.query(x).view(batch_size, -1, height * width)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        attention = torch.bmm(query.transpose(1, 2), key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.transpose(1, 2))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

class EnhancedCNN(nn.Module):
    """Enhanced CNN with attention and deeper architecture"""
    def __init__(self, num_classes=7):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        
        self.attention = AttentionBlock(128)
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.attention(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

# Preprocessing functions
def get_sound_data(path, target_sr=44100):
    """Load and resample audio file"""
    data, orig_sr = sf.read(path)
    data_resample = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
    
    # Convert to mono if stereo
    if len(data_resample.shape) > 1:
        data_resample = np.average(data_resample, axis=1)
    return data_resample, target_sr

def windows(data, window_size):
    """Generate overlapping windows"""
    start = 0
    while start + window_size < len(data):
        yield start, start + window_size
        start += (window_size // 2)

def extract_features_from_audio(audio_data, sr=44100, bands=128, frames=128, hop_length=512, n_fft=1024):
    """Extract MFCC features with delta and delta-delta from audio data"""
    window_size = hop_length * (frames - 1)
    features = []
    
    for start, end in windows(audio_data, window_size):
        signal = audio_data[start:end]
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=signal, 
            sr=sr, 
            n_mfcc=bands,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Ensure correct frame count
        if mfcc.shape[1] != frames:
            if mfcc.shape[1] > frames:
                mfcc = mfcc[:, :frames]
            else:
                pad_width = frames - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        
        # Create 3-channel feature map (MFCC + Delta + Delta-Delta)
        feature_3d = np.zeros((bands, frames, 3))
        feature_3d[:, :, 0] = mfcc  # Original MFCC
        feature_3d[:, :, 1] = librosa.feature.delta(mfcc, order=2)  # Delta-Delta
        feature_3d[:, :, 2] = librosa.feature.delta(mfcc)  # Delta
        
        features.append(feature_3d)
    
    return np.array(features)

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = EnhancedCNN(num_classes=7)
        
        # Load model weights
        model_path = "enhanced_best_model.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            return model, device
        else:
            st.error(f"Model file '{model_path}' not found!")
            return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def predict_emotion(model, device, features):
    """Predict emotion from features"""
    # Emotion labels
    emotions = ['fear', 'disgust', 'happy', 'angry', 'sad', 'neutral', 'boredom']
    
    with torch.no_grad():
        # Convert to tensor and reshape for model input (batch, channels, height, width)
        features_tensor = torch.FloatTensor(features).permute(0, 3, 1, 2).to(device)
        
        # Get predictions for all windows
        outputs = model(features_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Average predictions across all windows
        avg_probabilities = probabilities.mean(dim=0)
        predicted_class = torch.argmax(avg_probabilities).item()
        confidence = avg_probabilities[predicted_class].item()
        
        # Get all class probabilities
        all_probs = avg_probabilities.cpu().numpy()
        
        return emotions[predicted_class], confidence, dict(zip(emotions, all_probs))

def plot_waveform(audio_data, sr):
    """Create waveform plot"""
    fig = go.Figure()
    time_axis = np.linspace(0, len(audio_data) / sr, len(audio_data))
    
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=audio_data,
        mode='lines',
        name='Waveform',
        line=dict(color='#1e88e5', width=1)
    ))
    
    fig.update_layout(
        title='Audio Waveform',
        xaxis_title='Time (seconds)',
        yaxis_title='Amplitude',
        template='plotly_white',
        height=300
    )
    
    return fig

def plot_mfcc_spectrogram(mfcc_features):
    """Create MFCC spectrogram plot"""
    fig = go.Figure(data=go.Heatmap(
        z=mfcc_features,
        colorscale='Viridis',
        showscale=True
    ))
    
    fig.update_layout(
        title='MFCC Spectrogram',
        xaxis_title='Time Frames',
        yaxis_title='MFCC Coefficients',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_emotion_probabilities(emotion_probs):
    """Create emotion probability bar chart"""
    emotions = list(emotion_probs.keys())
    probabilities = list(emotion_probs.values())
    
    # Color bars based on probability
    colors = ['#4caf50' if p > 0.3 else '#ff9800' if p > 0.1 else '#f44336' for p in probabilities]
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=probabilities,
            marker_color=colors,
            text=[f'{p:.3f}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Emotion Prediction Probabilities',
        xaxis_title='Emotions',
        yaxis_title='Probability',
        template='plotly_white',
        height=400
    )
    
    return fig

# Main app
def main():
    st.markdown('<h1 class="main-header">🎵 Audio Emotion Recognition Demo</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an audio file to analyze its emotional content</p>', unsafe_allow_html=True)
    
    # Load model
    model, device = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please ensure 'enhanced_best_model.pth' is in the current directory.")
        return
    
    # Sidebar
    st.sidebar.markdown("## 📊 Model Information")
    st.sidebar.markdown("""
    **Architecture:** Enhanced CNN with Attention  
    **Accuracy:** 95.14%  
    **F1 Score:** 95.10%  
    **Emotions:** 7 classes  
    **Features:** MFCC + Delta + Delta-Delta  
    """)
    
    st.sidebar.markdown("## 🎯 Supported Emotions")
    emotions_info = {
        "😨 Fear": "Anxiety, terror, fright",
        "🤢 Disgust": "Revulsion, distaste",
        "😊 Happy": "Joy, pleasure, contentment",
        "😡 Angry": "Rage, irritation, fury",
        "😢 Sad": "Sorrow, melancholy, grief",
        "😐 Neutral": "Calm, balanced, unemotional",
        "😴 Boredom": "Disinterest, weariness"
    }
    
    for emotion, description in emotions_info.items():
        st.sidebar.markdown(f"**{emotion}**: {description}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎤 Audio Input")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'ogg'],
            help="Upload a WAV, MP3, FLAC, or OGG audio file"
        )
        
        # Sample files option
        st.markdown("### 📁 Or try sample files:")
        sample_files = list(Path("emoDB/wav").glob("*.wav"))[:10] if Path("emoDB/wav").exists() else []
        
        if sample_files:
            selected_sample = st.selectbox(
                "Select a sample audio file:",
                options=[""] + [f.name for f in sample_files]
            )
            
            if selected_sample and st.button("🎵 Analyze Sample"):
                uploaded_file = selected_sample
        
        # Process audio
        if uploaded_file is not None:
            try:
                # Handle file input
                if isinstance(uploaded_file, str):
                    # Sample file selected
                    audio_path = Path("emoDB/wav") / uploaded_file
                    st.audio(str(audio_path))
                else:
                    # Uploaded file
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        audio_path = tmp_file.name
                    
                    st.audio(uploaded_file)
                
                # Load and process audio
                with st.spinner("🔄 Processing audio..."):
                    audio_data, sr = get_sound_data(audio_path)
                    
                    # Extract features
                    features = extract_features_from_audio(audio_data, sr)
                    
                    if len(features) == 0:
                        st.error("Audio file too short to process. Please upload a longer file.")
                        return
                    
                    # Make prediction
                    predicted_emotion, confidence, emotion_probs = predict_emotion(model, device, features)
                
                # Clean up temporary file
                if not isinstance(uploaded_file, str) and os.path.exists(audio_path):
                    os.unlink(audio_path)
                
                # Display results
                st.markdown("## 🎯 Prediction Results")
                
                col3, col4, col5 = st.columns(3)
                
                with col3:
                    st.metric(
                        label="🎭 Predicted Emotion",
                        value=predicted_emotion.title(),
                        delta=f"{confidence:.1%} confidence"
                    )
                
                with col4:
                    st.metric(
                        label="⏱️ Audio Duration",
                        value=f"{len(audio_data)/sr:.2f}s"
                    )
                
                with col5:
                    st.metric(
                        label="🔢 Windows Analyzed",
                        value=len(features)
                    )
                
                # Emotion probability chart
                st.plotly_chart(plot_emotion_probabilities(emotion_probs), use_container_width=True)
                
                # Audio visualizations
                st.markdown("## 📊 Audio Analysis")
                
                # Waveform
                st.plotly_chart(plot_waveform(audio_data, sr), use_container_width=True)
                
                # MFCC Spectrogram (show first window)
                if len(features) > 0:
                    mfcc_data = features[0][:, :, 0]  # First window, MFCC channel
                    st.plotly_chart(plot_mfcc_spectrogram(mfcc_data), use_container_width=True)
                
                # Detailed results
                st.markdown("## 📈 Detailed Analysis")
                
                # Create detailed results table
                results_df = pd.DataFrame([
                    emotion_probs
                ]).round(4)
                
                st.dataframe(results_df, use_container_width=True)
                
                # Technical details
                with st.expander("🔧 Technical Details"):
                    st.markdown(f"""
                    **Audio Properties:**
                    - Sample Rate: {sr} Hz
                    - Duration: {len(audio_data)/sr:.2f} seconds
                    - Samples: {len(audio_data):,}
                    
                    **Feature Extraction:**
                    - Windows: {len(features)}
                    - MFCC Coefficients: 128
                    - Frames per Window: 128
                    - Features: MFCC + Delta + Delta-Delta (3 channels)
                    
                    **Model Architecture:**
                    - Enhanced CNN with Attention
                    - Input: 128×128×3
                    - Parameters: ~2.3M
                    - Device: {device}
                    """)
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
    
    with col2:
        st.markdown("## 💡 Tips")
        st.info("""
        **For best results:**
        - Use clear, high-quality audio
        - Audio should be at least 2-3 seconds long
        - Single speaker preferred
        - Minimize background noise
        """)
        
        st.markdown("## 🚀 Model Performance")
        performance_data = {
            "Metric": ["Accuracy", "F1 Score", "Precision", "Recall"],
            "Value": ["95.14%", "95.10%", "95.20%", "95.00%"]
        }
        st.table(pd.DataFrame(performance_data))

if __name__ == "__main__":
    main() 