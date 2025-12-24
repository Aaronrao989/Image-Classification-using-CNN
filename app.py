import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import streamlit as st
import tensorflow as tf
from PIL import Image
from inference import predict_image
import warnings
from urllib3.exceptions import NotOpenSSLWarning

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# Page configuration
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Crimson+Pro:wght@300;400;600&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Main container */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 800px;
    }
    
    /* Header styling */
    h1 {
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 2.8rem !important;
        background: linear-gradient(120deg, #00f2fe 0%, #4facfe 50%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem !important;
        letter-spacing: -1px;
        animation: glow 3s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { filter: brightness(1); }
        50% { filter: brightness(1.2); }
    }
    
    /* Subtitle text */
    .subtitle {
        font-family: 'Crimson Pro', serif;
        font-size: 1.1rem;
        color: #b8b8d1;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Upload section */
    .upload-container {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(79, 172, 254, 0.4);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        margin-bottom: 2rem;
    }
    
    .upload-container:hover {
        border-color: rgba(79, 172, 254, 0.7);
        background: rgba(255, 255, 255, 0.05);
        transform: translateY(-2px);
        box-shadow: 0 10px 40px rgba(79, 172, 254, 0.2);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: transparent;
    }
    
    [data-testid="stFileUploader"] label {
        font-family: 'Space Mono', monospace !important;
        color: #4facfe !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stFileUploader"] section {
        border: none !important;
        background: transparent !important;
        padding: 0 !important;
    }
    
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: #0f0c29 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3) !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(79, 172, 254, 0.5) !important;
    }
    
    /* Image display */
    [data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
        margin: 2rem 0;
        animation: fadeIn 0.6s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.95); }
        to { opacity: 1; transform: scale(1); }
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.9rem 3rem !important;
        font-family: 'Space Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        width: 100%;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        margin-top: 1rem;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 242, 254, 0.1) 0%, rgba(79, 172, 254, 0.1) 100%) !important;
        border-left: 4px solid #00f2fe !important;
        border-radius: 12px !important;
        padding: 1.2rem !important;
        font-family: 'Crimson Pro', serif !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #00f2fe !important;
        margin-top: 1.5rem;
        backdrop-filter: blur(10px);
        animation: slideIn 0.5s ease-out;
    }
    
    /* Info message */
    .stInfo {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%) !important;
        border-left: 4px solid #667eea !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 1.1rem !important;
        color: #b8b8d1 !important;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
        animation: slideIn 0.5s ease-out 0.1s both;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Decorative elements */
    .decorator {
        text-align: center;
        font-size: 2rem;
        color: rgba(79, 172, 254, 0.3);
        margin: 1rem 0;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.6; }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("# 🔮 CIFAR-10 Classifier")
st.markdown('<div class="subtitle">Neural vision powered by deep learning</div>', unsafe_allow_html=True)

# Decorative separator
st.markdown('<div class="decorator">⋯</div>', unsafe_allow_html=True)

# Upload section
st.markdown('<div class="upload-container">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("🧠 Analyzing image..."):
            label, confidence = predict_image(image)
        
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{confidence * 100:.2f}%**")