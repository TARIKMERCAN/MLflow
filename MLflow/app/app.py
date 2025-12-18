"""
app.py - Streamlit Iris Flower Prediction App

This app loads the best model trained by train_model.py and provides
an interactive interface for Iris flower classification with MLflow metadata.
"""

import streamlit as st
import joblib
import json
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered",
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #6C63FF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #666;
        text-align: center;
        padding: 0.5rem;
        font-size: 0.85rem;
        border-top: 1px solid #ddd;
    }
    .footer a {
        color: #6C63FF;
        text-decoration: none;
    }
    .stSlider > div > div > div {
        background-color: #6C63FF;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model and metadata."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.joblib")
    meta_path = os.path.join(script_dir, "model_meta.json")
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None
    
    model_data = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    
    return model_data, metadata


# Load model and metadata
model_data, metadata = load_model()

# Header
st.markdown('<div class="main-header">🌸 Iris Flower Classifier</div>', unsafe_allow_html=True)

if model_data is None or metadata is None:
    st.error("""
    ⚠️ **Model not found!** 
    
    Please run the training script first:
    ```bash
    python train_model.py
    ```
    """)
    st.stop()

# Extract model and scaler
model = model_data["model"]
scaler = model_data["scaler"]

# Iris flower classes
iris_classes = ["Setosa", "Versicolor", "Virginica"]
iris_images = {
    "Setosa": "🌷",
    "Versicolor": "🌺",
    "Virginica": "🌻"
}

# Description
st.markdown("""
Enter the measurements of an Iris flower below to predict its species.
The model will classify it as one of three species: **Setosa**, **Versicolor**, or **Virginica**.
""")

st.divider()

# Input features
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider(
        "Sepal Length (cm)", 
        min_value=4.0, max_value=8.0, value=5.8, step=0.1
    )
    sepal_width = st.slider(
        "Sepal Width (cm)", 
        min_value=2.0, max_value=4.5, value=3.0, step=0.1
    )

with col2:
    petal_length = st.slider(
        "Petal Length (cm)", 
        min_value=1.0, max_value=7.0, value=4.0, step=0.1
    )
    petal_width = st.slider(
        "Petal Width (cm)", 
        min_value=0.1, max_value=2.5, value=1.3, step=0.1
    )

# Predict button
if st.button("🔮 Predict Species", type="primary", use_container_width=True):
    # Prepare input
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    predicted_class = iris_classes[prediction]
    
    # Get probabilities if available
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_scaled)[0]
    
    # Display result
    st.markdown(f"""
    <div class="prediction-box">
        {iris_images[predicted_class]} Predicted: <strong>{predicted_class}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Show probabilities
    if probabilities is not None:
        st.subheader("Confidence Scores")
        for i, (cls, prob) in enumerate(zip(iris_classes, probabilities)):
            st.progress(prob, text=f"{cls}: {prob:.1%}")

st.divider()

# Model information
with st.expander("ℹ️ Model Information"):
    st.markdown(f"""
    | Property | Value |
    |----------|-------|
    | **Model Type** | {metadata['best_model']} |
    | **Version** | {metadata['version']} |
    | **Accuracy** | {metadata['metrics']['accuracy']:.1%} |
    | **F1-Macro** | {metadata['metrics']['f1_macro']:.4f} |
    | **Precision** | {metadata['metrics']['precision']:.4f} |
    | **Recall** | {metadata['metrics']['recall']:.4f} |
    | **MLflow Run ID** | `{metadata['mlflow_run_id'][:12]}...` |
    """)

# Footer with MLflow metadata
run_id_short = metadata['mlflow_run_id'][:12]
mlflow_url = "http://localhost:5000"

footer_html = f"""
<div class="footer">
    <strong>Version:</strong> {metadata['version']} • 
    <strong>Best model:</strong> {metadata['best_model']} • 
    <strong>MLflow run:</strong> {run_id_short}... • 
    <strong>Accuracy:</strong> {metadata['metrics']['accuracy']:.3f} • 
    <a href="{mlflow_url}" target="_blank">View in MLflow UI ↗</a>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)

# Add padding at the bottom for the fixed footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
