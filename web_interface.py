#!/usr/bin/env python3
"""
Web interface for image upload and prediction
"""

import streamlit as st
import requests
import base64
import io
from PIL import Image
import json


def encode_image_to_base64(image_bytes):
    """Convert image bytes to base64 string"""
    return base64.b64encode(image_bytes).decode("utf-8")


def main():
    st.set_page_config(
        page_title="Anomaly Detection Demo", page_icon="🔍", layout="wide"
    )

    st.title("🔍 Anomaly Detection API Demo")
    st.markdown("Upload images to test the anomaly detection model")

    # API URL configuration
    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000/predict",
        help="Make sure your FastAPI server is running",
    )

    # File uploader
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload one or more images for batch prediction",
    )

    if uploaded_files:
        st.subheader(f"📸 Uploaded {len(uploaded_files)} image(s)")

        # Display uploaded images
        cols = st.columns(min(len(uploaded_files), 4))
        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i % 4]:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_column_width=True)

        # Predict button
        if st.button("🚀 Run Prediction", type="primary"):
            with st.spinner("Processing images..."):
                try:
                    # Encode images to base64
                    base64_images = []
                    for uploaded_file in uploaded_files:
                        image_bytes = uploaded_file.getvalue()
                        b64_string = encode_image_to_base64(image_bytes)
                        base64_images.append(b64_string)

                    # Make API request
                    payload = {"images": base64_images}
                    response = requests.post(api_url, json=payload, timeout=30)
                    response.raise_for_status()

                    # Parse results
                    results = response.json()
                    predictions = results.get("predictions", [])

                    st.success(
                        f"✅ Prediction completed for {len(predictions)} images!"
                    )

                    # Display results
                    st.subheader("📊 Prediction Results")

                    for i, (uploaded_file, prediction) in enumerate(
                        zip(uploaded_files, predictions)
                    ):
                        with st.expander(
                            f"📋 Results for {uploaded_file.name}", expanded=True
                        ):
                            col1, col2 = st.columns([1, 2])

                            with col1:
                                image = Image.open(uploaded_file)
                                st.image(image, use_column_width=True)

                            with col2:
                                # Parse prediction (assuming transformers format)
                                if isinstance(prediction, list) and len(prediction) > 0:
                                    top_pred = prediction[0]
                                    label = top_pred.get("label", "Unknown")
                                    score = top_pred.get("score", 0.0)

                                    # Color coding
                                    if label.upper() == "SURFING":
                                        st.success(
                                            f"🏄‍♂️ **Prediction: {label.upper()}**"
                                        )
                                        st.success(f"📈 **Confidence: {score:.1%}**")
                                    else:
                                        st.error(f"⚠️ **Prediction: {label.upper()}**")
                                        st.error(f"📈 **Confidence: {score:.1%}**")

                                    # Show all predictions
                                    st.markdown("**All Predictions:**")
                                    for pred in prediction:
                                        st.write(
                                            f"- {pred['label']}: {pred['score']:.1%}"
                                        )
                                else:
                                    st.write("Raw prediction:", prediction)

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ Could not connect to API server. Make sure it's running!"
                    )
                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Try with fewer/smaller images.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Instructions
    with st.sidebar:
        st.markdown("## 🛠️ Setup Instructions")
        st.markdown(
            """
        1. **Start the local API server:**
        ```bash
        python api_local.py
        ```
        
        2. **Upload images** using the file uploader
        
        3. **Click 'Run Prediction'** to get results
        
        4. **View results** with confidence scores
        
        > **Note:** Only `api_local.py` is available (Azure Databricks deleted)
        """
        )

        st.markdown("## 📝 Notes")
        st.markdown(
            """
        - Supports JPG, JPEG, PNG formats
        - Multiple images for batch processing
        - Real-time predictions via FastAPI
        - Color-coded results (Green=Normal, Red=Anomaly)
        """
        )


if __name__ == "__main__":
    main()
