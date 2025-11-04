# Anomaly Detection API

## 🚀 Overview
FastAPI-based anomaly detection system for video frames, deployed as an alternative to Azure Databricks model serving (unavailable in free tier).

## 📊 Prediction Results

### Interactive Demo
🔗 **[View Live Results](https://ark0723.github.io/anomaly_detection/)** (GitHub Pages)


### Jupyter Notebook
📓 **[Interactive Testing Notebook](notebook/07_deployment.ipynb)** - Run this notebook to see live predictions with HTML visualization

![Prediction Results](assets/prediction_results_v2.png)

*Real-time anomaly detection results showing normal frames (green border) and anomalous frames (red border) with confidence scores.*

## 🛠️ Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start API Server
```bash
python api.py
# or
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 3. Test with Client
```bash
python client.py
```

### 4. Interactive Testing
Open `notebook/07_deployment.ipynb` for interactive testing with HTML visualization.

## 📁 Project Structure
```
anomaly_detection/
├── api.py                 # FastAPI server
├── client.py              # API client for testing
├── frames/                # Sample images
├── notebook/              # Jupyter notebooks
│   └── 07_deployment.ipynb # Interactive testing & visualization
└── requirements.txt       # Dependencies
```

## 🔧 Features
- **Base64 Image Processing**: Handles multiple images in batch
- **Real-time Predictions**: Fast inference with CPU optimization
- **Visual Results**: Color-coded prediction display
- **Error Handling**: Comprehensive logging and fallback mechanisms

## 🎯 API Endpoints
- `GET /`: Health check
- `POST /predict`: Batch image prediction

## 📝 Model Details
- **Framework**: Transformers pipeline via MLflow
- **Input**: Base64-encoded images
- **Output**: Classification scores (normal/anomaly)
- **Deployment**: Local FastAPI (Databricks alternative)
