#!/usr/bin/env python3
"""
Local version of API that doesn't depend on Databricks
"""

import torch
import mlflow.transformers
from pathlib import Path
import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- Local Model Configuration ---
LOCAL_MODEL_PATH = "models/anomaly_detection_model"
MODEL_INFO_PATH = "models/model_info.json"

def load_local_model():
    """Load model from local storage"""
    model_path = Path(LOCAL_MODEL_PATH)
    
    if not model_path.exists():
        print(f"❌ Local model not found at: {model_path}")
        print("💡 Run 'python download_model.py' first to download the model")
        return None
    
    try:
        print(f"📂 Loading model from: {model_path}")
        pipe = mlflow.transformers.load_model(str(model_path))
        
        print("🖥️ Forcing model to CPU...")
        device = torch.device("cpu")
        pipe.model.to(device)
        pipe.device = device
        
        print(f"✅ Model successfully loaded on device: {device}")
        
        # Load model info if available
        info_path = Path(MODEL_INFO_PATH)
        if info_path.exists():
            with open(info_path, 'r') as f:
                model_info = json.load(f)
            print(f"📋 Original run ID: {model_info.get('original_run_id', 'Unknown')}")
            print(f"📅 Downloaded: {model_info.get('download_date', 'Unknown')}")
        
        return pipe
        
    except Exception as e:
        print(f"❌ Error loading local model: {e}")
        return None

# Load model at startup
print("🚀 Starting Anomaly Detection API (Local Mode)")
pipe = load_local_model()

if pipe is None:
    print("\n" + "="*50)
    print("❌ FAILED TO LOAD MODEL")
    print("="*50)
    print("To fix this issue:")
    print("1. Make sure you have downloaded the model:")
    print("   python download_model.py")
    print("2. Or check if the model path is correct:")
    print(f"   {LOCAL_MODEL_PATH}")
    print("="*50)
    exit(1)

# Define FastAPI app
app = FastAPI(
    title="Anomaly Detection API (Local)",
    description="Local version - no Databricks dependency"
)

# Add a root endpoint for basic health check
@app.get("/")
async def root():
    return {
        "message": "Anomaly Detection API is running (Local Mode)",
        "endpoints": ["/predict (POST)"],
        "model_source": "Local MLflow model",
        "databricks_required": False
    }

# Define the model input
class ModelInput(BaseModel):
    images: list[str]

# Define the predict endpoint
@app.post("/predict")
async def predict(data: ModelInput):
    """
    Input: a list of base64 strings
    Output: prediction result
    """
    try:
        print(f"📥 Received {len(data.images)} images for prediction.")
        
        predictions = pipe(data.images)
        print("✅ Prediction successful.")
        
        return {"predictions": predictions}
        
    except Exception as e:
        print("\n--- 500 INTERNAL SERVER ERROR ---")
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        print("---------------------------------\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n🌐 Starting server...")
    print("📍 Local model mode - no Databricks required")
    uvicorn.run(app, host="0.0.0.0", port=8000)
