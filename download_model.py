#!/usr/bin/env python3
"""
Download model from Databricks MLflow to local storage
Run this BEFORE deleting Databricks workspace
"""

import os
from dotenv import load_dotenv
import mlflow
from pathlib import Path

# Load environment variables
load_dotenv()

# MLflow configuration
mlflow.set_tracking_uri("databricks")
RUN_ID = "6ffbb8dafe5e4395b1c93d2547ad160a"
MODEL_URI = f"runs:/{RUN_ID}/model"

def download_model():
    """Download model from Databricks to local storage"""
    
    print(f"🔄 Downloading model from: {MODEL_URI}")
    
    try:
        # Create local models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Download model using MLflow
        print("📥 Downloading model artifacts...")
        downloaded_path = mlflow.artifacts.download_artifacts(
            artifact_uri=MODEL_URI,
            dst_path="models/"
        )
        
        print(f"✅ Model downloaded to: {downloaded_path}")
        
        # Test loading the downloaded model
        print("🧪 Testing local model loading...")
        pipe = mlflow.transformers.load_model(downloaded_path)
        print("✅ Local model loads successfully!")
        
        # Save model info
        model_info = {
            "original_run_id": RUN_ID,
            "original_uri": MODEL_URI,
            "local_path": str(downloaded_path),
            "download_date": str(pd.Timestamp.now())
        }
        
        import json
        with open("models/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"📋 Model info saved to: models/model_info.json")
        print("\n🎉 Model successfully downloaded!")
        print("💡 You can now update api.py to use the local model")
        
        return downloaded_path
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("🔍 Make sure:")
        print("  - Databricks workspace is still accessible")
        print("  - .env file has correct credentials")
        print("  - MLflow run ID is valid")
        return None

if __name__ == "__main__":
    import pandas as pd
    download_model()
