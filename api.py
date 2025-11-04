from dotenv import load_dotenv

import torch
import mlflow.transformers
from transformers import pipeline

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# load environment variables from .env file
load_dotenv()

# MLflow가 자동으로 .env 파일의 변수를 읽어 인증합니다.
mlflow.set_tracking_uri("databricks")

# This is the MLflow Run ID from your notebook
RUN_ID = "6ffbb8dafe5e4395b1c93d2547ad160a"
MODEL_URI = f"runs:/{RUN_ID}/model"

print(f"Loading model from: {MODEL_URI}")
# pyfunc_model = mlflow.pyfunc.load_model(MODEL_URI)
try:
    pipe = mlflow.transformers.load_model(MODEL_URI)
    print("Pipeline loaded. Forcing to CPU...")
    device = torch.device("cpu")
    pipe.model.to(device)
    pipe.device = device
    print(f"Model successfully forced to device: {device}")

except Exception as e:
    print(f"Error loading model or forcing to CPU: {e}")
    exit()


# define the fastapi app
app = FastAPI(title="Anomaly Detection API")


# Add a root endpoint for basic health check
@app.get("/")
async def root():
    return {
        "message": "Anomaly Detection API is running",
        "endpoints": ["/predict (POST)"],
    }


# define the model input : a list of base64 strings
class ModelInput(BaseModel):
    images: list[str]


# define the predict endpoint
@app.post("/predict")
async def predict(data: ModelInput):
    """
    Input: a list of base64 strings
    Output: prediction result
    """
    try:
        # Transformers pipeline expects a list of base64 strings directly
        # run inference
        print(
            f"Received {len(data.images)} images for prediction."
        )  # <-- 1. 요청 수신 확인 로그

        predictions = pipe(data.images)
        print("Prediction successful.")  # <-- 2. 성공 로그
        return {"predictions": predictions}
    except Exception as e:
        print("\n--- 500 INTERNAL SERVER ERROR ---")
        print(f"Error during prediction: {e}")
        import traceback

        traceback.print_exc()  # <--- 이것이 전체 오류 내용을 출력합니다.
        print("---------------------------------\n")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
