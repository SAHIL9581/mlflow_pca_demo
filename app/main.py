# app/main.py (THE ABSOLUTELY FINAL VERSION)
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# --- CHANGE 1: Import mlflow.sklearn instead of mlflow.pyfunc ---
import mlflow.sklearn
import numpy as np

# --- Configuration ---
MLFLOW_URI = "file:///C:/Users/Sahil/mlflow_pca_demo/mlruns"
RUN_ID = "0ff5289ea4ea4f21a7bd424f017210ad"
# ---------------------

MODEL_SUBPATH = "model"
app = FastAPI(title="PCA Projection Service")

# Load the pipeline once at startup
pipeline = None
try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"runs:/{RUN_ID}/{MODEL_SUBPATH}"
    # --- CHANGE 1 (continued): Use mlflow.sklearn.load_model ---
    pipeline = mlflow.sklearn.load_model(model_uri)
    print("Model loaded successfully!")
except Exception as e:
    print(f"--- FATAL ERROR LOADING MODEL ---: {e}")
    pipeline = None

class Payload(BaseModel):
    data: list[list[float]]

@app.post("/project")
def project(payload: Payload):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not available. Check server logs.")
    try:
        X = np.asarray(payload.data, dtype=float)
        # --- CHANGE 2: Use .transform() instead of .predict() ---
        embedding = pipeline.transform(X).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def read_root():
    return {"message": f"PCA Service is running. Model Run ID: {RUN_ID}"}