# train.py (FINAL CORRECTED VERSION)
import os
import tempfile
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

# We will use the functions from pipelines.py
from pipelines import make_dimred_pipeline

DATA_PATH = "data/iris.csv"
EXPERIMENT = "pca-two-pipelines"

def load_data(path=DATA_PATH):
    return pd.read_csv(path).values

def main():
    X_raw = load_data()

    # Set up MLflow tracking
    absolute_path = os.path.abspath("mlruns").replace('\\', '/')
    tracking_uri = "file:///" + absolute_path
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    params = dict(
        impute_strategy="median",
        contamination=0.05,
        n_components=2
    )

    with mlflow.start_run(run_name="train-full-pipeline"):
        mlflow.log_params(params)
        mlflow.log_artifact(DATA_PATH, artifact_path="input_data")

        # --- CORRECTED LOGIC ---

        # Step 1: Create a pipeline for imputing and scaling data.
        impute_scale_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy=params["impute_strategy"])),
            ("scaler",  StandardScaler())
        ])
        X_scaled = impute_scale_pipe.fit_transform(X_raw)

        # Step 2: Use IsolationForest to detect outliers on the scaled data.
        outlier_detector = IsolationForest(
            contamination=params["contamination"], random_state=42
        )
        # fit_predict returns -1 for outliers and 1 for inliers.
        inlier_mask = outlier_detector.fit_predict(X_scaled) == 1
        
        # Step 3: Remove the outliers to create the clean dataset.
        X_clean = X_scaled[inlier_mask]

        # Step 4: Apply dimensionality reduction on the clean data.
        dimred_pipe = make_dimred_pipeline(n_components=params["n_components"])
        X_emb = dimred_pipe.fit_transform(X_clean)

        # --- Log metrics ---
        expl_var = dimred_pipe.named_steps["pca"].explained_variance_ratio_
        mlflow.log_metric("n_samples_before_cleaning", len(X_raw))
        mlflow.log_metric("n_final_samples", len(X_clean))
        mlflow.log_metric("cum_expl_var", expl_var.cumsum()[-1])

        # --- Combine into ONE re-usable pipeline for INFERENCE ---
        full_inference_pipeline = Pipeline([
            ("impute_scale", impute_scale_pipe),
            ("dimred", dimred_pipe)
        ])
        mlflow.sklearn.log_model(full_inference_pipeline, artifact_path="model")
        
        print("--- SCRIPT COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    main()