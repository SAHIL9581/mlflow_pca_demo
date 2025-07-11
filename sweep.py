# sweep.py
import itertools, os, mlflow, numpy as np, pandas as pd
from pipelines import make_cleaning_pipeline, make_dimred_pipeline
from sklearn.pipeline import Pipeline

DATA_PATH = "data/iris.csv"
EXPERIMENT = "pca-two-pipelines"

grid = {
    "n_components":  [2, 3, 4],
    "contamination": [0.01, 0.05]
}

def load_data():
    return pd.read_csv(DATA_PATH).values

def run_once(n_components, contamination):
    X = load_data()

    with mlflow.start_run(nested=True):
        print(f"Running sweep with n_components={n_components}, contamination={contamination}")
        mlflow.log_params({"n_components": n_components,
                           "contamination": contamination})

        clean = make_cleaning_pipeline(contamination=contamination)
        dim   = make_dimred_pipeline(n_components=n_components)
        pipe  = Pipeline([("clean", clean), ("dim", dim)])

        X_emb = pipe.fit_transform(X)

        cum_var = pipe.named_steps["dim"].named_steps["pca"]\
                       .explained_variance_ratio_.cumsum()[-1]
        mlflow.log_metric("cum_expl_var", cum_var)

        mlflow.sklearn.log_model(pipe, "model")

def sweep():
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment(EXPERIMENT)

    combos = itertools.product(grid["n_components"],
                               grid["contamination"])
    for n, c in combos:
        run_once(n, c)
    
    print("\nSweep completed. All runs logged to MLflow.")

if __name__ == "__main__":
    sweep()