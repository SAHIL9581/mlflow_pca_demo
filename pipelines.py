# pipelines.py
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def make_cleaning_pipeline(impute_strategy="median", contamination=0.05):
    """Returns a pipeline that cleans raw numeric data."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        ("scaler",  StandardScaler()),
        ("outlier", IsolationForest(random_state=42,
                                    contamination=contamination))
    ])

def make_dimred_pipeline(n_components=2):
    """Returns a PCA pipeline. (Could later swap PCA for UMAP, etc.)"""
    return Pipeline([
        ("pca", PCA(n_components=n_components,
                    svd_solver="full",
                    random_state=42))
    ])