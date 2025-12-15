import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def build_customer_segmentation(n_clusters=5, random_state=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto"))
    ])

def fit_predict(model, df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    labels = model.fit_predict(df[feature_cols])
    return pd.Series(labels, index=df.index, name="segment")
