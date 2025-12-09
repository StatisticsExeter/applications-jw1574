import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from pathlib import Path
from course.utils import find_project_root
from course.unsupervised_classification.tree import _scatter_clusters, _pca

VIGNETTE_DIR = Path('data_cache') / 'vignettes' / 'unsupervised_classification'


def _DBSCAN(df, eps=0.5, min_samples=5):
    """
    Fit DBSCAN clustering to standardized data.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(df)
    return db


def dbscan(eps=0.5, min_samples=5):
    """
    Full DBSCAN workflow:
    - Load data
    - Standardize
    - Fit DBSCAN
    - PCA for visualization
    - Save scatter plot
    """
    base_dir = find_project_root()
    df = pd.read_csv(base_dir / 'data_cache' / 'la_collision.csv')

    # Standardize the features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Fit DBSCAN
    db = _DBSCAN(df_scaled, eps=eps, min_samples=min_samples)
    clusters = db.labels_  # -1 indicates noise points

    # PCA for visualization
    df_plot = _pca(df_scaled)
    df_plot['cluster'] = clusters.astype(str)  # convert to string for Plotly color grouping
    # Ensure the directory exists
    base_dir = find_project_root()

    outpath = base_dir / VIGNETTE_DIR / 'dbscan_scatter.html'
    fig = _scatter_clusters(df_plot)
    fig.write_html(outpath)

    # Print summary
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points.")
    print(f"DBSCAN scatter plot saved to {outpath}")
