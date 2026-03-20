"""
Dataset Augmentation Script
===========================
Enhances the normalized dataset using insights from Unsupervised Learning:
1. Drops rows identified as noise by DBSCAN (cluster == -1)
2. Appends K-Means cluster IDs (k=8) as one-hot encoded features
3. Appends the top 2 Linear Discriminants (LD1, LD2) from LDA

Outputs the enriched dataset for supervised modeling.
"""

import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ===========================
# Configuration
# ===========================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORMALIZED_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'analysis_data', 'normalized_data.csv')
AUGMENTED_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'analysis_data', 'augmented_data.csv')

META_COLS = ['system:index', 'SNo', 'target_class', 'year', '.geo', 'Level_1', 'Level_2']


def run_augmentation():
    print("=" * 70)
    print("DATASET AUGMENTATION")
    print("=" * 70)
    
    print(f"Loading normalized data...")
    df = pd.read_csv(NORMALIZED_CSV)
    initial_len = len(df)
    
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].values
    
    # ---------------------------------------------------------
    # 1. K-Means
    # ---------------------------------------------------------
    print("\n[1/3] Applying K-Means (k=8)...")
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    # One-hot encode clusters
    cluster_dummies = pd.get_dummies(clusters, prefix='KMeans_Cluster')
    
    # Use boolean arrays to ensure True/False instead of 1/0 for newer pandas
    df = pd.concat([df, cluster_dummies.astype(int)], axis=1)
    
    print(f"  Added {cluster_dummies.shape[1]} one-hot encoded cluster features.")

    # ---------------------------------------------------------
    # 2. LDA (Linear Discriminant Analysis)
    # ---------------------------------------------------------
    print("\n[2/3] Extracting top 2 LDA components...")
    lda = LinearDiscriminantAnalysis(n_components=2)
    # LDA uses Level_1 as target to maximize broad category separation
    X_lda = lda.fit_transform(X, df['Level_1'])
    
    df['LD1'] = X_lda[:, 0]
    df['LD2'] = X_lda[:, 1]
    
    var_explained = lda.explained_variance_ratio_
    print(f"  Added LD1 ({var_explained[0]*100:.1f}% variance) and LD2 ({var_explained[1]*100:.1f}% variance).")

    # ---------------------------------------------------------
    # 3. DBSCAN (Noise Removal)
    # ---------------------------------------------------------
    print("\n[3/3] Running DBSCAN for noise removal...")
    # DBSCAN must be run on the exact same PCA space constructed previously (95% variance)
    pca = PCA()
    pca.fit(X)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_95 = np.argmax(cum_var >= 0.95) + 1
    
    pca_95 = PCA(n_components=n_95, random_state=42)
    X_reduced = pca_95.fit_transform(X)
    
    # Calculate eps heuristically (95th percentile distance to 2*n_95 neighbor)
    min_samples = 2 * n_95
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X_reduced)
    distances, _ = neighbors_fit.kneighbors(X_reduced)
    eps_estimate = np.percentile(np.sort(distances[:, min_samples - 1], axis=0), 95)
    
    dbscan = DBSCAN(eps=eps_estimate, min_samples=min_samples, n_jobs=-1)
    dbscan_clusters = dbscan.fit_predict(X_reduced)
    
    # Identify and drop noise points (-1)
    noise_mask = (dbscan_clusters == -1)
    n_noise = noise_mask.sum()
    
    df_clean = df[~noise_mask].reset_index(drop=True)
    
    print(f"  Identified {n_noise} noise points ({n_noise/initial_len*100:.1f}% of data).")
    print(f"  Dropped noise rows.")

    # ---------------------------------------------------------
    # Finalize and Save
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original shape:  {initial_len} rows x {df.shape[1] - 8 - 2} columns (excluding metadata)")
    print(f"Augmented shape: {len(df_clean)} rows x {df_clean.shape[1]} columns")
    print(f"New features added: KMeans_Cluster_0 to KMeans_Cluster_7, LD1, LD2")
    
    df_clean.to_csv(AUGMENTED_CSV, index=False)
    print(f"\nSaved augmented dataset to: {AUGMENTED_CSV}")


if __name__ == '__main__':
    run_augmentation()
