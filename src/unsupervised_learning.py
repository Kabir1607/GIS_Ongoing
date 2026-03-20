"""
Unsupervised Learning & Dimensionality Reduction
================================================
Applies PCA, K-Means clustering, DBSCAN (Density-based), and 
LDA (Linear Discriminant Analysis) on the normalized dataset.
Saves visualizations of the clusters and variance explained.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ===========================
# Configuration
# ===========================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORMALIZED_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'analysis_data', 'normalized_data.csv')
VISUALS_DIR = os.path.join(PROJECT_DIR, 'data_visuals', 'unsupervised')
REPORT_FILE = os.path.join(PROJECT_DIR, 'reports', 'unsupervised_results.txt')

META_COLS = ['system:index', 'SNo', 'target_class', 'year', '.geo', 'Level_1', 'Level_2']


def section_print(report, msg):
    print(msg)
    report.append(msg)


def run_unsupervised():
    os.makedirs(VISUALS_DIR, exist_ok=True)
    
    print("Loading normalized data...")
    df = pd.read_csv(NORMALIZED_CSV)
    
    feat_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feat_cols].values
    
    report_lines = []
    section_print(report_lines, "UNSUPERVISED LEARNING & DIMENSIONALITY REDUCTION REPORT")
    section_print(report_lines, "=" * 70)
    section_print(report_lines, f"Data: {len(df)} samples, {len(feat_cols)} normalized features\n")
    
    # ====================================================================
    # 1. PCA Analysis
    # ====================================================================
    print("\n--- 1. Running PCA ---")
    pca = PCA()
    pca.fit(X)
    
    var_explained = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_explained)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cum_var) + 1), cum_var, 'b-', marker='.', markersize=4)
    plt.axhline(y=0.90, color='orange', linestyle='--', label='90% Variance')
    plt.axhline(y=0.95, color='green', linestyle='--', label='95% Variance')
    plt.grid(True, alpha=0.3)
    plt.title('PCA: Cumulative Explained Variance', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'pca_variance_explained.png'), dpi=150)
    plt.close()
    
    n_90 = np.argmax(cum_var >= 0.90) + 1
    n_95 = np.argmax(cum_var >= 0.95) + 1
    
    section_print(report_lines, "--- PCA Summary ---")
    section_print(report_lines, f"Components for 90% variance: {n_90} (out of {len(feat_cols)})")
    section_print(report_lines, f"Components for 95% variance: {n_95}")
    section_print(report_lines, f"Top 5 components explain:    {cum_var[4]*100:.1f}%\n")
    
    # 2D PCA Projection
    pca_2d = PCA(n_components=2)
    X_pca2 = pca_2d.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_pca2[:, 0], y=X_pca2[:, 1], hue=df['Level_1'], 
                    palette='Set1', alpha=0.6, s=15, edgecolor=None)
    plt.title('2D PCA Projection (Colored by Level 1 Class)', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'pca_2d_projection.png'), dpi=150)
    plt.close()
    
    # ====================================================================
    # 2. K-Means Clustering
    # ====================================================================
    print("\n--- 2. Running K-Means ---")
    k = 8  # Matching the 8 granular Level_2 valid classes
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_clusters = kmeans.fit_predict(X)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=kmeans_clusters, cmap='tab10', alpha=0.6, s=15)
    plt.title(f'K-Means Clustering (K={k}) on Top 2 Principal Components', fontsize=14, fontweight='bold')
    plt.xlabel(f'PC1')
    plt.ylabel(f'PC2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, f'kmeans_clusters_k{k}.png'), dpi=150)
    plt.close()
    
    df['KMeans_Cluster'] = kmeans_clusters
    section_print(report_lines, f"--- K-Means Clustering (K={k}) ---")
    section_print(report_lines, "Cluster composition by Level 1 class:")
    cross_tab_km = pd.crosstab(df['KMeans_Cluster'], df['Level_1'])
    section_print(report_lines, cross_tab_km.to_string() + "\n")
    
    # ====================================================================
    # 3. DBSCAN (Density-Based Clustering)
    # ====================================================================
    print("\n--- 3. Running DBSCAN ---")
    # To avoid the curse of dimensionality and memory explosions, run DBSCAN 
    # on the PCA components that explain 95% of the variance
    pca_95 = PCA(n_components=n_95, random_state=42)
    X_reduced = pca_95.fit_transform(X)
    
    # Heuristic: k-distance graph to find 'eps'
    # We use min_samples = 2 * dimensions (standard rule of thumb)
    min_samples = 2 * n_95
    print(f"Calculating k-distance graph for min_samples={min_samples}...")
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X_reduced)
    distances, indices = neighbors_fit.kneighbors(X_reduced)
    distances = np.sort(distances[:, min_samples - 1], axis=0) # Sort distances to the k-th nearest neighbor
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f'K-Distance Graph (k={min_samples}) for DBSCAN eps', fontsize=14, fontweight='bold')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel(f'{min_samples}-th Nearest Neighbor Distance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'dbscan_k_distance_graph.png'), dpi=150)
    plt.close()
    
    # The 'elbow' usually occurs at the point of maximum curvature.
    # We estimate 'eps' programmatically by finding where the slope increases sharply
    # (using 95th percentile as a naive fallback if the knee isn't obvious)
    eps_estimate = np.percentile(distances, 95)
    print(f"Estimated eps value (95th percentile distance): {eps_estimate:.2f}")
    
    dbscan = DBSCAN(eps=eps_estimate, min_samples=min_samples, n_jobs=-1)
    dbscan_clusters = dbscan.fit_predict(X_reduced)
    
    n_clusters_db = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
    n_noise = list(dbscan_clusters).count(-1)
    pct_noise = n_noise / len(X_reduced) * 100
    
    df['DBSCAN_Cluster'] = dbscan_clusters
    
    section_print(report_lines, "--- DBSCAN (Density-Based Clustering) ---")
    section_print(report_lines, f"Input space: {n_95} Principal Components (95% variance)")
    section_print(report_lines, f"Hyperparameters: eps={eps_estimate:.2f}, min_samples={min_samples}")
    section_print(report_lines, f"Clusters found: {n_clusters_db}")
    section_print(report_lines, f"Noise points: {n_noise} ({pct_noise:.1f}%)\n")
    
    # Plot DBSCAN clusters on 2D PCA
    plt.figure(figsize=(12, 8))
    # Plot noise points in grey/black
    noise_mask = (dbscan_clusters == -1)
    sns.scatterplot(x=X_pca2[noise_mask, 0], y=X_pca2[noise_mask, 1], 
                    color='gray', alpha=0.3, s=10, label='Noise / Outliers', edgecolor=None)
    # Plot clustered points
    sns.scatterplot(x=X_pca2[~noise_mask, 0], y=X_pca2[~noise_mask, 1], 
                    hue=dbscan_clusters[~noise_mask], palette='tab20', 
                    alpha=0.7, s=20, legend='full', edgecolor=None)
    plt.title(f'DBSCAN Clusters mapped on Top 2 PCs (Noise points in gray)', fontsize=14)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'dbscan_clusters.png'), dpi=150)
    plt.close()

    # ====================================================================
    # 4. LDA (Linear Discriminant Analysis)
    # ====================================================================
    print("\n--- 4. Running LDA ---")
    # LDA is supervised dimensionality reduction: it finds linear combinations 
    # of features that maximize separation BETWEEN classes.
    # Target: Level 1 (5 broad categories -> max 4 LDA components)
    
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X, df['Level_1'])
    
    lda_var = lda.explained_variance_ratio_
    
    # Variance Explained Bar Chart
    plt.figure(figsize=(8, 5))
    plt.bar([f"LD{i+1}" for i in range(len(lda_var))], lda_var * 100, color='mediumpurple')
    for i, v in enumerate(lda_var):
        plt.text(i, v * 100 + 1, f"{v*100:.1f}%", ha='center', fontweight='bold')
    plt.title('LDA: Variance Explained by Linear Discriminants', fontsize=13, fontweight='bold')
    plt.ylabel('Percentage of Variance (%)')
    plt.ylim(0, max(lda_var) * 100 + 15)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'lda_variance_explained.png'), dpi=150)
    plt.close()
    
    # 2D LDA Projection (LD1 vs LD2)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_lda[:, 0], y=X_lda[:, 1], hue=df['Level_1'], 
                    palette='Set1', alpha=0.6, s=15, edgecolor=None)
    plt.title('2D LDA Projection (Maximized Class Separability)', fontsize=14, fontweight='bold')
    plt.xlabel(f'Linear Discriminant 1 ({lda_var[0]*100:.1f}%)')
    plt.ylabel(f'Linear Discriminant 2 ({lda_var[1]*100:.1f}%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'lda_2d_projection.png'), dpi=150)
    plt.close()
    
    section_print(report_lines, "--- LDA (Linear Discriminant Analysis) ---")
    section_print(report_lines, f"Target classes: {df['Level_1'].nunique()} (Level 1)")
    section_print(report_lines, f"Linear discriminants produced: {len(lda_var)}")
    section_print(report_lines, f"Variance captured by LD1: {lda_var[0]*100:.1f}%")
    section_print(report_lines, f"Variance captured by LD2: {lda_var[1]*100:.1f}%")
    section_print(report_lines, "Summary: LDA projection shows how highly separable the broad Level 1 classes are when the optimization goal is class separation (unlike PCA, which is blind to labels).\n")
    
    # ====================================================================
    # Save Report
    # ====================================================================
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"\nSaved Unsupervised Learning Report: {REPORT_FILE}")
    print(f"Visualizations saved to: {VISUALS_DIR}/")

if __name__ == '__main__':
    run_unsupervised()
