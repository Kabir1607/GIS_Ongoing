import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def generate_eda_report(df, output_dir='reports/EDA'):
    """
    Generates Exploratory Data Analysis graphs and metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Class Distribution
    if 'target_class' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='target_class')
        plt.title("Class Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/class_distribution.png")
        plt.close()
        
    print(f"EDA generated at {output_dir}")


def apply_unsupervised_learning(df, feature_cols, output_dir='reports/EDA'):
    """
    Applies PCA and K-Means clustering.
    Normalization is explicitly skipped per user constraints, preserving raw domains for testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract features, ignoring NaNs for unsupervised analysis
    X_raw = df[feature_cols].copy().dropna()
    
    # 1. PCA
    print("Applying PCA (without normalization)...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_raw)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
    plt.title("2D PCA Projection (Unnormalized)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(f"{output_dir}/pca_projection.png")
    plt.close()
    
    # 2. K-Means
    print("Applying K-Means Clustering...")
    kmeans = KMeans(n_clusters=8, random_state=42) # Approximating the 8-9 valid classes
    clusters = kmeans.fit_predict(X_raw)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', alpha=0.5)
    plt.title("K-Means Clusters mapping onto Top 2 PCA Components")
    plt.savefig(f"{output_dir}/kmeans_clusters.png")
    plt.close()
    
    return pca_result, clusters

def run_analysis(csv_path):
    """
    Main function to execute EDA logic once extraction completes.
    """
    if not os.path.exists(csv_path):
        print(f"Waiting for extracted data at {csv_path} to begin EDA.")
        return
        
    df = pd.read_csv(csv_path)
    generate_eda_report(df)
    
    # Filter numeric representation for Unsupervised testing
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    apply_unsupervised_learning(df, feature_cols=numeric_cols)

if __name__ == '__main__':
    # Typically this will run on the exported CSV generated via Earth Engine
    run_analysis('data/extracted_features.csv')
