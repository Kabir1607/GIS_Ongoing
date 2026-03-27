"""
Phase 0b: Unsupervised Learning Augmentation for Dataset 3
==========================================================
Applies DBSCAN noise removal, K-Means clustering, PCA, and LDA
to generate augmented feature sets for A/B testing.

Usage:
    python src/dataset_3_src/unsupervised_augmentation.py
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import joblib
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'dataset_3', 'processed')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models', 'dataset_3')

N_CLUSTERS = 10
N_PCA_COMPONENTS = 5
DBSCAN_EPS = 3.0
DBSCAN_MIN_SAMPLES = 5
RANDOM_STATE = 42


def load_processed_data(level='level_1'):
    """Loads the prepared train/test splits."""
    suffix = level
    
    X_train_baseline = pd.read_csv(os.path.join(DATA_DIR, f'X_train_baseline_{suffix}.csv'))
    X_test_baseline = pd.read_csv(os.path.join(DATA_DIR, f'X_test_baseline_{suffix}.csv'))
    X_train_full = pd.read_csv(os.path.join(DATA_DIR, f'X_train_full_{suffix}.csv'))
    X_test_full = pd.read_csv(os.path.join(DATA_DIR, f'X_test_full_{suffix}.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, f'y_train_{suffix}.csv'))['target']
    y_test = pd.read_csv(os.path.join(DATA_DIR, f'y_test_{suffix}.csv'))['target']
    
    with open(os.path.join(DATA_DIR, f'feature_meta_{suffix}.json'), 'r') as f:
        feature_meta = json.load(f)
    
    return {
        'X_train_baseline': X_train_baseline,
        'X_test_baseline': X_test_baseline,
        'X_train_full': X_train_full,
        'X_test_full': X_test_full,
        'y_train': y_train,
        'y_test': y_test,
        'feature_meta': feature_meta,
    }


def apply_dbscan_noise_removal(X_train, y_train, X_test):
    """
    Uses DBSCAN to identify and remove outlier points from training data.
    Test data is not modified.
    Returns cleaned train data, original test data, and the noise mask.
    """
    print("\n  [DBSCAN] Fitting on training data...")
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
    labels = db.fit_predict(X_train)
    
    noise_mask = labels == -1
    noise_count = noise_mask.sum()
    print(f"  [DBSCAN] Found {noise_count} noise points ({noise_count/len(X_train)*100:.1f}%)")
    
    X_train_clean = X_train[~noise_mask].reset_index(drop=True)
    y_train_clean = y_train[~noise_mask].reset_index(drop=True)
    
    return X_train_clean, y_train_clean, noise_mask


def apply_kmeans_features(X_train, X_test, n_clusters=N_CLUSTERS):
    """
    Fits K-Means on training data and appends cluster ID as one-hot features
    to both train and test.
    """
    print(f"\n  [K-Means] Fitting with k={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    train_clusters = km.fit_predict(X_train)
    test_clusters = km.predict(X_test)
    
    # One-hot encode cluster IDs
    for i in range(n_clusters):
        X_train[f'kmeans_cluster_{i}'] = (train_clusters == i).astype(int)
        X_test[f'kmeans_cluster_{i}'] = (test_clusters == i).astype(int)
    
    print(f"  [K-Means] Added {n_clusters} one-hot cluster features")
    return X_train, X_test, km


def apply_pca_features(X_train, X_test, n_components=N_PCA_COMPONENTS):
    """Appends PCA components as additional features."""
    print(f"\n  [PCA] Fitting with {n_components} components...")
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    
    # Fit on original feature columns only (not on previously added cluster features)
    original_cols = [c for c in X_train.columns if not c.startswith('kmeans_')]
    
    train_pca = pca.fit_transform(X_train[original_cols])
    test_pca = pca.transform(X_test[original_cols])
    
    for i in range(n_components):
        X_train[f'pca_{i}'] = train_pca[:, i]
        X_test[f'pca_{i}'] = test_pca[:, i]
    
    explained_var = pca.explained_variance_ratio_.sum()
    print(f"  [PCA] Total explained variance: {explained_var:.3f}")
    return X_train, X_test, pca


def apply_lda_features(X_train, X_test, y_train):
    """Appends LDA components as additional features."""
    n_classes = len(np.unique(y_train))
    n_lda = min(n_classes - 1, X_train.shape[1])
    
    print(f"\n  [LDA] Fitting with {n_lda} components...")
    
    # Use only original + cluster features (not PCA to avoid redundancy)
    original_cols = [c for c in X_train.columns if not c.startswith('pca_') and not c.startswith('lda_')]
    
    lda = LDA(n_components=n_lda)
    train_lda = lda.fit_transform(X_train[original_cols], y_train)
    test_lda = lda.transform(X_test[original_cols])
    
    for i in range(n_lda):
        X_train[f'lda_{i}'] = train_lda[:, i]
        X_test[f'lda_{i}'] = test_lda[:, i]
    
    print(f"  [LDA] Added {n_lda} discriminant components")
    return X_train, X_test, lda


def augment_and_save(level='level_1'):
    """
    Full augmentation pipeline for a single classification level.
    Produces 4 dataset variants:
      - baseline (no noise reduction)
      - baseline_augmented (with noise reduction + unsupervised features)
      - full (with AEF, no noise reduction)
      - full_augmented (with AEF + noise reduction + unsupervised features)
    """
    print(f"\n{'=' * 60}")
    print(f"UNSUPERVISED AUGMENTATION — {level.upper()}")
    print(f"{'=' * 60}")
    
    data = load_processed_data(level)
    
    # ── VARIANT 1 & 2: Baseline features ──
    print(f"\n--- Processing BASELINE features ---")
    
    X_train_b = data['X_train_baseline'].copy()
    X_test_b = data['X_test_baseline'].copy()
    y_train = data['y_train'].copy()
    y_test = data['y_test'].copy()
    
    # Variant 1: baseline (already saved by prepare_data.py, skip)
    
    # Variant 2: baseline_augmented
    X_train_b_clean, y_train_clean, noise_mask = apply_dbscan_noise_removal(
        X_train_b.copy(), y_train.copy(), X_test_b.copy()
    )
    X_test_b_aug = X_test_b.copy()
    
    X_train_b_clean, X_test_b_aug, km_b = apply_kmeans_features(X_train_b_clean, X_test_b_aug)
    X_train_b_clean, X_test_b_aug, pca_b = apply_pca_features(X_train_b_clean, X_test_b_aug)
    X_train_b_clean, X_test_b_aug, lda_b = apply_lda_features(X_train_b_clean, X_test_b_aug, y_train_clean)
    
    X_train_b_clean.to_csv(os.path.join(DATA_DIR, f'X_train_baseline_augmented_{level}.csv'), index=False)
    X_test_b_aug.to_csv(os.path.join(DATA_DIR, f'X_test_baseline_augmented_{level}.csv'), index=False)
    y_train_clean.to_csv(os.path.join(DATA_DIR, f'y_train_baseline_augmented_{level}.csv'), index=False, header=['target'])
    
    # ── VARIANT 3 & 4: Full features (with AEF) ──
    print(f"\n--- Processing FULL features (with AEF) ---")
    
    X_train_f = data['X_train_full'].copy()
    X_test_f = data['X_test_full'].copy()
    y_train = data['y_train'].copy()
    
    # Variant 3: full (already saved by prepare_data.py, skip)
    
    # Variant 4: full_augmented
    X_train_f_clean, y_train_f_clean, noise_mask_f = apply_dbscan_noise_removal(
        X_train_f.copy(), y_train.copy(), X_test_f.copy()
    )
    X_test_f_aug = X_test_f.copy()
    
    X_train_f_clean, X_test_f_aug, km_f = apply_kmeans_features(X_train_f_clean, X_test_f_aug)
    X_train_f_clean, X_test_f_aug, pca_f = apply_pca_features(X_train_f_clean, X_test_f_aug)
    X_train_f_clean, X_test_f_aug, lda_f = apply_lda_features(X_train_f_clean, X_test_f_aug, y_train_f_clean)
    
    X_train_f_clean.to_csv(os.path.join(DATA_DIR, f'X_train_full_augmented_{level}.csv'), index=False)
    X_test_f_aug.to_csv(os.path.join(DATA_DIR, f'X_test_full_augmented_{level}.csv'), index=False)
    y_train_f_clean.to_csv(os.path.join(DATA_DIR, f'y_train_full_augmented_{level}.csv'), index=False, header=['target'])
    
    # Save fitted transformers
    joblib.dump(km_b, os.path.join(MODEL_DIR, f'kmeans_baseline_{level}.pkl'))
    joblib.dump(pca_b, os.path.join(MODEL_DIR, f'pca_baseline_{level}.pkl'))
    joblib.dump(lda_b, os.path.join(MODEL_DIR, f'lda_baseline_{level}.pkl'))
    joblib.dump(km_f, os.path.join(MODEL_DIR, f'kmeans_full_{level}.pkl'))
    joblib.dump(pca_f, os.path.join(MODEL_DIR, f'pca_full_{level}.pkl'))
    joblib.dump(lda_f, os.path.join(MODEL_DIR, f'lda_full_{level}.pkl'))
    
    print(f"\n  Augmented datasets saved to: {DATA_DIR}")
    print(f"  Fitted transformers saved to: {MODEL_DIR}")


def main():
    print("=" * 60)
    print("PHASE 0b: UNSUPERVISED AUGMENTATION")
    print("=" * 60)
    
    augment_and_save('level_1')
    augment_and_save('level_2')
    
    print("\n" + "=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)
    print("\nReady for Phase 1. Run:")
    print("  python src/dataset_3_src/phase1_xgboost_ab.py")


if __name__ == '__main__':
    main()
