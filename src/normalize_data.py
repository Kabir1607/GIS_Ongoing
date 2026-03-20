"""
Data Normalization Script
=========================
Applies StandardScaler (z-score normalization) to all numeric feature columns
in the binned extracted dataset. 

- Fits scaler on training data only (80/20 split) to prevent data leakage
- Saves: normalized full dataset, scaler object (pickle), and normalization stats
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ===========================
# Configuration
# ===========================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'analysis_data', 'binned_extracted_data.csv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'analysis_data')
NORMALIZED_CSV = os.path.join(OUTPUT_DIR, 'normalized_data.csv')
SCALER_PKL = os.path.join(OUTPUT_DIR, 'scaler.pkl')
STATS_TXT = os.path.join(PROJECT_DIR, 'reports', 'normalization_stats.txt')

# Metadata columns — NOT normalized
META_COLS = ['system:index', 'SNo', 'target_class', 'year', '.geo', 'Level_1', 'Level_2']


def normalize():
    print("=" * 70)
    print("DATA NORMALIZATION")
    print("=" * 70)

    # Load data
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Separate features and metadata
    feat_cols = [c for c in df.columns if c not in META_COLS]
    print(f"Feature columns to normalize: {len(feat_cols)}")
    print(f"Metadata columns preserved:   {len([c for c in META_COLS if c in df.columns])}")

    X = df[feat_cols].values
    meta = df[[c for c in META_COLS if c in df.columns]]

    # Train/test split (80/20) — fit scaler ONLY on train to avoid leakage
    X_train, X_test, idx_train, idx_test = train_test_split(
        X, df.index, test_size=0.2, random_state=42,
        stratify=df['Level_1']  # stratified by Level 1 class
    )

    print(f"\nTrain/test split (stratified by Level_1):")
    print(f"  Train: {len(X_train)} rows")
    print(f"  Test:  {len(X_test)} rows")

    # Fit StandardScaler on training data only
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Transform ALL data using train-fitted scaler
    X_normalized = scaler.transform(X)

    # Build normalized DataFrame
    df_norm = pd.DataFrame(X_normalized, columns=feat_cols, index=df.index)
    df_norm = pd.concat([meta.reset_index(drop=True), df_norm.reset_index(drop=True)], axis=1)

    # Save normalized dataset
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_norm.to_csv(NORMALIZED_CSV, index=False)
    print(f"\nNormalized data saved to: {NORMALIZED_CSV}")
    print(f"Shape: {df_norm.shape}")

    # Save scaler for reuse in model training
    with open(SCALER_PKL, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {SCALER_PKL}")

    # Print and save normalization statistics
    stats_lines = []
    stats_lines.append("NORMALIZATION STATISTICS")
    stats_lines.append("=" * 70)
    stats_lines.append(f"Method: StandardScaler (z-score: (x - mean) / std)")
    stats_lines.append(f"Fitted on: {len(X_train)} training samples (80% stratified split)")
    stats_lines.append(f"Applied to: {len(df)} total samples")
    stats_lines.append("")
    stats_lines.append(f"{'Feature':<14} {'Train Mean':>12} {'Train Std':>12} {'Post-Norm Min':>14} {'Post-Norm Max':>14}")
    stats_lines.append("-" * 70)

    for i, col in enumerate(feat_cols):
        mean = scaler.mean_[i]
        std = scaler.scale_[i]
        post_min = X_normalized[:, i].min()
        post_max = X_normalized[:, i].max()
        stats_lines.append(f"  {col:<12} {mean:>12.4f} {std:>12.4f} {post_min:>14.4f} {post_max:>14.4f}")

    stats_lines.append("")
    stats_lines.append("Post-normalization verification (should be ~0 mean, ~1 std on train):")
    train_normalized = scaler.transform(X_train)
    stats_lines.append(f"  Train mean range: [{train_normalized.mean(axis=0).min():.6f}, {train_normalized.mean(axis=0).max():.6f}]")
    stats_lines.append(f"  Train std range:  [{train_normalized.std(axis=0).min():.6f}, {train_normalized.std(axis=0).max():.6f}]")

    stats_text = "\n".join(stats_lines)
    print(f"\n{stats_text}")

    os.makedirs(os.path.dirname(STATS_TXT), exist_ok=True)
    with open(STATS_TXT, 'w') as f:
        f.write(stats_text)
    print(f"\nStats saved to: {STATS_TXT}")


if __name__ == '__main__':
    normalize()
