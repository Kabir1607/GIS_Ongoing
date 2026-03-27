"""
Phase 0: Data Preparation Pipeline for Dataset 3
=================================================
Merges EE-extracted features with binned labels, normalizes, splits,
and generates both baseline and AEF feature sets.

Usage:
    python src/dataset_3_src/prepare_data.py
"""

import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'dataset_3')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models', 'dataset_3')

BINNED_CSV = os.path.join(DATA_DIR, 'raw_dataset_3_binned.csv')

# Expected feature columns after EE extraction
# Sentinel-2 optical bands + indices
OPTICAL_FEATURES = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
INDEX_FEATURES = ['NDVI', 'NDWI', 'GCVI']
# Landsat 8 bands
LANDSAT_FEATURES = ['L8_blue', 'L8_green', 'L8_red', 'L8_nir', 'L8_swir1', 'L8_swir2']
# AEF embedding dimensions (64)
AEF_PREFIX = 'aef_'  # Will match columns starting with this prefix

BASELINE_FEATURES = OPTICAL_FEATURES + INDEX_FEATURES + LANDSAT_FEATURES
# AEF features will be detected dynamically from column names

CSI_FORMULAS = {
    'CSI_1': ('L8_blue', 'L8_green', 'prod'),     
    'CSI_2': ('L8_blue', 'L8_red', 'prod'),     
    'CSI_3': ('L8_blue', 'L8_nir', 'prod'),  
    'CSI_4': ('L8_green', 'L8_red', 'prod'),    
    'CSI_5': ('L8_green', 'L8_nir', 'prod'),    
}
CSI_FEATURES = list(CSI_FORMULAS.keys())

RANDOM_STATE = 42
TEST_SIZE = 0.20
EPSILON = 1e-8


def load_and_merge_ee_data():
    print("=" * 60)
    print("PHASE 0: DATA PREPARATION")
    print("=" * 60)
    
    csv_path = os.path.join(DATA_DIR, 'dataset_3_combined_GEE.csv')
    if not os.path.exists(csv_path):
        print(f"  WARNING: {csv_path} not found.")
        return None
        
    print(f"\n[1/6] Loading combined dataset from: {csv_path}")
    df_merged = pd.read_csv(csv_path)
    print(f"  Loaded {len(df_merged)} rows with {len(df_merged.columns)} columns")
    
    # Compute CSI features on RAW data before any scaling
    print("\n[2/6] Computing Cross-Sensor & Novel Landsat Indices on RAW data...")
    for csi_name, (band_a, band_b, op) in CSI_FORMULAS.items():
        if band_a in df_merged.columns and band_b in df_merged.columns:
            a, b = df_merged[band_a], df_merged[band_b]
            if op == 'diff':
                df_merged[csi_name] = a - b
            elif op == 'prod':
                df_merged[csi_name] = (a * b) / (a + b + EPSILON)
            print(f"  --> Computed {csi_name} using {band_a} and {band_b} ({op})")
            
    return df_merged


def identify_feature_columns(df):
    """
    Dynamically identifies baseline and AEF feature columns from the merged DF.
    Returns (baseline_cols, aef_cols, all_feature_cols).
    """
    all_cols = df.columns.tolist()
    
    # Baseline features
    baseline_cols = [c for c in BASELINE_FEATURES if c in all_cols] + [c for c in CSI_FEATURES if c in all_cols]
    
    # AEF features (columns starting with 'aef_' or containing embedding patterns)
    aef_cols = [c for c in all_cols if c.lower().startswith(AEF_PREFIX) or 
                (c.startswith('b') and c[1:].isdigit())]  # e.g., b0, b1, ..., b63
    
    # If AEF columns use a different naming convention, detect numeric-only columns
    if len(aef_cols) == 0:
        # Check for columns that look like embedding dimensions
        potential_aef = [c for c in all_cols if c not in baseline_cols and 
                        c not in ['SNo', 'S.No', 'label', 'GPS ID', 'lat', 'lon', 'class',
                                  'class description', 'date collected', 'location',
                                  'Level_1', 'Level_2', 'target_class', 'year',
                                  'system:index', '.geo']]
        # Filter to numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        aef_cols = [c for c in potential_aef if c in numeric_cols and c not in baseline_cols]
    
    all_feature_cols = baseline_cols + aef_cols
    
    print(f"\n  Baseline features ({len(baseline_cols)}): {baseline_cols}")
    print(f"  AEF features ({len(aef_cols)}): {aef_cols[:5]}... (showing first 5)")
    print(f"  Total features: {len(all_feature_cols)}")
    
    return baseline_cols, aef_cols, all_feature_cols


def clean_features(df, feature_cols):
    """
    Handles missing values and infinities in feature columns.
    """
    print(f"\n[4/6] Cleaning features...")
    
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Count missing values
    missing_before = df[feature_cols].isna().sum().sum()
    inf_count = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).sum().sum()
    print(f"  Missing values: {missing_before}")
    print(f"  Infinite values: {inf_count}")
    
    # Replace infinities with NaN, then fill NaN with column median
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    for col in feature_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    missing_after = df[feature_cols].isna().sum().sum()
    print(f"  Missing values after cleaning: {missing_after}")
    
    return df


def split_and_normalize(df, baseline_cols, aef_cols, all_feature_cols, target_col='Level_1'):
    """
    Performs stratified train/test split and Z-score normalization.
    Saves 4 dataset variants:
      1. Baseline features only (train/test)
      2. Full features with AEF (train/test)
    """
    print(f"\n[5/6] Splitting and normalizing ({target_col})...")
    
    # Encode target labels
    le = LabelEncoder()
    
    # Filter out classes with fewer than 2 instances to avoid stratify crash
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df_filtered = df[df[target_col].isin(valid_classes)].copy()
    
    if len(df_filtered) < len(df):
        print(f"  Dropped {len(df) - len(df_filtered)} rows due to rare classes (<2 samples).")
        
    df_filtered['target_encoded'] = le.fit_transform(df_filtered[target_col])
    
    # Stratified split
    X_all = df_filtered[all_feature_cols]
    y = df_filtered['target_encoded']
    coords = df_filtered[['lat', 'lon']]  # Preserve for spatial validation
    
    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        X_all, y, coords, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Class distribution (train):")
    for cls_name, cls_id in zip(le.classes_, range(len(le.classes_))):
        count = (y_train == cls_id).sum()
        print(f"    {cls_name}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Normalize with Z-score (fit on train only)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=all_feature_cols, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=all_feature_cols, 
        index=X_test.index
    )
    
    # Create baseline-only versions
    X_train_baseline = X_train_scaled[baseline_cols]
    X_test_baseline = X_test_scaled[baseline_cols]
    
    # Create full (baseline + AEF) versions
    X_train_full = X_train_scaled[all_feature_cols]
    X_test_full = X_test_scaled[all_feature_cols]
    
    # Create unnormalized versions
    X_train_baseline_unnorm = X_train[baseline_cols]
    X_test_baseline_unnorm = X_test[baseline_cols]
    X_train_full_unnorm = X_train[all_feature_cols]
    X_test_full_unnorm = X_test[all_feature_cols]
    
    return {
        'X_train_baseline': X_train_baseline,
        'X_test_baseline': X_test_baseline,
        'X_train_full': X_train_full,
        'X_test_full': X_test_full,
        'X_train_baseline_unnorm': X_train_baseline_unnorm,
        'X_test_baseline_unnorm': X_test_baseline_unnorm,
        'X_train_full_unnorm': X_train_full_unnorm,
        'X_test_full_unnorm': X_test_full_unnorm,
        'y_train': y_train,
        'y_test': y_test,
        'coords_train': coords_train,
        'coords_test': coords_test,
        'label_encoder': le,
        'scaler': scaler,
        'baseline_cols': baseline_cols,
        'aef_cols': aef_cols,
        'all_feature_cols': all_feature_cols,
    }


def save_prepared_data(data_dict, target_level='Level_1'):
    """
    Saves all prepared data splits as CSVs and serialized objects.
    """
    print(f"\n[6/6] Saving prepared data for {target_level}...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    suffix = target_level.lower()
    
    # Save feature matrices
    data_dict['X_train_baseline'].to_csv(
        os.path.join(OUTPUT_DIR, f'X_train_baseline_{suffix}.csv'), index=False)
    data_dict['X_test_baseline'].to_csv(
        os.path.join(OUTPUT_DIR, f'X_test_baseline_{suffix}.csv'), index=False)
    data_dict['X_train_full'].to_csv(
        os.path.join(OUTPUT_DIR, f'X_train_full_{suffix}.csv'), index=False)
    data_dict['X_test_full'].to_csv(
        os.path.join(OUTPUT_DIR, f'X_test_full_{suffix}.csv'), index=False)

    # Save unnormalized feature matrices
    data_dict['X_train_baseline_unnorm'].to_csv(
        os.path.join(OUTPUT_DIR, f'X_train_baseline_unnorm_{suffix}.csv'), index=False)
    data_dict['X_test_baseline_unnorm'].to_csv(
        os.path.join(OUTPUT_DIR, f'X_test_baseline_unnorm_{suffix}.csv'), index=False)
    data_dict['X_train_full_unnorm'].to_csv(
        os.path.join(OUTPUT_DIR, f'X_train_full_unnorm_{suffix}.csv'), index=False)
    data_dict['X_test_full_unnorm'].to_csv(
        os.path.join(OUTPUT_DIR, f'X_test_full_unnorm_{suffix}.csv'), index=False)
    
    # Save labels
    data_dict['y_train'].to_csv(
        os.path.join(OUTPUT_DIR, f'y_train_{suffix}.csv'), index=False, header=['target'])
    data_dict['y_test'].to_csv(
        os.path.join(OUTPUT_DIR, f'y_test_{suffix}.csv'), index=False, header=['target'])
    
    # Save coordinates for spatial validation
    data_dict['coords_train'].to_csv(
        os.path.join(OUTPUT_DIR, f'coords_train_{suffix}.csv'), index=False)
    data_dict['coords_test'].to_csv(
        os.path.join(OUTPUT_DIR, f'coords_test_{suffix}.csv'), index=False)
    
    # Save encoders and scaler
    joblib.dump(data_dict['label_encoder'], 
                os.path.join(MODEL_DIR, f'label_encoder_{suffix}.pkl'))
    joblib.dump(data_dict['scaler'], 
                os.path.join(MODEL_DIR, f'scaler_{suffix}.pkl'))
    
    # Save feature column lists for downstream scripts
    import json
    feature_meta = {
        'baseline_cols': data_dict['baseline_cols'],
        'aef_cols': data_dict['aef_cols'],
        'all_feature_cols': data_dict['all_feature_cols'],
    }
    with open(os.path.join(OUTPUT_DIR, f'feature_meta_{suffix}.json'), 'w') as f:
        json.dump(feature_meta, f, indent=2)
    
    print(f"  All data saved to: {OUTPUT_DIR}")
    print(f"  Encoders saved to: {MODEL_DIR}")


def main():
    # Step 1: Load and merge
    df = load_and_merge_ee_data()
    
    if df is None:
        print("\n" + "=" * 60)
        print("DATA NOT READY — Earth Engine exports are still processing.")
        print("Once downloads are available, place them in:")
        print(f"  {os.path.join(DATA_DIR, 'ee_exports')}/")
        print("Then re-run this script.")
        print("=" * 60)
        return
    
    # Step 2: Identify feature columns
    baseline_cols, aef_cols, all_feature_cols = identify_feature_columns(df)
    
    # Step 3: Clean
    df = clean_features(df, all_feature_cols)
    
    # Step 4: Process for Level_1
    print("\n" + "=" * 60)
    print("Processing for LEVEL 1 classification")
    print("=" * 60)
    l1_data = split_and_normalize(df, baseline_cols, aef_cols, all_feature_cols, 'Level_1')
    save_prepared_data(l1_data, 'Level_1')
    
    # Step 5: Process for Level_2
    print("\n" + "=" * 60)
    print("Processing for LEVEL 2 classification")
    print("=" * 60)
    l2_data = split_and_normalize(df, baseline_cols, aef_cols, all_feature_cols, 'Level_2')
    save_prepared_data(l2_data, 'Level_2')
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nReady for Phase 1. Run:")
    print(f"  python src/dataset_3_src/phase1_xgboost_ab.py")


if __name__ == '__main__':
    main()
