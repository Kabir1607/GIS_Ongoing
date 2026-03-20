"""
Hierarchical Binning Script
============================
Merges all extracted batch CSVs, applies the hierarchical binning system
(Level 1 + Level 2 classes), drops Unknown/Drop classes, and saves
a single consolidated file to data/dataset_2/analysis_data/.

Hierarchy (from binning_summarised.png):
  Level 1 (6 broad):  Forest, Agriculture, Grassland/Shrub, Urban/Built-up,
                       Barren/Landslide, Water
  Level 2 (granular): Dense Canopy, Secondary/Degraded, Bamboo,
                       Tree-based/Perennial Plantation, Shifting Cultivation (Jhum),
                       Wet/Valley Agriculture (Rice), Other Crop, etc.
"""

import pandas as pd
import os
import glob

# ===========================
# Configuration
# ===========================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED_DIR = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'dataset_downloaded')
BINNING_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'hierarchical_binning_1.csv')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'analysis_data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'binned_extracted_data.csv')


def load_binning_map(binning_csv):
    """Load hierarchical binning CSV and return a dict: Class_ID -> (Level_1, Level_2)"""
    df = pd.read_csv(binning_csv)
    binning_map = {}
    for _, row in df.iterrows():
        class_id = str(row['Class_ID']).strip()
        level_1 = str(row['Level_1_Class']).strip()
        level_2 = str(row['Level_2_Class']).strip()
        binning_map[class_id] = (level_1, level_2)
    return binning_map


def merge_and_bin():
    """Merge all extracted CSVs and apply hierarchical binning."""
    
    # ---- Step 1: Load binning map ----
    print("=" * 70)
    print("STEP 1: Loading hierarchical binning map")
    print("=" * 70)
    
    binning_map = load_binning_map(BINNING_CSV)
    print(f"Loaded {len(binning_map)} class-to-bin mappings")
    
    # Show the hierarchy
    level_1_classes = sorted(set(v[0] for v in binning_map.values()))
    level_2_classes = sorted(set(v[1] for v in binning_map.values()))
    print(f"Level 1 classes ({len(level_1_classes)}): {level_1_classes}")
    print(f"Level 2 classes ({len(level_2_classes)}): {level_2_classes}")
    
    # ---- Step 2: Merge all extracted batch CSVs ----
    print(f"\n{'=' * 70}")
    print("STEP 2: Merging extracted batch CSVs")
    print("=" * 70)
    
    csv_files = sorted(glob.glob(os.path.join(EXTRACTED_DIR, '*.csv')))
    print(f"Found {len(csv_files)} batch files")
    
    all_dfs = []
    for f in csv_files:
        batch = pd.read_csv(f)
        all_dfs.append(batch)
        print(f"  Loaded {os.path.basename(f)}: {len(batch)} rows")
    
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal merged rows: {len(df)}")
    
    # Remove duplicates by SNo (in case any overlap between batches)
    before_dedup = len(df)
    df = df.drop_duplicates(subset='SNo', keep='first')
    if len(df) < before_dedup:
        print(f"Removed {before_dedup - len(df)} duplicate SNo entries")
    
    # ---- Step 3: Apply binning ----
    print(f"\n{'=' * 70}")
    print("STEP 3: Applying hierarchical binning")
    print("=" * 70)
    
    # Map target_class to Level_1 and Level_2
    def get_level_1(class_id):
        class_id = str(class_id).strip()
        if class_id in binning_map:
            return binning_map[class_id][0]
        return 'UNMAPPED'
    
    def get_level_2(class_id):
        class_id = str(class_id).strip()
        if class_id in binning_map:
            return binning_map[class_id][1]
        return 'UNMAPPED'
    
    df['Level_1'] = df['target_class'].apply(get_level_1)
    df['Level_2'] = df['target_class'].apply(get_level_2)
    
    # Check for unmapped classes
    unmapped = df[df['Level_1'] == 'UNMAPPED']['target_class'].unique()
    if len(unmapped) > 0:
        print(f"\nWARNING: {len(unmapped)} unmapped class(es): {unmapped}")
    else:
        print("All classes successfully mapped!")
    
    # ---- Step 4: Drop Unknown/Drop classes ----
    print(f"\n{'=' * 70}")
    print("STEP 4: Dropping Unknown/Drop classes")
    print("=" * 70)
    
    drop_mask = df['Level_1'] == 'Unknown/Drop'
    dropped = df[drop_mask]
    print(f"Dropping {len(dropped)} rows with Level_1 = 'Unknown/Drop':")
    if len(dropped) > 0:
        for cls, count in dropped['target_class'].value_counts().items():
            print(f"  {cls}: {count} rows")
    
    df_final = df[~drop_mask].copy()
    
    # ---- Step 5: Summary & Save ----
    print(f"\n{'=' * 70}")
    print("STEP 5: Final summary and save")
    print("=" * 70)
    
    print(f"\nFinal dataset: {len(df_final)} rows, {len(df_final.columns)} columns")
    
    print(f"\n--- Level 1 Distribution ---")
    l1_counts = df_final['Level_1'].value_counts()
    print(f"{'Level 1 Class':<25} {'Count':>8} {'%':>8}")
    print("-" * 43)
    for cls, count in l1_counts.items():
        pct = count / len(df_final) * 100
        print(f"  {cls:<23} {count:>8} {pct:>7.1f}%")
    
    print(f"\n--- Level 2 Distribution ---")
    l2_counts = df_final['Level_2'].value_counts()
    print(f"{'Level 2 Class':<40} {'Count':>8} {'%':>8}")
    print("-" * 58)
    for cls, count in l2_counts.items():
        pct = count / len(df_final) * 100
        print(f"  {cls:<38} {count:>8} {pct:>7.1f}%")
    
    # Sort by SNo for consistent ordering
    df_final = df_final.sort_values('SNo').reset_index(drop=True)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"Shape: {df_final.shape}")
    
    # Show columns in the final file
    print(f"\nColumns ({len(df_final.columns)}):")
    for i, col in enumerate(df_final.columns):
        print(f"  {i+1:3d}. {col}")


if __name__ == '__main__':
    merge_and_bin()
