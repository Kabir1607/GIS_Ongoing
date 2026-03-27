import pandas as pd
import os

def apply_binning():
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_3', 'raw_dataset_3.csv')
    BINNING_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_3', 'binning_dataset_3.csv')
    OUTPUT_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_3', 'raw_dataset_3_binned.csv')
    REPORT_PATH = os.path.join(PROJECT_DIR, 'data', 'dataset_3', 'binning_report.txt')
    
    print(f"Loading raw data from: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    
    print(f"Loading binning map from: {BINNING_CSV}")
    binning = pd.read_csv(BINNING_CSV)
    
    # Create lookup dictionaries
    l1_map = dict(zip(binning['raw_class'].str.strip(), binning['Level_1'].str.strip()))
    l2_map = dict(zip(binning['raw_class'].str.strip(), binning['Level_2'].str.strip()))
    
    # Apply mappings
    df['class'] = df['class'].str.strip()
    df['Level_1'] = df['class'].map(l1_map)
    df['Level_2'] = df['class'].map(l2_map)
    
    # Report unmapped classes
    unmapped = df[df['Level_1'].isna()]['class'].unique()
    if len(unmapped) > 0:
        print(f"\nWARNING: {len(unmapped)} classes could not be mapped:")
        for u in unmapped:
            count = len(df[df['class'] == u])
            print(f"  '{u}' ({count} rows)")
    else:
        print("All classes mapped successfully!")
    
    # Generate report
    lines = []
    lines.append("=" * 60)
    lines.append("DATASET 3 BINNING REPORT")
    lines.append("=" * 60 + "\n")
    
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Mapped rows: {df['Level_1'].notna().sum()}")
    lines.append(f"Unmapped rows: {df['Level_1'].isna().sum()}\n")
    
    lines.append("--- Level 1 Distribution ---")
    l1_counts = df['Level_1'].value_counts()
    for cls, cnt in l1_counts.items():
        pct = cnt / len(df) * 100
        lines.append(f"  {cls:<25} {cnt:>5} ({pct:.1f}%)")
    
    lines.append("\n--- Level 2 Distribution ---")
    l2_counts = df['Level_2'].value_counts()
    for cls, cnt in l2_counts.items():
        pct = cnt / len(df) * 100
        lines.append(f"  {cls:<25} {cnt:>5} ({pct:.1f}%)")
    
    with open(REPORT_PATH, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nBinning report saved to: {REPORT_PATH}")
    
    # Save binned CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Binned dataset saved to: {OUTPUT_CSV}")

if __name__ == '__main__':
    apply_binning()
