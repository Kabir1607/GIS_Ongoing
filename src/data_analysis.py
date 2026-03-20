"""
Data Analysis Script
====================
1. Inspects extracted CSV columns and row counts per batch
2. Analyzes binning_new.csv for classes in multiple bins
3. Merges extracted batches with cleaned_dataset_2 via SNo for row matching
   (handles the dropped NaN row by using SNo as the join key instead of index position)
"""

import pandas as pd
import os
import glob

# ===========================
# Configuration
# ===========================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED_DIR = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'dataset_downloaded')
CLEANED_CSV = os.path.join(PROJECT_DIR, 'data', 'cleaned_dataset_2.csv')
BINNING_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'binning_new.csv')
REPORT_DIR = os.path.join(PROJECT_DIR, 'reports')

def inspect_extracted_data():
    """Inspect the first extracted CSV for its fields and row counts per batch."""
    print("=" * 70)
    print("PART 1: EXTRACTED DATA INSPECTION")
    print("=" * 70)
    
    csv_files = sorted(glob.glob(os.path.join(EXTRACTED_DIR, '*.csv')))
    if not csv_files:
        print("No CSV files found in dataset_downloaded/")
        return
    
    # Show columns from first file
    first_df = pd.read_csv(csv_files[0], nrows=1)
    print(f"\nColumns in extracted CSV ({len(first_df.columns)} total):")
    for i, col in enumerate(first_df.columns):
        print(f"  {i+1:3d}. {col}")
    
    # Row counts per batch
    print(f"\n{'Batch File':<50} {'Rows':>8}")
    print("-" * 60)
    total_extracted = 0
    for f in csv_files:
        count = sum(1 for _ in open(f)) - 1  # subtract header
        total_extracted += count
        print(f"  {os.path.basename(f):<48} {count:>8}")
    print("-" * 60)
    print(f"  {'TOTAL EXTRACTED ROWS':<48} {total_extracted:>8}")
    
    # Compare with cleaned dataset
    cleaned_count = sum(1 for _ in open(CLEANED_CSV)) - 1
    print(f"  {'CLEANED DATASET ROWS':<48} {cleaned_count:>8}")
    print(f"  {'MISSING (cleaned - extracted)':<48} {cleaned_count - total_extracted:>8}")
    
    return total_extracted, cleaned_count


def analyze_binning():
    """Analyze binning_new.csv: row counts and classes falling into multiple bins."""
    print("\n" + "=" * 70)
    print("PART 2: BINNING ANALYSIS")
    print("=" * 70)
    
    df = pd.read_csv(BINNING_CSV)
    
    # The bin columns are everything after Class_ID and Description
    bin_columns = [c for c in df.columns if c not in ['Class_ID', 'Description']]
    
    print(f"\nBin categories ({len(bin_columns)}): {bin_columns}")
    print(f"Total classes in binning table: {len(df)}")
    
    # For each class, count how many bins it maps to
    print(f"\n{'Class_ID':<30} {'Description':<35} {'Bins Assigned':>5}  Bin Names")
    print("-" * 120)
    
    multi_bin_classes = []
    unbinned_classes = []
    
    for _, row in df.iterrows():
        class_id = row['Class_ID']
        desc = row['Description']
        
        # A cell is "marked" if it has a 1 or any non-empty/non-NaN value
        assigned_bins = []
        for col in bin_columns:
            val = row[col]
            if pd.notna(val):
                # Check if it's a number 1 or a text note
                try:
                    if float(val) == 1:
                        assigned_bins.append(col)
                except (ValueError, TypeError):
                    # It's a text note like "not sure if tree or crops"
                    assigned_bins.append(f"{col}(?)")
        
        bin_count = len(assigned_bins)
        marker = " *** MULTI-BIN ***" if bin_count > 1 else ""
        if bin_count == 0:
            marker = " *** UNBINNED ***"
            unbinned_classes.append(class_id)
        
        if bin_count > 1:
            multi_bin_classes.append((class_id, desc, assigned_bins))
        
        print(f"  {str(class_id):<28} {str(desc):<35} {bin_count:>3}    {', '.join(assigned_bins)}{marker}")
    
    # Summary
    print(f"\n--- SUMMARY ---")
    print(f"Classes in multiple bins: {len(multi_bin_classes)}")
    for cid, desc, bins in multi_bin_classes:
        print(f"  {cid} ({desc}) -> {bins}")
    
    print(f"\nUnbinned classes: {len(unbinned_classes)}")
    for cid in unbinned_classes:
        print(f"  {cid}")
    
    return multi_bin_classes, unbinned_classes


def analyze_class_counts_in_cleaned_data():
    """Count how many rows in cleaned_dataset_2 belong to each class."""
    print("\n" + "=" * 70)
    print("PART 3: CLASS ROW COUNTS IN CLEANED DATASET")
    print("=" * 70)
    
    df = pd.read_csv(CLEANED_CSV)
    class_counts = df['class'].value_counts().sort_index()
    
    print(f"\n{'Class':<30} {'Row Count':>10}")
    print("-" * 42)
    for cls, count in class_counts.items():
        print(f"  {cls:<28} {count:>10}")
    print("-" * 42)
    print(f"  {'TOTAL':<28} {class_counts.sum():>10}")
    
    return class_counts


def analyze_row_matching():
    """
    Figure out how to match extracted rows back to cleaned dataset.
    
    Problem: 1 row was dropped due to NaN coordinates during extraction,
    so positional index-based matching won't work. 
    
    Solution: Use the 'SNo' column as the join key. The extraction code
    stores SNo from cleaned_dataset_2 into each extracted feature, so we
    can reliably join on SNo regardless of dropped rows.
    """
    print("\n" + "=" * 70)
    print("PART 4: ROW MATCHING STRATEGY (NaN-dropped row handling)")
    print("=" * 70)
    
    # Load cleaned dataset to check for NaN coordinates
    cleaned_df = pd.read_csv(CLEANED_CSV)
    nan_rows = cleaned_df[cleaned_df['lon'].isna() | cleaned_df['lat'].isna()]
    
    print(f"\nCleaned dataset shape: {cleaned_df.shape}")
    print(f"Rows with NaN lat/lon: {len(nan_rows)}")
    if len(nan_rows) > 0:
        print("NaN rows (SNo values):")
        for _, row in nan_rows.iterrows():
            print(f"  SNo={row['SNo']}, class={row['class']}, lat={row['lat']}, lon={row['lon']}")
    
    # Load first batch and check SNo continuity
    first_batch = pd.read_csv(
        os.path.join(EXTRACTED_DIR, 'LULC_Data_Extraction_0_to_2500.csv'),
        nrows=10
    )
    print(f"\nFirst 10 SNo values in extracted data: {first_batch['SNo'].tolist()}")
    print(f"First 10 SNo values in cleaned data:   {cleaned_df['SNo'].head(10).tolist()}")
    
    # Check if SNo values match
    sno_match = (first_batch['SNo'].values == cleaned_df['SNo'].head(10).values).all()
    print(f"SNo values match between extracted and cleaned: {sno_match}")
    
    # Concatenate all extracted batches to find actual gap
    print("\nLoading all extracted batches to find any SNo gaps...")
    all_extracted = []
    csv_files = sorted(glob.glob(os.path.join(EXTRACTED_DIR, '*.csv')))
    for f in csv_files:
        batch = pd.read_csv(f, usecols=['SNo', 'target_class'])
        all_extracted.append(batch)
    
    combined = pd.concat(all_extracted, ignore_index=True)
    combined_snos = set(combined['SNo'].values)
    cleaned_snos = set(cleaned_df['SNo'].values)
    
    # Only consider the range 1 to max(extracted SNo)
    max_extracted_sno = max(combined_snos)
    expected_snos = set(range(1, max_extracted_sno + 1))
    
    missing_in_extracted = expected_snos - combined_snos
    print(f"\nMax SNo in extracted data: {max_extracted_sno}")
    print(f"Total unique SNo values extracted: {len(combined_snos)}")
    print(f"Missing SNo values (within range): {len(missing_in_extracted)}")
    if missing_in_extracted and len(missing_in_extracted) <= 20:
        missing_list = sorted(missing_in_extracted)
        print(f"Missing SNo values: {missing_list}")
        # Look up what class these missing SNo values belong to
        for sno in missing_list:
            match = cleaned_df[cleaned_df['SNo'] == sno]
            if len(match) > 0:
                row = match.iloc[0]
                print(f"  SNo {sno}: class={row['class']}, lat={row['lat']}, lon={row['lon']}")
    
    print(f"\n--- ROW MATCHING RECOMMENDATION ---")
    print(f"Use 'SNo' as the join key between extracted data and cleaned_dataset_2.")
    print(f"This bypasses the positional offset caused by the dropped NaN row(s).")
    print(f"The extracted CSVs contain 'SNo' and 'target_class' fields that directly")
    print(f"correspond to the cleaned dataset's 'SNo' and 'class' columns.")
    
    return combined


if __name__ == '__main__':
    inspect_extracted_data()
    analyze_binning()
    analyze_class_counts_in_cleaned_data()
    analyze_row_matching()
