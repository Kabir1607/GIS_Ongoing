"""
Lost Data Analysis
==================
Identifies which classes lost data due to the failed batch (rows beyond SNo 40000)
and the dropped NaN row (SNo 18391), and calculates the percentage impact.
"""

import pandas as pd
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_CSV = os.path.join(PROJECT_DIR, 'data', 'cleaned_dataset_2.csv')

def main():
    df = pd.read_csv(CLEANED_CSV)
    
    # The extracted data covers SNo 1–40001 (40,000 rows), minus the NaN row (SNo 18391)
    # Everything with SNo > 40001 is lost, plus the NaN row
    max_extracted_sno = 40001
    nan_sno = 18391
    
    extracted_mask = (df['SNo'] <= max_extracted_sno) & (df['SNo'] != nan_sno)
    lost_mask = ~extracted_mask
    
    df_extracted = df[extracted_mask]
    df_lost = df[lost_mask]
    
    total_rows = len(df)
    total_lost = len(df_lost)
    
    print("=" * 80)
    print("LOST DATA ANALYSIS")
    print(f"Total rows in cleaned dataset: {total_rows}")
    print(f"Successfully extracted:         {len(df_extracted)} ({len(df_extracted)/total_rows*100:.2f}%)")
    print(f"Total lost:                     {total_lost} ({total_lost/total_rows*100:.2f}%)")
    print("=" * 80)
    
    # Per-class breakdown
    total_per_class = df['class'].value_counts().sort_index()
    lost_per_class = df_lost['class'].value_counts().sort_index()
    extracted_per_class = df_extracted['class'].value_counts().sort_index()
    
    print(f"\n{'Class':<28} {'Total':>7} {'Extracted':>10} {'Lost':>6} {'% Lost':>8} {'% of Total Lost':>16}")
    print("-" * 80)
    
    for cls in total_per_class.index:
        total = total_per_class[cls]
        lost = lost_per_class.get(cls, 0)
        extracted = extracted_per_class.get(cls, 0)
        pct_of_class = (lost / total * 100) if total > 0 else 0
        pct_of_total_lost = (lost / total_lost * 100) if total_lost > 0 else 0
        
        flag = ""
        if lost > 0 and pct_of_class >= 50:
            flag = " *** >50% LOST ***"
        elif lost == total:
            flag = " *** ALL LOST ***"
        
        print(f"  {cls:<26} {total:>7} {extracted:>10} {lost:>6} {pct_of_class:>7.1f}% {pct_of_total_lost:>14.1f}%{flag}")
    
    print("-" * 80)
    total_check = total_per_class.sum()
    lost_check = lost_per_class.sum() if len(lost_per_class) > 0 else 0
    extracted_check = extracted_per_class.sum()
    print(f"  {'TOTAL':<26} {total_check:>7} {extracted_check:>10} {lost_check:>6} {lost_check/total_check*100:>7.1f}%")
    
    # Classes that are ENTIRELY lost
    print(f"\n--- Classes ENTIRELY lost (100% of samples missing) ---")
    entirely_lost = [cls for cls in total_per_class.index 
                     if lost_per_class.get(cls, 0) == total_per_class[cls]]
    if entirely_lost:
        for cls in entirely_lost:
            print(f"  {cls}: {total_per_class[cls]} rows")
    else:
        print("  None")
    
    # Classes with >50% loss
    print(f"\n--- Classes with >50% data lost ---")
    heavy_loss = [cls for cls in total_per_class.index 
                  if 0 < lost_per_class.get(cls, 0) < total_per_class[cls]
                  and lost_per_class.get(cls, 0) / total_per_class[cls] > 0.5]
    if heavy_loss:
        for cls in heavy_loss:
            lost = lost_per_class[cls]
            total = total_per_class[cls]
            print(f"  {cls}: {lost}/{total} lost ({lost/total*100:.1f}%)")
    else:
        print("  None")
    
    # Classes with zero loss
    print(f"\n--- Classes with NO data lost ---")
    no_loss = [cls for cls in total_per_class.index if lost_per_class.get(cls, 0) == 0]
    for cls in no_loss:
        print(f"  {cls}: {total_per_class[cls]} rows (all retained)")


if __name__ == '__main__':
    main()
