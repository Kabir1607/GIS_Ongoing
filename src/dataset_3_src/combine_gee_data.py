import pandas as pd
import glob
import os

gee_dir = '/home/Kdixter/Desktop/final_analysis/data/dataset_3/dataset_3_GEE_data'
output_file = '/home/Kdixter/Desktop/final_analysis/data/dataset_3/dataset_3_combined_GEE.csv'

all_files = glob.glob(os.path.join(gee_dir, "Dataset3_LULC_Extraction_*.csv"))
print(f"Found {len(all_files)} files to combine.")

dfs = []
for f in all_files:
    dfs.append(pd.read_csv(f))

combined_df = pd.concat(dfs, ignore_index=True)
print(f"Combined GEE data shape: {combined_df.shape}")

labels_file = '/home/Kdixter/Desktop/final_analysis/data/dataset_3/raw_dataset_3_binned.csv'
labels_df = pd.read_csv(labels_file)
print(f"Labels data shape: {labels_df.shape}")

# Inspect common columns to merge on
print("GEE columns:", combined_df.columns.tolist()[:5], "...")
print("Labels columns:", labels_df.columns.tolist()[:5], "...")

# GEE exports usually include 'system:index' which ranges from 0 to N-1 for the feature collection
if 'system:index' in combined_df.columns:
    # Convert system:index to integer to merge with the source dataframe's index
    combined_df['system:index'] = combined_df['system:index'].astype(str).str.replace(r'[^0-9]', '', regex=True).astype(int)
    combined_df = combined_df.sort_values('system:index').reset_index(drop=True)
    
merged_df = pd.concat([labels_df, combined_df], axis=1)
print(f"Merged output shape: {merged_df.shape}")

# Save the unified dataset
merged_df.to_csv(output_file, index=False)
print(f"Saved combined data to {output_file}")
