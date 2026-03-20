"""
Comprehensive EDA Script
========================
Performs full Exploratory Data Analysis on the binned extracted dataset.
Outputs:
  - Text report  -> reports/EDA_results.txt
  - Bar charts   -> data_visuals/
"""

import pandas as pd
import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================
# Configuration
# ===========================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'analysis_data', 'binned_extracted_data.csv')
REPORT_FILE = os.path.join(PROJECT_DIR, 'reports', 'EDA_results.txt')
VISUALS_DIR = os.path.join(PROJECT_DIR, 'data_visuals')

# Metadata columns (not features)
META_COLS = ['system:index', 'SNo', 'target_class', 'year', '.geo', 'Level_1', 'Level_2']

# Feature groups for organized analysis
FEATURE_GROUPS = {
    'AEF Embeddings (A00-A63)': [f'A{str(i).zfill(2)}' for i in range(64)],
    'Sentinel-2 Bands': ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'],
    'Landsat 8 Bands': ['L8_blue', 'L8_green', 'L8_red', 'L8_nir', 'L8_swir1', 'L8_swir2'],
    'Spectral Indices': ['NDVI', 'NDWI', 'GCVI'],
    'Climate': ['Precipitation'],
}


class TeeWriter:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, 'w')
    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    def close(self):
        self.file.close()
        sys.stdout = self.terminal


def section(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def run_eda():
    os.makedirs(VISUALS_DIR, exist_ok=True)
    tee = TeeWriter(REPORT_FILE)
    sys.stdout = tee

    print("EXPLORATORY DATA ANALYSIS REPORT")
    print(f"Generated from: {DATA_CSV}")
    print(f"{'=' * 80}")

    df = pd.read_csv(DATA_CSV)
    feat_cols = [c for c in df.columns if c not in META_COLS]

    # ====================================================================
    # 1. DATASET OVERVIEW
    # ====================================================================
    section("1. DATASET OVERVIEW")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Feature columns: {len(feat_cols)}")
    print(f"Metadata columns: {len(META_COLS)}")
    print(f"\nData types:")
    for dtype, count in df[feat_cols].dtypes.value_counts().items():
        print(f"  {dtype}: {count}")
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # ====================================================================
    # 2. MISSING VALUES
    # ====================================================================
    section("2. MISSING VALUES")
    nan_counts = df[feat_cols].isna().sum()
    nan_any = nan_counts[nan_counts > 0]
    if len(nan_any) == 0:
        print("No missing values in any feature column.")
    else:
        print(f"Features with missing values ({len(nan_any)}):")
        for col, count in nan_any.items():
            pct = count / len(df) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")

    # ====================================================================
    # 3. DESCRIPTIVE STATISTICS (grouped by feature type)
    # ====================================================================
    section("3. DESCRIPTIVE STATISTICS BY FEATURE GROUP")
    for group_name, cols in FEATURE_GROUPS.items():
        present_cols = [c for c in cols if c in df.columns]
        if not present_cols:
            continue
        print(f"\n--- {group_name} ({len(present_cols)} features) ---")
        stats = df[present_cols].describe().T
        stats['skew'] = df[present_cols].skew()
        stats['kurtosis'] = df[present_cols].kurtosis()
        # Compact display
        print(f"{'Feature':<14} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Skew':>8} {'Kurt':>8}")
        print("-" * 74)
        for idx, row in stats.iterrows():
            print(f"  {idx:<12} {row['min']:>10.2f} {row['max']:>10.2f} "
                  f"{row['mean']:>10.2f} {row['std']:>10.2f} "
                  f"{row['skew']:>8.2f} {row['kurtosis']:>8.2f}")

    # ====================================================================
    # 4. CLASS BALANCE (Level 1 and Level 2)
    # ====================================================================
    section("4. CLASS BALANCE")

    print("--- Level 1 (Broad Classes) ---")
    l1_counts = df['Level_1'].value_counts()
    print(f"{'Class':<28} {'Count':>8} {'%':>8} {'Bar'}")
    print("-" * 70)
    max_count = l1_counts.max()
    for cls, count in l1_counts.items():
        pct = count / len(df) * 100
        bar = '█' * int(count / max_count * 30)
        print(f"  {cls:<26} {count:>8} {pct:>7.1f}% {bar}")

    print(f"\nImbalance ratio (max/min): {l1_counts.max() / l1_counts.min():.1f}x")

    print(f"\n--- Level 2 (Granular Sub-Classes) ---")
    l2_counts = df['Level_2'].value_counts()
    print(f"{'Class':<40} {'Count':>8} {'%':>8}")
    print("-" * 58)
    max_count = l2_counts.max()
    for cls, count in l2_counts.items():
        pct = count / len(df) * 100
        bar = '█' * int(count / max_count * 30)
        print(f"  {cls:<38} {count:>8} {pct:>7.1f}%  {bar}")

    print(f"\nImbalance ratio (max/min): {l2_counts.max() / l2_counts.min():.1f}x")

    print(f"\n--- Original target_class distribution ---")
    tc_counts = df['target_class'].value_counts()
    print(f"{'Class':<28} {'Count':>8} {'%':>8}")
    print("-" * 48)
    for cls, count in tc_counts.items():
        pct = count / len(df) * 100
        print(f"  {cls:<26} {count:>8} {pct:>7.1f}%")

    # ====================================================================
    # 5. YEAR DISTRIBUTION
    # ====================================================================
    section("5. YEAR DISTRIBUTION")
    year_counts = df['year'].value_counts().sort_index()
    print(f"{'Year':>6} {'Count':>8} {'%':>8}")
    print("-" * 26)
    for yr, count in year_counts.items():
        pct = count / len(df) * 100
        print(f"  {yr:>4} {count:>8} {pct:>7.1f}%")

    # ====================================================================
    # 6. OUTLIER DETECTION (IQR method)
    # ====================================================================
    section("6. OUTLIER DETECTION (IQR Method)")
    # Only check non-AEF features (AEF are embeddings, not raw sensor values)
    outlier_cols = []
    for group, cols in FEATURE_GROUPS.items():
        if 'AEF' not in group:
            outlier_cols.extend([c for c in cols if c in df.columns])

    print(f"{'Feature':<14} {'IQR':>10} {'Lower':>10} {'Upper':>10} {'Outliers':>10} {'%':>8}")
    print("-" * 66)
    for col in outlier_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = n_outliers / len(df) * 100
        print(f"  {col:<12} {iqr:>10.2f} {lower:>10.2f} {upper:>10.2f} {n_outliers:>10} {pct:>7.1f}%")

    # ====================================================================
    # 7. CORRELATION ANALYSIS
    # ====================================================================
    section("7. CORRELATION ANALYSIS")

    # Non-AEF features for readable correlation matrix
    non_aef_feats = []
    for group, cols in FEATURE_GROUPS.items():
        if 'AEF' not in group:
            non_aef_feats.extend([c for c in cols if c in df.columns])

    corr = df[non_aef_feats].corr()

    # Highly correlated pairs
    print("Highly correlated feature pairs (|r| > 0.8):")
    print(f"{'Feature 1':<14} {'Feature 2':<14} {'Correlation':>12}")
    print("-" * 42)
    seen = set()
    for i, c1 in enumerate(non_aef_feats):
        for j, c2 in enumerate(non_aef_feats):
            if i >= j:
                continue
            r = corr.loc[c1, c2]
            if abs(r) > 0.8 and (c1, c2) not in seen:
                seen.add((c1, c2))
                print(f"  {c1:<12} {c2:<12} {r:>12.3f}")

    # Correlation heatmap (non-AEF)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=True, ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix (Non-AEF Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'correlation_heatmap.png'), dpi=150)
    plt.close()
    print(f"\nCorrelation heatmap saved to data_visuals/correlation_heatmap.png")

    # ====================================================================
    # 8. FEATURE DISTRIBUTIONS (box plots per group, colored by Level 1)
    # ====================================================================
    section("8. FEATURE DISTRIBUTION PLOTS")

    for group_name, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if not present or 'AEF' in group_name:
            continue  # Skip AEF box plots (64 features would be unreadable)

        fig, ax = plt.subplots(figsize=(max(10, len(present)*2), 6))
        df_melt = df[present + ['Level_1']].melt(id_vars='Level_1',
                                                  var_name='Feature',
                                                  value_name='Value')
        sns.boxplot(data=df_melt, x='Feature', y='Value', hue='Level_1', ax=ax)
        ax.set_title(f'Distribution: {group_name} by Level 1 Class', fontsize=13, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        safe_name = group_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(VISUALS_DIR, f'boxplot_{safe_name}.png'), dpi=150)
        plt.close()
        print(f"Box plot saved: boxplot_{safe_name}.png")

    # AEF summary distribution (aggregate stats)
    aef_cols = [c for c in FEATURE_GROUPS['AEF Embeddings (A00-A63)'] if c in df.columns]
    if aef_cols:
        fig, ax = plt.subplots(figsize=(14, 5))
        aef_means = df[aef_cols].mean()
        aef_stds = df[aef_cols].std()
        x = range(len(aef_cols))
        ax.bar(x, aef_means, yerr=aef_stds, alpha=0.7, capsize=2, color='steelblue')
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Mean ± Std')
        ax.set_title('AEF V1 Embedding Dimensions: Mean ± Std', fontsize=13, fontweight='bold')
        ax.set_xticks(x[::4])
        ax.set_xticklabels([aef_cols[i] for i in range(0, len(aef_cols), 4)], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, 'aef_embedding_summary.png'), dpi=150)
        plt.close()
        print("AEF embedding summary saved: aef_embedding_summary.png")

    # ====================================================================
    # 9. PER-CLASS FEATURE MEANS (Level 1)
    # ====================================================================
    section("9. PER-CLASS FEATURE MEANS (Level 1)")
    for group_name, cols in FEATURE_GROUPS.items():
        present = [c for c in cols if c in df.columns]
        if not present or 'AEF' in group_name:
            continue
        print(f"\n--- {group_name} ---")
        grouped = df.groupby('Level_1')[present].mean()
        print(grouped.round(2).to_string())

    # ====================================================================
    # 10. BINNING POPULATION BAR CHARTS
    # ====================================================================
    section("10. BINNING POPULATION")

    # Level 1 bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_l1 = sns.color_palette('Set2', len(l1_counts))
    bars = ax.bar(range(len(l1_counts)), l1_counts.values, color=colors_l1)
    ax.set_xticks(range(len(l1_counts)))
    ax.set_xticklabels(l1_counts.index, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Level 1 Binning Population', fontsize=14, fontweight='bold')
    # Add count labels on bars
    for bar, count in zip(bars, l1_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'binning_population_level1.png'), dpi=150)
    plt.close()
    print("Saved: binning_population_level1.png")

    # Level 2 bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    colors_l2 = sns.color_palette('Set2', len(l2_counts))
    bars = ax.bar(range(len(l2_counts)), l2_counts.values, color=colors_l2)
    ax.set_xticks(range(len(l2_counts)))
    ax.set_xticklabels(l2_counts.index, rotation=35, ha='right', fontsize=10)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Level 2 Binning Population', fontsize=14, fontweight='bold')
    for bar, count in zip(bars, l2_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{count:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'binning_population_level2.png'), dpi=150)
    plt.close()
    print("Saved: binning_population_level2.png")

    # Combined stacked chart: Level 2 within Level 1
    fig, ax = plt.subplots(figsize=(14, 7))
    cross = pd.crosstab(df['Level_2'], df['Level_1'])
    cross.plot(kind='barh', stacked=True, ax=ax, colormap='Set2')
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_title('Level 2 Sub-Classes within Level 1 Groups', fontsize=14, fontweight='bold')
    ax.legend(title='Level 1', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'binning_hierarchy_stacked.png'), dpi=150)
    plt.close()
    print("Saved: binning_hierarchy_stacked.png")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    section("SUMMARY")
    print(f"Dataset: {df.shape[0]} rows, {len(feat_cols)} features")
    print(f"Level 1 classes: {df['Level_1'].nunique()} | Level 2 classes: {df['Level_2'].nunique()}")
    has_nan = len(nan_any) > 0 if not isinstance(nan_any, pd.Series) else not nan_any.empty
    print(f"Missing values: {len(nan_any) if has_nan else 'None'}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"\nReport saved to: {REPORT_FILE}")
    print(f"Visualizations saved to: {VISUALS_DIR}/")

    tee.close()


if __name__ == '__main__':
    run_eda()
