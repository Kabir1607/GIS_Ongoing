"""
Spectral Index Discovery from Unsupervised Learning — Dataset 2
================================================================
Reverse-engineers PCA/LDA components into interpretable spectral
index formulas by exhaustively testing band combinations and
ranking by correlation to unsupervised components + class separability.

Phases:
  A) PCA/LDA loading analysis (which bands dominate each component)
  B) Exhaustive 2-band and 3-band candidate formula generation
  C) Correlation against PCA/LDA components
  D) Class separability scoring (ANOVA F-stat, Fisher Discriminant Ratio)
  E) Comparison against known spectral indices

Usage:
    python src/spectral_index_discovery.py
"""

import pandas as pd
import numpy as np
import os
import itertools
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORMALIZED_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'analysis_data', 'normalized_data.csv')
RAW_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'analysis_data', 'binned_extracted_data.csv')
VISUALS_DIR = os.path.join(PROJECT_DIR, 'data_visuals', 'spectral_discovery')
REPORT_DIR = os.path.join(PROJECT_DIR, 'reports')

META_COLS = ['system:index', 'SNo', 'target_class', 'year', '.geo', 'Level_1', 'Level_2']

# Raw spectral bands (the ones we can combine into formulas)
S2_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
L8_BANDS = ['L8_blue', 'L8_green', 'L8_red', 'L8_nir', 'L8_swir1', 'L8_swir2']
ALL_BANDS = S2_BANDS + L8_BANDS

# Known spectral indices for comparison
KNOWN_INDICES = {
    'NDVI':  ('nir', 'red',   lambda a, b: (a - b) / (a + b + 1e-10)),
    'NDWI':  ('green', 'nir', lambda a, b: (a - b) / (a + b + 1e-10)),
    'GCVI':  ('nir', 'green', lambda a, b: (a / (b + 1e-10)) - 1),
    'MNDWI': ('green', 'swir1', lambda a, b: (a - b) / (a + b + 1e-10)),
    'NDBI':  ('swir1', 'nir',   lambda a, b: (a - b) / (a + b + 1e-10)),
    'BSI':   None,  # 4-band; handled separately
    'SAVI':  ('nir', 'red',   lambda a, b: 1.5 * (a - b) / (a + b + 0.5)),
    'EVI':   None,  # 3-band; handled separately
    'LSWI':  ('nir', 'swir1', lambda a, b: (a - b) / (a + b + 1e-10)),
    'NBR':   ('nir', 'swir2', lambda a, b: (a - b) / (a + b + 1e-10)),
}

N_PCA_COMPONENTS = 5
EPSILON = 1e-10  # Avoid division by zero


# ──────────────────────────────────────────────────────────────────
# PHASE A: PCA/LDA Loading Analysis
# ──────────────────────────────────────────────────────────────────
def phase_a_loadings(X, feat_cols, labels, report):
    """Extracts and reports PCA and LDA loadings on RAW spectral bands only."""
    # Identify which columns are raw bands
    band_indices = [i for i, c in enumerate(feat_cols) if c in ALL_BANDS]
    band_names = [feat_cols[i] for i in band_indices]
    
    report.append("\n" + "=" * 70)
    report.append("PHASE A: PCA / LDA LOADING ANALYSIS")
    report.append("=" * 70)
    
    # ── PCA on ALL features first ──
    pca_all = PCA(n_components=N_PCA_COMPONENTS)
    X_pca = pca_all.fit_transform(X)
    
    report.append(f"\nPCA on all {len(feat_cols)} features:")
    report.append(f"  Top {N_PCA_COMPONENTS} components explain {sum(pca_all.explained_variance_ratio_)*100:.1f}%")
    
    # ── PCA on RAW BANDS only ──
    X_bands = X[:, band_indices]
    pca_bands = PCA(n_components=min(N_PCA_COMPONENTS, len(band_names)))
    X_pca_bands = pca_bands.fit_transform(X_bands)
    
    report.append(f"\nPCA on {len(band_names)} RAW spectral bands only:")
    for i in range(pca_bands.n_components_):
        report.append(f"\n  PC{i+1} ({pca_bands.explained_variance_ratio_[i]*100:.1f}% variance):")
        loadings = pca_bands.components_[i]
        # Sort by absolute magnitude
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        for j in sorted_idx:
            report.append(f"    {band_names[j]:>10}: {loadings[j]:+.4f}")
    
    # ── LDA on raw bands ──
    lda = LDA()
    X_lda = lda.fit_transform(X_bands, labels)
    
    lda_var = lda.explained_variance_ratio_
    report.append(f"\nLDA on RAW bands (target: Level_1):")
    for i in range(len(lda_var)):
        report.append(f"\n  LD{i+1} ({lda_var[i]*100:.1f}% between-class variance):")
        loadings = lda.scalings_[:, i]
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        for j in sorted_idx:
            report.append(f"    {band_names[j]:>10}: {loadings[j]:+.4f}")
    
    # ── Visualization: loading heatmap ──
    os.makedirs(VISUALS_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # PCA heatmap
    pca_df = pd.DataFrame(pca_bands.components_, 
                          columns=band_names,
                          index=[f'PC{i+1}' for i in range(pca_bands.n_components_)])
    sns.heatmap(pca_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=axes[0])
    axes[0].set_title('PCA Loadings (Raw Bands)', fontsize=13, fontweight='bold')
    
    # LDA heatmap
    n_lda = min(lda.scalings_.shape[1], 5)
    lda_df = pd.DataFrame(lda.scalings_[:, :n_lda].T, 
                          columns=band_names,
                          index=[f'LD{i+1}' for i in range(n_lda)])
    sns.heatmap(lda_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=axes[1])
    axes[1].set_title('LDA Scalings (Raw Bands → Level 1)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'loading_heatmaps.png'), dpi=150)
    plt.close()
    
    return pca_bands, lda, X_pca_bands, X_lda, band_indices, band_names


# ──────────────────────────────────────────────────────────────────
# PHASE B: Exhaustive Candidate Generation
# ──────────────────────────────────────────────────────────────────
def phase_b_generate_candidates(X, feat_cols, band_indices, band_names, report):
    """Generates all 2-band and 3-band candidate spectral indices."""
    report.append("\n" + "=" * 70)
    report.append("PHASE B: EXHAUSTIVE CANDIDATE GENERATION")
    report.append("=" * 70)
    
    X_bands = X[:, band_indices]
    n_bands = len(band_names)
    n_samples = X_bands.shape[0]
    
    candidates = {}  # name → values array
    formulas = {}    # name → formula string
    
    # ── 2-band operations ──
    for i, j in itertools.combinations(range(n_bands), 2):
        a, b = X_bands[:, i], X_bands[:, j]
        na, nb = band_names[i], band_names[j]
        
        # Normalized Difference: (A - B) / (A + B)
        nd = (a - b) / (a + b + EPSILON)
        key = f'ND({na},{nb})'
        candidates[key] = nd
        formulas[key] = f'({na} - {nb}) / ({na} + {nb})'
        
        # Reverse normalized difference
        nd_rev = (b - a) / (a + b + EPSILON)
        key_rev = f'ND({nb},{na})'
        candidates[key_rev] = nd_rev
        formulas[key_rev] = f'({nb} - {na}) / ({nb} + {na})'
        
        # Simple ratio
        sr = a / (b + EPSILON)
        key_sr = f'SR({na},{nb})'
        candidates[key_sr] = sr
        formulas[key_sr] = f'{na} / {nb}'
        
        sr_rev = b / (a + EPSILON)
        key_sr_rev = f'SR({nb},{na})'
        candidates[key_sr_rev] = sr_rev
        formulas[key_sr_rev] = f'{nb} / {na}'
        
        # Difference
        diff = a - b
        key_d = f'DI({na},{nb})'
        candidates[key_d] = diff
        formulas[key_d] = f'{na} - {nb}'
        
        # Product (normalized by sum to keep scale bounded)
        prod = (a * b) / (a + b + EPSILON)
        key_p = f'PR({na},{nb})'
        candidates[key_p] = prod
        formulas[key_p] = f'({na} * {nb}) / ({na} + {nb})'
    
    # ── 3-band operations (EVI-like) ──
    for i, j, k in itertools.combinations(range(n_bands), 3):
        a, b, c = X_bands[:, i], X_bands[:, j], X_bands[:, k]
        na, nb, nc = band_names[i], band_names[j], band_names[k]
        
        # (A - B) / (A + C) — generalized normalized difference with alternative denominator
        val = (a - b) / (a + c + EPSILON)
        key = f'T3({na},{nb},{nc})'
        candidates[key] = val
        formulas[key] = f'({na} - {nb}) / ({na} + {nc})'
        
        # (A - B) / (B + C) — another variant
        val2 = (a - b) / (b + c + EPSILON)
        key2 = f'T3b({na},{nb},{nc})'
        candidates[key2] = val2
        formulas[key2] = f'({na} - {nb}) / ({nb} + {nc})'
    
    # Remove any candidates with NaN or Inf
    valid_candidates = {}
    valid_formulas = {}
    for key, vals in candidates.items():
        if np.all(np.isfinite(vals)):
            valid_candidates[key] = vals
            valid_formulas[key] = formulas[key]
        else:
            # Replace inf with nan, then impute median
            clean = np.where(np.isfinite(vals), vals, np.nan)
            median = np.nanmedian(clean)
            clean = np.where(np.isnan(clean), median, clean)
            if np.all(np.isfinite(clean)):
                valid_candidates[key] = clean
                valid_formulas[key] = formulas[key]
    
    n2_band = sum(1 for k in valid_candidates if k.startswith(('ND(', 'SR(', 'DI(', 'PR(')))
    n3_band = sum(1 for k in valid_candidates if k.startswith('T3'))
    
    report.append(f"\n  Generated {len(valid_candidates)} total candidate formulas:")
    report.append(f"    2-band combinations: {n2_band}")
    report.append(f"    3-band combinations: {n3_band}")
    
    return valid_candidates, valid_formulas


# ──────────────────────────────────────────────────────────────────
# PHASE C: Correlation Against Unsupervised Components
# ──────────────────────────────────────────────────────────────────
def phase_c_correlations(candidates, formulas, X_pca, X_lda, report):
    """Pearson correlation of each candidate against PCA/LDA components."""
    report.append("\n" + "=" * 70)
    report.append("PHASE C: CORRELATION AGAINST PCA / LDA COMPONENTS")
    report.append("=" * 70)
    
    results = []
    
    for key, vals in candidates.items():
        row = {'name': key, 'formula': formulas[key]}
        
        # Correlate against each PC
        best_pc_r = 0
        best_pc_id = None
        for pc_idx in range(X_pca.shape[1]):
            r, p = stats.pearsonr(vals, X_pca[:, pc_idx])
            row[f'r_PC{pc_idx+1}'] = r
            row[f'r2_PC{pc_idx+1}'] = r ** 2
            if abs(r) > abs(best_pc_r):
                best_pc_r = r
                best_pc_id = f'PC{pc_idx+1}'
        
        row['best_PC'] = best_pc_id
        row['best_PC_r'] = best_pc_r
        row['best_PC_r2'] = best_pc_r ** 2
        
        # Correlate against each LD
        best_ld_r = 0
        best_ld_id = None
        for ld_idx in range(X_lda.shape[1]):
            r, p = stats.pearsonr(vals, X_lda[:, ld_idx])
            row[f'r_LD{ld_idx+1}'] = r
            row[f'r2_LD{ld_idx+1}'] = r ** 2
            if abs(r) > abs(best_ld_r):
                best_ld_r = r
                best_ld_id = f'LD{ld_idx+1}'
        
        row['best_LD'] = best_ld_id
        row['best_LD_r'] = best_ld_r
        row['best_LD_r2'] = best_ld_r ** 2
        
        results.append(row)
    
    results_df = pd.DataFrame(results)
    
    # Top 10 by PCA correlation
    top_pca = results_df.nlargest(10, 'best_PC_r2')
    report.append("\n  Top 10 candidates by PCA R²:")
    report.append(f"  {'Rank':<5} {'Name':<25} {'Formula':<40} {'Best PC':<8} {'R²':<8}")
    report.append("  " + "-" * 86)
    for rank, (_, row) in enumerate(top_pca.iterrows(), 1):
        report.append(f"  {rank:<5} {row['name']:<25} {row['formula']:<40} {row['best_PC']:<8} {row['best_PC_r2']:.4f}")
    
    # Top 10 by LDA correlation
    top_lda = results_df.nlargest(10, 'best_LD_r2')
    report.append(f"\n  Top 10 candidates by LDA R²:")
    report.append(f"  {'Rank':<5} {'Name':<25} {'Formula':<40} {'Best LD':<8} {'R²':<8}")
    report.append("  " + "-" * 86)
    for rank, (_, row) in enumerate(top_lda.iterrows(), 1):
        report.append(f"  {rank:<5} {row['name']:<25} {row['formula']:<40} {row['best_LD']:<8} {row['best_LD_r2']:.4f}")
    
    return results_df


# ──────────────────────────────────────────────────────────────────
# PHASE D: Class Separability Scoring
# ──────────────────────────────────────────────────────────────────
def phase_d_separability(candidates, formulas, labels, results_df, report):
    """ANOVA F-statistic and Fisher Discriminant Ratio for each candidate."""
    report.append("\n" + "=" * 70)
    report.append("PHASE D: CLASS SEPARABILITY SCORING")
    report.append("=" * 70)
    
    unique_classes = np.unique(labels)
    
    f_stats = []
    fdr_scores = []
    
    for key, vals in candidates.items():
        # ANOVA F-statistic
        groups = [vals[labels == c] for c in unique_classes]
        try:
            f_stat, p_val = stats.f_oneway(*groups)
        except Exception:
            f_stat, p_val = 0, 1
        
        # Fisher Discriminant Ratio: between-class variance / within-class variance
        overall_mean = np.mean(vals)
        between_var = sum(len(g) * (np.mean(g) - overall_mean) ** 2 for g in groups) / len(unique_classes)
        within_var = sum(np.var(g) * len(g) for g in groups) / len(vals)
        fdr = between_var / (within_var + EPSILON)
        
        f_stats.append({'name': key, 'f_stat': f_stat, 'p_value': p_val, 'fdr': fdr})
    
    sep_df = pd.DataFrame(f_stats)
    
    # Merge with correlation results
    full_df = results_df.merge(sep_df, on='name')
    
    # Top 20 by ANOVA F-statistic
    top_f = full_df.nlargest(20, 'f_stat')
    report.append("\n  Top 20 candidates by ANOVA F-statistic (class separability):")
    report.append(f"  {'Rank':<5} {'Name':<25} {'Formula':<40} {'F-stat':<12} {'FDR':<10} {'PC R²':<8} {'LD R²':<8}")
    report.append("  " + "-" * 108)
    for rank, (_, row) in enumerate(top_f.iterrows(), 1):
        report.append(
            f"  {rank:<5} {row['name']:<25} {row['formula']:<40} "
            f"{row['f_stat']:<12.1f} {row['fdr']:<10.4f} "
            f"{row['best_PC_r2']:<8.4f} {row['best_LD_r2']:<8.4f}"
        )
    
    return full_df


# ──────────────────────────────────────────────────────────────────
# PHASE E: Comparison Against Known Indices
# ──────────────────────────────────────────────────────────────────
def phase_e_comparison(X, feat_cols, band_indices, band_names, labels, full_df, report):
    """Computes known spectral indices and compares against top candidates."""
    report.append("\n" + "=" * 70)
    report.append("PHASE E: COMPARISON WITH KNOWN SPECTRAL INDICES")
    report.append("=" * 70)
    
    X_bands = X[:, band_indices]
    band_lookup = {name: X_bands[:, i] for i, name in enumerate(band_names)}
    
    known_results = []
    for idx_name, spec in KNOWN_INDICES.items():
        if spec is None:
            continue  # Skip multi-band indices without simple lambda
        
        band_a, band_b, func = spec
        if band_a in band_lookup and band_b in band_lookup:
            vals = func(band_lookup[band_a], band_lookup[band_b])
            
            # ANOVA
            unique_classes = np.unique(labels)
            groups = [vals[labels == c] for c in unique_classes]
            try:
                f_stat, _ = stats.f_oneway(*groups)
            except:
                f_stat = 0
            
            known_results.append({
                'name': idx_name, 
                'f_stat': f_stat,
                'bands': f'{band_a}, {band_b}'
            })
    
    report.append(f"\n  Known indices — ANOVA F-statistics:")
    report.append(f"  {'Index':<10} {'Bands':<20} {'F-stat':<12}")
    report.append("  " + "-" * 42)
    for r in sorted(known_results, key=lambda x: x['f_stat'], reverse=True):
        report.append(f"  {r['name']:<10} {r['bands']:<20} {r['f_stat']:<12.1f}")
    
    # Flag novel candidates that outperform known indices
    max_known_f = max(r['f_stat'] for r in known_results) if known_results else 0
    novel = full_df[full_df['f_stat'] > max_known_f].nlargest(10, 'f_stat')
    
    if len(novel) > 0:
        report.append(f"\n  *** NOVEL FORMULAS OUTPERFORMING ALL KNOWN INDICES (F > {max_known_f:.1f}) ***")
        report.append(f"  {'Rank':<5} {'Name':<25} {'Formula':<40} {'F-stat':<12}")
        report.append("  " + "-" * 82)
        for rank, (_, row) in enumerate(novel.iterrows(), 1):
            report.append(f"  {rank:<5} {row['name']:<25} {row['formula']:<40} {row['f_stat']:<12.1f}")
    else:
        report.append(f"\n  No novel candidates outperform the best known index (F={max_known_f:.1f}).")
        report.append("  Top candidates may still be useful as complementary features.")
    
    return known_results


# ──────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────
def plot_top_candidates(X, feat_cols, band_indices, band_names, labels, full_df):
    """Generates a visual summary of the top discovered indices."""
    X_bands = X[:, band_indices]
    band_lookup = {name: X_bands[:, i] for i, name in enumerate(band_names)}
    
    top_10 = full_df.nlargest(10, 'f_stat')
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    
    for ax, (_, row) in zip(axes.flatten(), top_10.iterrows()):
        name = row['name']
        # Reconstruct the values from the formula tokens
        # For plotting, we store the precomputed values
        vals = np.zeros(len(labels))  # Placeholder — recompute below
        
        # Parse the formula to compute values
        formula = row['formula']
        # Simple reconstruction for ND, SR, DI, PR types
        try:
            if name.startswith('ND('):
                parts = name[3:-1].split(',')
                a, b = band_lookup[parts[0]], band_lookup[parts[1]]
                vals = (a - b) / (a + b + EPSILON)
            elif name.startswith('SR('):
                parts = name[3:-1].split(',')
                a, b = band_lookup[parts[0]], band_lookup[parts[1]]
                vals = a / (b + EPSILON)
            elif name.startswith('DI('):
                parts = name[3:-1].split(',')
                a, b = band_lookup[parts[0]], band_lookup[parts[1]]
                vals = a - b
            elif name.startswith('PR('):
                parts = name[3:-1].split(',')
                a, b = band_lookup[parts[0]], band_lookup[parts[1]]
                vals = (a * b) / (a + b + EPSILON)
            elif name.startswith('T3(') or name.startswith('T3b('):
                prefix = 'T3(' if name.startswith('T3(') else 'T3b('
                parts = name[len(prefix):-1].split(',')
                a, b, c = band_lookup[parts[0]], band_lookup[parts[1]], band_lookup[parts[2]]
                if prefix == 'T3(':
                    vals = (a - b) / (a + c + EPSILON)
                else:
                    vals = (a - b) / (b + c + EPSILON)
        except (KeyError, IndexError):
            continue
        
        for cls, color in zip(unique_labels, colors):
            mask = labels == cls
            ax.hist(vals[mask], bins=30, alpha=0.5, color=color, label=cls, density=True)
        
        ax.set_title(f'{name}\nF={row["f_stat"]:.0f}', fontsize=9, fontweight='bold')
        ax.set_xlabel(formula, fontsize=7)
    
    axes[0, 0].legend(fontsize=6, loc='upper right')
    plt.suptitle('Top 10 Discovered Spectral Index Candidates — Distribution by LULC Class', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'top10_candidates_distributions.png'), dpi=150)
    plt.close()
    
    # ── F-stat comparison bar chart ──
    fig, ax = plt.subplots(figsize=(14, 6))
    top_20 = full_df.nlargest(20, 'f_stat')
    bars = ax.barh(range(len(top_20)), top_20['f_stat'].values, color='steelblue')
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels([f"{row['name']}" for _, row in top_20.iterrows()], fontsize=8)
    ax.set_xlabel('ANOVA F-statistic (higher = better class separation)')
    ax.set_title('Top 20 Candidate Spectral Indices by Class Separability', fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, 'top20_f_stat_ranking.png'), dpi=150)
    plt.close()


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(VISUALS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("SPECTRAL INDEX DISCOVERY — Dataset 2")
    print("=" * 60)
    
    # Load data
    print("\nLoading normalized data (for PCA/LDA fitting)...")
    df_norm = pd.read_csv(NORMALIZED_CSV)
    feat_cols = [c for c in df_norm.columns if c not in META_COLS]
    X_norm = df_norm[feat_cols].values
    labels = df_norm['Level_1'].values
    
    print("Loading raw data (for spectral formula generation)...")
    df_raw = pd.read_csv(RAW_CSV)
    # Raw band columns may have same names; extract them
    raw_feat_cols = [c for c in df_raw.columns if c not in META_COLS]
    X_raw = df_raw[raw_feat_cols].values
    
    print(f"  Normalized: {X_norm.shape[0]} samples × {X_norm.shape[1]} features")
    print(f"  Raw:        {X_raw.shape[0]} samples × {len(raw_feat_cols)} features")
    print(f"  Classes: {np.unique(labels)}")
    
    report = []
    report.append("SPECTRAL INDEX DISCOVERY REPORT — Dataset 2")
    report.append("=" * 70)
    report.append(f"Normalized data: {X_norm.shape[0]} samples × {X_norm.shape[1]} features")
    report.append(f"Raw data: {X_raw.shape[0]} samples × {len(raw_feat_cols)} features")
    report.append(f"Classes: {list(np.unique(labels))}")
    report.append(f"NOTE: PCA/LDA fitted on normalized data; formulas computed on RAW band values")
    
    # Phase A — PCA/LDA on normalized data
    print("\n--- Phase A: Loading Analysis ---")
    pca, lda, X_pca, X_lda, band_idx, band_names = phase_a_loadings(X_norm, feat_cols, labels, report)
    
    # Phase B — Generate candidates on RAW band values
    print("\n--- Phase B: Generating Candidates ---")
    # Use raw_feat_cols and X_raw for formula generation
    raw_band_idx = [i for i, c in enumerate(raw_feat_cols) if c in ALL_BANDS]
    candidates, formulas = phase_b_generate_candidates(X_raw, raw_feat_cols, raw_band_idx, [raw_feat_cols[i] for i in raw_band_idx], report)
    print(f"  Total candidates: {len(candidates)}")
    
    # Phase C — Correlate raw-value candidates against normalized PCA/LDA components
    print("\n--- Phase C: Correlating with PCA/LDA ---")
    results_df = phase_c_correlations(candidates, formulas, X_pca, X_lda, report)
    
    # Phase D
    print("\n--- Phase D: Class Separability ---")
    full_df = phase_d_separability(candidates, formulas, labels, results_df, report)
    
    # Phase E — Compare using raw values
    print("\n--- Phase E: Known Index Comparison ---")
    phase_e_comparison(X_raw, raw_feat_cols, raw_band_idx, [raw_feat_cols[i] for i in raw_band_idx], labels, full_df, report)
    
    # Visualizations — using raw values
    print("\n--- Generating Visualizations ---")
    plot_top_candidates(X_raw, raw_feat_cols, raw_band_idx, [raw_feat_cols[i] for i in raw_band_idx], labels, full_df)
    
    # Save full results
    full_df.to_csv(os.path.join(REPORT_DIR, 'spectral_discovery_full_results.csv'), index=False)
    
    # Save report
    report_path = os.path.join(REPORT_DIR, 'spectral_index_discovery.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\n{'=' * 60}")
    print(f"COMPLETE")
    print(f"  Report:  {report_path}")
    print(f"  Data:    {os.path.join(REPORT_DIR, 'spectral_discovery_full_results.csv')}")
    print(f"  Visuals: {VISUALS_DIR}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
