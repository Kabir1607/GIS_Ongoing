"""
Supervised Learning & A/B Testing
=================================
Trains classification models (Random Forest, XGBoost) using two feature sets:
  Current pipeline:  Uses Augmented Data
  Set A (Baseline):  Optical bands + Indices + Precipitation + Unsupervised Features
  Set B (AEF):       Same as Baseline + AEF V1 embeddings

Target: 'Level_2' classes
Metric: Custom Macro F1 Score (Excluding 'Barren / Landslide' due to low N)

Imbalance Controls:
1. SMOTE applies exclusively to 'Forest' and 'Grassland / Shrub' classes on training split only.
2. RandomForest uses class_weight='balanced_subsample'.
3. XGBoost uses explicit sample_weights computing inverse frequencies.

Spatial Validation:
1. Extracts longitude/latitude from `.geo`
2. Runs Adversarial Validation (DAV) to check train/test spatial shift
3. Computes Mean Test Nearest-Neighbor Distance (Test -> Train)
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb

# SMOTE for data augmentation
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# ===========================
# Configuration
# ===========================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUGMENTED_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'analysis_data', 'augmented_data.csv')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
VISUALS_DIR = os.path.join(PROJECT_DIR, 'data_visuals', 'models')
REPORT_FILE = os.path.join(PROJECT_DIR, 'reports', 'model_performance_report.txt')

META_COLS = ['system:index', 'SNo', 'target_class', 'year', '.geo', 'Level_1', 'Level_2']


def extract_coordinates(df):
    """
    Parses longitude and latitude from GeoJSON string in '.geo'.
    Example format: {"type":"Point","coordinates":[93.123, 27.456]}
    """
    lons, lats = [], []
    for geo_str in df['.geo']:
        try:
            geo_dict = json.loads(geo_str)
            lon, lat = geo_dict['coordinates']
            lons.append(lon)
            lats.append(lat)
        except:
            lons.append(np.nan)
            lats.append(np.nan)
    
    df['longitude'] = lons
    df['latitude'] = lats
    return df

def run_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VISUALS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    
    report = []
    def log(msg):
        print(msg)
        report.append(msg)
        
    log("=" * 70)
    log("SUPERVISED MODELING: A/B TESTING AEF EMBEDDINGS")
    log("With SMOTE & Spatial Validation Matrix")
    log("=" * 70)
    
    # 1. Load Data
    log(f"Loading augmented data from: {AUGMENTED_CSV}")
    df = pd.read_csv(AUGMENTED_CSV)
    
    # Extract coordinates
    df = extract_coordinates(df)
    
    # Define Target
    y_raw = df['Level_2']
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    
    # Identify classes to ignore in custom Macro F1
    ignore_classes_list = ['Barren / Landslide']
    valid_class_indices = [i for i, c in enumerate(class_names) if c not in ignore_classes_list]
    valid_class_names = [c for c in class_names if c not in ignore_classes_list]
    
    # Define Feature Sets
    all_feats = [c for c in df.columns if c not in META_COLS and c not in ['longitude', 'latitude']]
    aef_feats = [c for c in all_feats if c.startswith('A') and len(c)==3 and c[1].isdigit()]
    base_feats = [c for c in all_feats if c not in aef_feats]
    
    log(f"Dataset Size: {len(df)} samples")
    log(f"Target: Level_2 ({len(class_names)} classes)")
    log(f"Ignoring class '{ignore_classes_list[0]}' in custom Macro F1 metric (N={sum(df['Level_2'] == ignore_classes_list[0])})")
    log(f"Feature Set A (Baseline): {len(base_feats)} features (Optical + Indices + Climate + Unsupervised)")
    log(f"Feature Set B (w/ AEF):   {len(all_feats)} features (+64 AEF embeddings)")
    log("-" * 70)

    # Train/test split (same random_state to ensure fair comparison)
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)
    
    y_train_orig, y_test = y[idx_train], y[idx_test]
    
    # ====================================================================
    # SPATIAL VALIDATION
    # ====================================================================
    log("\n[1/3] Running Spatial Validation (Train/Test Split Diagnostics)...")
    
    train_df = df.loc[idx_train].copy()
    test_df = df.loc[idx_test].copy()
    
    # A. Adversarial Validation (DAV)
    # Train a model to predict if a row is Train(0) or Test(1)
    train_df['ADV_LABEL'] = 0
    test_df['ADV_LABEL'] = 1
    combined_dav = pd.concat([train_df, test_df], axis=0).sample(frac=1, random_state=42)
    
    dav_X = combined_dav[base_feats]  # Check purely on baseline features
    dav_y = combined_dav['ADV_LABEL']
    
    dav_clf = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
    roc_scores = cross_val_score(dav_clf, dav_X, dav_y, cv=3, scoring='roc_auc')
    dav_auc = np.mean(roc_scores)
    
    log(f"  Adversarial Validation ROC AUC: {dav_auc:.4f}")
    if dav_auc < 0.6:
        log("    -> Split is Excellent (Test set characteristics match Train set).")
    elif dav_auc < 0.75:
        log("    -> Split is Okay (Mild covariate shift detected).")
    else:
        log("    -> WARNING: Split is Poor (Severe spatial covariate shift, test set is highly OOD).")
        
    # B. Nearest-Neighbor Spatial Distance
    # Calculate physical distance (in unprojected coordinate degrees roughly)
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(train_df[['longitude', 'latitude']])
    distances, _ = nn.kneighbors(test_df[['longitude', 'latitude']])
    mean_nn_dist = np.mean(distances)
    log(f"  Mean Test-to-Train Nearest Neighbor Distance: {mean_nn_dist:.5f} degrees")
    
    # ====================================================================
    # SMOTE TARGET DEFINITIONS
    # ====================================================================
    log("\n[2/3] Configuring SMOTE targeting limits...")
    TARGET_OVERSAMPLE = 4000 # Boost minority targets to 4000 samples for better boundary definition
    
    forest_id = le.transform(['Dense Canopy'])[0]
    grassland_id = le.transform(['Grassland / Shrub'])[0]
    
    smote_dict = {}
    train_counts = pd.Series(y_train_orig).value_counts()
    for cls_id in [forest_id, grassland_id]:
        if train_counts[cls_id] < TARGET_OVERSAMPLE:
            smote_dict[cls_id] = TARGET_OVERSAMPLE
            
    smote = SMOTE(sampling_strategy=smote_dict, random_state=42)

    # Initialize models
    rf_clf = RandomForestClassifier(n_estimators=150, max_depth=15, n_jobs=-1, random_state=42, class_weight='balanced_subsample')
    
    models = {
        'RandomForest': rf_clf,
        'XGBoost': None # initialized later to use sample_weights properly
    }
    
    feature_sets = {
        'Baseline (No AEF)': base_feats,
        'Full (With AEF)': all_feats
    }
    
    results = {}
    
    log("\n[3/3] Commencing Model Training loops...")
    for model_name in models.keys():
        results[model_name] = {}
        for set_name, cols in feature_sets.items():
            
            # Reset the train data for the specific feature set
            X_train_orig = df.loc[idx_train, cols].values
            X_test  = df.loc[idx_test, cols].values
            
            # Apply SMOTE
            log(f"\nTraining {model_name} on {set_name}...")
            X_train, y_train = smote.fit_resample(X_train_orig, y_train_orig)
            log(f"  Post-SMOTE shape: {X_train.shape} (synthesized {len(X_train) - len(X_train_orig)} instances)")
            
            # Compute Sample Weights for XGBoost on the Post-SMOTE data
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
            
            # Initialize / Setup Clf
            if model_name == 'XGBoost':
                clf = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42, eval_metric='mlogloss')
            else:
                clf = models[model_name] # Use pre-initialized RF
            
            start_t = time.time()
            if model_name == 'XGBoost':
                clf.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                clf.fit(X_train, y_train)
                
            train_t = time.time() - start_t
            
            preds = clf.predict(X_test)
            
            # Metrics
            f1_full_macro = f1_score(y_test, preds, average='macro')
            
            # Calculate Custom Macro F1 (Excluding Barren/Landslide)
            f1_per_class = f1_score(y_test, preds, average=None)
            f1_custom_macro = np.mean([f1_per_class[i] for i in valid_class_indices])
            
            clf_rep = classification_report(y_test, preds, target_names=class_names, digits=4)
            cm = confusion_matrix(y_test, preds)
            
            log(f"  Time: {train_t:.1f}s | Full Macro F1: {f1_full_macro:.4f} | Custom Macro F1 (Valid Classes): {f1_custom_macro:.4f}")
            
            # Save artifacts
            model_path = os.path.join(MODEL_DIR, f"{model_name}_{set_name.split()[0].lower()}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(clf, f)
                
            results[model_name][set_name] = {
                'f1_custom_macro': f1_custom_macro,
                'f1_full_macro': f1_full_macro,
                'rep': clf_rep,
                'time': train_t,
                'cm': cm,
                'importances': clf.feature_importances_ if hasattr(clf, 'feature_importances_') else None,
                'cols': cols
            }
            
            # Plot Confusion Matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f'{model_name} - {set_name}\nConfusion Matrix')
            plt.ylabel('True Class')
            plt.xlabel('Predicted Class')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(VISUALS_DIR, f"cm_{model_name}_{set_name.split()[0]}.png"))
            plt.close()
            
            # Plot Feature Importances (Top 25)
            if hasattr(clf, 'feature_importances_'):
                imp = pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False).head(25)
                plt.figure(figsize=(10, 8))
                
                # Color code features
                colors = []
                for c in imp.index:
                    if c.startswith('A') and len(c)==3 and c[1].isdigit():
                        colors.append('#ff7f0e') # Orange for AEF
                    elif c.startswith('KMeans') or c.startswith('LD'):
                        colors.append('#2ca02c') # Green for Unsupervised
                    else:
                        colors.append('#1f77b4') # Blue for Baseline Optical/Indices
                
                imp.plot(kind='barh', color=colors)
                plt.title(f'{model_name} - {set_name}\nTop 25 Feature Importances')
                plt.gca().invert_yaxis()
                
                import matplotlib.patches as mpatches
                aef_patch = mpatches.Patch(color='#ff7f0e', label='AEF Embedding')
                unsup_patch = mpatches.Patch(color='#2ca02c', label='Unsupervised Feature')
                base_patch = mpatches.Patch(color='#1f77b4', label='Optical/Index/Climate')
                plt.legend(handles=[base_patch, unsup_patch, aef_patch], loc='lower right')
                
                plt.tight_layout()
                plt.savefig(os.path.join(VISUALS_DIR, f"feat_imp_{model_name}_{set_name.split()[0]}.png"))
                plt.close()

    # ====================================================================
    # COMPREHENSIVE REPORT GENERATION
    # ====================================================================
    log("\n" + "=" * 70)
    log("A/B TEST SUMMARY (Custom Macro F1 Score - Excluding Barren/Landslide)")
    log("=" * 70)
    log(f"{'Model':<15} {'Baseline (No AEF)':<20} {'Full (With AEF)':<20} {'Absolute Gain'}")
    log("-" * 70)
    
    rf_base = results['RandomForest']['Baseline (No AEF)']['f1_custom_macro']
    rf_full = results['RandomForest']['Full (With AEF)']['f1_custom_macro']
    xgb_base = results['XGBoost']['Baseline (No AEF)']['f1_custom_macro']
    xgb_full = results['XGBoost']['Full (With AEF)']['f1_custom_macro']
    
    log(f"{'RandomForest':<15} {rf_base:.4f}               {rf_full:.4f}               {rf_full - rf_base:+.4f}")
    log(f"{'XGBoost':<15} {xgb_base:.4f}               {xgb_full:.4f}               {xgb_full - xgb_base:+.4f}")

    # Detailed Class-by-Class Breakdown
    log("\n" + "=" * 70)
    log("DETAILED REPORTS")
    log("=" * 70)
    
    for model_name in models.keys():
        log(f"\n--- {model_name} (Full config with AEF) ---")
        log(results[model_name]['Full (With AEF)']['rep'])

    with open(REPORT_FILE, 'w') as f:
        f.write("\n".join(report))
        
    print(f"\nFull report saved to: {REPORT_FILE}")
    print(f"Models saved into:    {MODEL_DIR}")
    print(f"Visualizations saved: {VISUALS_DIR}")


if __name__ == '__main__':
    run_models()
