"""
Phase 1: XGBoost A/B Experiments with Feature Group Sweep & Bayesian Tuning
===========================================================================
Phase 1A: Sweeps 8 feature group configurations (F1–F8) to find the best
          input feature scope (S2-only, L8-only, AEF-only, combinations).
Phase 1B: A/B tests noise reduction ON/OFF using the best feature group.

Each configuration uses Optuna Bayesian search (TPE) for hyperparameter tuning.
SMOTE is applied to Tree-based Agriculture in each training fold.

Usage:
    python src/dataset_3_src/phase1_xgboost_ab.py [--level level_1|level_2] [--n_trials 50]
    python src/dataset_3_src/phase1_xgboost_ab.py --experiments 1.1 1.2 1.3
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import time
import argparse
import joblib
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'dataset_3', 'processed')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models', 'dataset_3')
REPORT_DIR = os.path.join(PROJECT_DIR, 'reports', 'dataset_3')

RANDOM_STATE = 42
N_FOLDS = 5
MIN_SAMPLES_FOR_F1 = 30  # Classes with fewer test samples are excluded from Custom Macro F1
SMOTE_TARGET_CLASS_NAME = 'Tree-based Agriculture'

# Feature column definitions
S2_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
L8_BANDS = ['L8_blue', 'L8_green', 'L8_red', 'L8_nir', 'L8_swir1', 'L8_swir2']
INDEX_FEATURES = ['NDVI', 'NDWI', 'GCVI']
AEF_PREFIX = 'A'  # Columns A00–A63
# Top Cross-Sensor Indices discovered from spectral_index_discovery.py
# These are pre-computed in prepare_data.py on RAW reflectance values.
CSI_FEATURES = ['CSI_1', 'CSI_2', 'CSI_3', 'CSI_4', 'CSI_5']

# Feature group definitions (Axis 4 in testing plan)
FEATURE_GROUPS = {
    'F1_S2_Only':      {'cols': S2_BANDS},
    'F2_L8_Only':      {'cols': L8_BANDS},
    'F3_AEF_Only':     {'cols': 'AEF'},  # Detected dynamically
    'F4_S2_L8':        {'cols': S2_BANDS + L8_BANDS},
    'F5_S2_L8_Idx':    {'cols': S2_BANDS + L8_BANDS + INDEX_FEATURES},
    'F6_S2_L8_AEF':    {'cols': S2_BANDS + L8_BANDS + ['AEF']},  # + dynamic AEF
    'F7_All':          {'cols': 'ALL'},  # All feature columns
    'F8_All_CSI':      {'cols': 'ALL+CSI'},  # All + computed CSI
    'F12_L8_CSI':      {'cols': L8_BANDS + CSI_FEATURES},
}

# Experiment configurations
EXPERIMENTS = {
    # Phase 1A — Feature Group Sweep (Noise OFF)
    '1.1':  {'feature_group': 'F1_S2_Only',      'noise_reduction': False, 'name': 'F1_S2_Only'},
    '1.2':  {'feature_group': 'F2_L8_Only',      'noise_reduction': False, 'name': 'F2_L8_Only'},
    '1.3':  {'feature_group': 'F3_AEF_Only',     'noise_reduction': False, 'name': 'F3_AEF_Only'},
    '1.4':  {'feature_group': 'F4_S2_L8',        'noise_reduction': False, 'name': 'F4_S2_L8'},
    '1.5':  {'feature_group': 'F5_S2_L8_Idx',    'noise_reduction': False, 'name': 'F5_S2_L8_Idx'},
    '1.6':  {'feature_group': 'F6_S2_L8_AEF',    'noise_reduction': False, 'name': 'F6_S2_L8_AEF'},
    '1.7':  {'feature_group': 'F7_All',          'noise_reduction': False, 'name': 'F7_All'},
    '1.8':  {'feature_group': 'F8_All_CSI',      'noise_reduction': False, 'name': 'F8_All_CSI'},
    '1.12': {'feature_group': 'F12_L8_CSI',      'noise_reduction': False, 'name': 'F12_L8_CSI'},
    # Phase 1B — Noise Reduction (Best Feature Group)
    '1.9':  {'feature_group': 'BEST',            'noise_reduction': True,  'name': 'Noise_ON'},
    '1.10': {'feature_group': 'BEST',            'noise_reduction': False, 'name': 'Noise_OFF'},
}


def resolve_feature_columns(df, feature_group_name):
    """
    Given a DataFrame and a feature group key, returns the list of column names
    to keep. Also computes CSI columns on-the-fly if needed.
    """
    all_cols = list(df.columns)
    aef_cols = sorted([c for c in all_cols if c.startswith(AEF_PREFIX) and len(c) <= 3])
    EPSILON = 1e-10
    
    fg = FEATURE_GROUPS[feature_group_name]
    spec = fg['cols']
    
    if spec == 'ALL':
        return all_cols, df
    
    if spec == 'ALL+CSI':
        # ALL columns (including pre-computed CSI) 
        return list(df.columns), df
    
    if spec == 'AEF':
        return aef_cols, df[aef_cols]
    
    # Build explicit column list
    selected = []
    for item in spec:
        if item == 'AEF':
            selected.extend(aef_cols)
        elif item in all_cols:
            selected.append(item)
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in selected:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    
    return unique, df[unique]


def load_experiment_data(exp_config, level='level_1', unnormalized=False):
    """
    Loads the correct dataset variant and applies feature group selection.
    """
    noise = exp_config['noise_reduction']
    feature_group = exp_config['feature_group']
    
    # Determine augmented vs raw and normalized vs unnormalized
    augmented = '_augmented' if noise else ''
    unnorm = '_unnorm' if unnormalized else ''
    
    # Always load 'full' scope (all features), then subset by feature group
    X_train = pd.read_csv(os.path.join(DATA_DIR, f'X_train_full{unnorm}{augmented}_{level}.csv'))
    X_test = pd.read_csv(os.path.join(DATA_DIR, f'X_test_full{unnorm}_{level}.csv'))
    
    # For augmented data, the y_train may have been filtered by DBSCAN
    if noise:
        y_train = pd.read_csv(os.path.join(DATA_DIR, f'y_train_full{augmented}_{level}.csv'))['target']
    else:
        y_train = pd.read_csv(os.path.join(DATA_DIR, f'y_train_{level}.csv'))['target']
    
    y_test = pd.read_csv(os.path.join(DATA_DIR, f'y_test_{level}.csv'))['target']
    
    # Apply feature group selection (skip if 'BEST' — handled externally)
    if feature_group != 'BEST':
        _, X_train = resolve_feature_columns(X_train, feature_group)
        _, X_test = resolve_feature_columns(X_test, feature_group)
    
    return X_train, X_test, y_train, y_test


def get_class_weights(y, label_encoder):
    """Compute inverse-frequency class weights."""
    classes = np.unique(y)
    total = len(y)
    weights = {}
    for c in classes:
        count = (y == c).sum()
        weights[c] = total / (len(classes) * count)
    return weights


def apply_smote_selective(X_train, y_train, label_encoder):
    """
    Applies SMOTE only to minority classes that need it.
    Specifically targets Tree-based Agriculture.
    """
    class_counts = pd.Series(y_train).value_counts()
    
    # Find the encoded ID for Tree-based Agriculture
    try:
        target_id = list(label_encoder.classes_).index(SMOTE_TARGET_CLASS_NAME)
    except ValueError:
        print(f"    [SMOTE] '{SMOTE_TARGET_CLASS_NAME}' not found in label encoder. Skipping SMOTE.")
        return X_train, y_train
    
    target_count = class_counts.get(target_id, 0)
    if target_count < 2:
        print(f"    [SMOTE] Too few '{SMOTE_TARGET_CLASS_NAME}' samples ({target_count}). Skipping SMOTE.")
        return X_train, y_train
    
    # Set target to median count of other classes
    median_count = int(class_counts.median())
    sampling_strategy = {target_id: min(median_count, max(target_count * 3, 50))}
    
    # Only apply if we're actually increasing
    if sampling_strategy[target_id] <= target_count:
        print(f"    [SMOTE] '{SMOTE_TARGET_CLASS_NAME}' already at {target_count}. No SMOTE needed.")
        return X_train, y_train
    
    try:
        k_neighbors = min(5, target_count - 1)
        if k_neighbors < 1:
            print(f"    [SMOTE] Cannot apply — too few samples for k_neighbors.")
            return X_train, y_train
        
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE,
            k_neighbors=k_neighbors
        )
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        new_synthetic = len(X_resampled) - len(X_train)
        print(f"    [SMOTE] Generated {new_synthetic} synthetic samples for '{SMOTE_TARGET_CLASS_NAME}'")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"    [SMOTE] Failed: {e}. Continuing without SMOTE.")
        return X_train, y_train


def custom_macro_f1(y_true, y_pred, min_samples=MIN_SAMPLES_FOR_F1):
    """
    Custom Macro F1 that excludes classes with fewer than min_samples in y_true.
    """
    classes = np.unique(y_true)
    f1_scores = []
    for c in classes:
        if (y_true == c).sum() >= min_samples:
            f1_c = f1_score(y_true == c, y_pred == c, average='binary', zero_division=0)
            f1_scores.append(f1_c)
    return np.mean(f1_scores) if len(f1_scores) > 0 else 0.0


def create_objective(X_train, y_train, label_encoder, n_folds=N_FOLDS, use_gpu=False):
    """
    Returns an Optuna objective function for XGBoost hyperparameter tuning.
    Uses stratified K-Fold CV with SMOTE applied inside each fold.
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'tree_method': 'hist',
            'random_state': RANDOM_STATE,
            'eval_metric': 'mlogloss',
            'verbosity': 0,
        }
        
        if use_gpu:
            from xgboost import __version__ as xgb_version
            if xgb_version.startswith('2'):
                params['device'] = 'cuda'
            else:
                params['tree_method'] = 'gpu_hist'
        
        # Compute class weights
        class_weights = get_class_weights(y_train, label_encoder)
        sample_weights_map = {c: w for c, w in class_weights.items()}
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx].copy()
            y_fold_train = y_train.iloc[train_idx].copy()
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Apply SMOTE inside the fold
            X_fold_train, y_fold_train = apply_smote_selective(
                X_fold_train, y_fold_train, label_encoder
            )
            
            # Apply sample weights
            sample_weights = np.array([sample_weights_map.get(c, 1.0) for c in y_fold_train])
            
            model = XGBClassifier(**params)
            model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
            
            y_pred = model.predict(X_fold_val)
            score = custom_macro_f1(y_fold_val.values, y_pred)
            fold_scores.append(score)
        
        return np.mean(fold_scores)
    
    return objective


def run_experiment(exp_id, exp_config, level='level_1', n_trials=50, unnormalized=False, use_gpu=False, n_jobs=1):
    """
    Runs a single experiment: loads data, tunes hyperparameters, trains final model,
    evaluates on test set, and saves all artifacts.
    """
    norm_status = "UNNORMALIZED" if unnormalized else "NORMALIZED"
    exp_name = exp_config['name'] + ("_unnorm" if unnormalized else "_norm")
    
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT {exp_id}: {exp_name} ({norm_status})")
    print(f"  Feature Group:   {exp_config['feature_group']}")
    print(f"  Noise Reduction: {'ON' if exp_config['noise_reduction'] else 'OFF'}")
    print(f"  Level: {level}")
    print(f"{'=' * 60}")
    
    start_time = time.time()
    
    # Load data
    print(f"\n[1/5] Loading data variant...")
    X_train, X_test, y_train, y_test = load_experiment_data(exp_config, level, unnormalized)
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    
    # Load label encoder
    le = joblib.load(os.path.join(MODEL_DIR, f'label_encoder_{level}.pkl'))
    
    # Bayesian hyperparameter search
    print(f"\n[2/5] Running Bayesian hyperparameter search ({n_trials} trials, {n_jobs} jobs)...")
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    objective = create_objective(X_train, y_train, le, use_gpu=use_gpu)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=(n_jobs == 1))
    
    best_params = study.best_params
    best_cv_score = study.best_value
    print(f"  Best CV Custom Macro F1: {best_cv_score:.4f}")
    print(f"  Best params: {best_params}")
    
    # Train final model with best params on full training set
    print(f"\n[3/5] Training final model with best hyperparameters...")
    X_train_final, y_train_final = apply_smote_selective(X_train.copy(), y_train.copy(), le)
    
    class_weights = get_class_weights(y_train_final, le)
    sample_weights = np.array([class_weights.get(c, 1.0) for c in y_train_final])
    
    best_params.update({
        'tree_method': 'hist',
        'random_state': RANDOM_STATE,
        'eval_metric': 'mlogloss',
        'verbosity': 0,
    })
    
    if use_gpu:
        from xgboost import __version__ as xgb_version
        if xgb_version.startswith('2'):
            best_params['device'] = 'cuda'
        else:
            best_params['tree_method'] = 'gpu_hist'
            
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train_final, y_train_final, sample_weight=sample_weights)
    
    # Evaluate on test set
    print(f"\n[4/5] Evaluating on test set...")
    y_pred = final_model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_macro_f1 = f1_score(y_test, y_pred, average='macro')
    test_custom_f1 = custom_macro_f1(y_test.values, y_pred)
    
    print(f"  Test Accuracy:       {test_accuracy:.4f}")
    print(f"  Test Macro F1:       {test_macro_f1:.4f}")
    print(f"  Test Custom Macro F1:{test_custom_f1:.4f}")
    
    # Classification report
    target_names = le.inverse_transform(sorted(np.unique(np.concatenate([y_test.values, y_pred]))))
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    elapsed = time.time() - start_time
    
    # Save everything
    print(f"\n[5/5] Saving artifacts...")
    exp_dir = os.path.join(REPORT_DIR, f'phase1_{exp_id}_{exp_name}')
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(MODEL_DIR, f'xgboost_{exp_id}_{exp_name}_{level}.pkl')
    joblib.dump(final_model, model_path)
    
    # Save report
    report_lines = [
        f"Experiment: {exp_id} — {exp_name}",
        f"Level: {level}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Runtime: {elapsed:.1f}s",
        f"",
        f"Configuration:",
        f"  Feature Group:   {exp_config['feature_group']}",
        f"  Noise Reduction: {'ON' if exp_config['noise_reduction'] else 'OFF'}",
        f"  Data Scale:      {'Unnormalized' if unnormalized else 'Normalized'}",
        f"  Bayesian Trials: {n_trials}",
        f"",
        f"Best Hyperparameters:",
    ]
    for k, v in best_params.items():
        report_lines.append(f"  {k}: {v}")
    
    report_lines.extend([
        f"",
        f"Results:",
        f"  Best CV Custom Macro F1: {best_cv_score:.4f}",
        f"  Test Accuracy:           {test_accuracy:.4f}",
        f"  Test Macro F1:           {test_macro_f1:.4f}",
        f"  Test Custom Macro F1:    {test_custom_f1:.4f}",
        f"",
        f"Classification Report:",
        report,
        f"",
        f"Train shape: {X_train.shape}",
        f"Test shape:  {X_test.shape}",
        f"Features:    {list(X_train.columns)}",
    ])
    
    report_path = os.path.join(exp_dir, f'report_{level}.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Save confusion matrix
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(os.path.join(exp_dir, f'confusion_matrix_{level}.csv'))
    
    # Save Optuna study results
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(exp_dir, f'bayesian_search_{level}.csv'), index=False)
    
    print(f"  Model saved to: {model_path}")
    print(f"  Report saved to: {report_path}")
    
    return {
        'exp_id': exp_id,
        'exp_name': exp_name,
        'best_cv_f1': best_cv_score,
        'test_accuracy': test_accuracy,
        'test_macro_f1': test_macro_f1,
        'test_custom_f1': test_custom_f1,
        'best_params': best_params,
        'runtime_s': elapsed,
    }


def generate_comparison_report(results, level='level_1'):
    """Generates a comparative summary across all Phase 1 experiments."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    lines = [
        "=" * 70,
        f"PHASE 1 COMPARISON REPORT — {level.upper()}",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"{'Exp':<5} {'Config':<25} {'CV F1':<10} {'Test Acc':<10} {'Test F1':<10} {'Custom F1':<10} {'Time':<8}",
        "-" * 70,
    ]
    
    best_result = None
    for r in results:
        lines.append(
            f"{r['exp_id']:<5} {r['exp_name']:<25} "
            f"{r['best_cv_f1']:<10.4f} {r['test_accuracy']:<10.4f} "
            f"{r['test_macro_f1']:<10.4f} {r['test_custom_f1']:<10.4f} "
            f"{r['runtime_s']:<8.1f}s"
        )
        if best_result is None or r['test_custom_f1'] > best_result['test_custom_f1']:
            best_result = r
    
    lines.extend([
        "-" * 70,
        "",
        f"WINNER: Experiment {best_result['exp_id']} ({best_result['exp_name']})",
        f"  Test Custom Macro F1: {best_result['test_custom_f1']:.4f}",
        "",
        "This configuration should be carried forward to Phase 2 (Ensemble).",
    ])
    
    report_path = os.path.join(REPORT_DIR, f'phase1_comparison_{level}.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nComparison report saved to: {report_path}")
    return best_result


def main():
    parser = argparse.ArgumentParser(description='Phase 1: XGBoost A/B Testing')
    parser.add_argument('--level', default='level_1', choices=['level_1', 'level_2'],
                       help='Classification level to test')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Bayesian search trials per experiment')
    parser.add_argument('--unnormalized', action='store_true',
                       help='Run tests on the unnormalized data variant')
    parser.add_argument('--gpu', action='store_true',
                       help='Enable GPU acceleration for XGBoost')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of parallel trials for Bayesian search')
    parser.add_argument('--experiments', nargs='+', default=None,
                       help='Specific experiment IDs to run (e.g., 1.1 1.2). Default: all')
    args = parser.parse_args()
    
    norm_status = "UNNORMALIZED" if args.unnormalized else "NORMALIZED"
    print("=" * 60)
    print(f"PHASE 1: XGBoost A/B BASELINE EXPERIMENTS ({norm_status})")
    print(f"Level: {args.level} | Trials: {args.n_trials}")
    print("=" * 60)
    
    # Check if data exists
    test_file = os.path.join(DATA_DIR, f"X_train_baseline{'_unnorm' if args.unnormalized else ''}_{args.level}.csv")
    if not os.path.exists(test_file):
        print(f"\nERROR: Prepared data not found at {test_file}")
        print("Please run prepare_data.py first:")
        print("  python src/dataset_3_src/prepare_data.py")
        sys.exit(1)
    
    # Run experiments
    exp_ids = args.experiments if args.experiments else sorted(EXPERIMENTS.keys())
    results = []
    
    for exp_id in exp_ids:
        if exp_id not in EXPERIMENTS:
            print(f"WARNING: Unknown experiment ID '{exp_id}'. Skipping.")
            continue
        
        result = run_experiment(exp_id, EXPERIMENTS[exp_id], args.level, args.n_trials, args.unnormalized, args.gpu, args.n_jobs)
        results.append(result)
    
    # Generate comparison
    if len(results) > 1:
        winner = generate_comparison_report(results, args.level)
        
        # Save winner config for Phase 2
        winner_path = os.path.join(REPORT_DIR, f'phase1_winner_{args.level}.json')
        with open(winner_path, 'w') as f:
            json.dump(winner, f, indent=2, default=str)
        print(f"\nWinner config saved to: {winner_path}")
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
