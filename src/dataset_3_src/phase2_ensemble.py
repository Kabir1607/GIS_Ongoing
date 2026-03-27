"""
Phase 2: Ensemble Architecture Experiments
===========================================
Builds on the Phase 1 winner to test ensemble architectures:
  2.1  XGBoost + CNN     → Stacking with Logistic Regression
  2.2  XGBoost + CNN     → Stacking with MLP
  2.3  XGBoost + CNN + LSTM → Stacking with Logistic Regression
  2.4  XGBoost + CNN + LSTM → Stacking with MLP

Uses PyTorch for CNN and LSTM base learners.
All base learners are tuned via Optuna.

Usage:
    python src/dataset_3_src/phase2_ensemble.py [--level level_1|level_2] [--n_trials 30]
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

from phase1_xgboost_ab import EXPERIMENTS, FEATURE_GROUPS

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'dataset_3', 'processed')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models', 'dataset_3')
REPORT_DIR = os.path.join(PROJECT_DIR, 'reports', 'dataset_3')

RANDOM_STATE = 42
N_FOLDS = 5
MIN_SAMPLES_FOR_F1 = 30
SMOTE_TARGET_CLASS_NAME = 'Tree-based Agriculture'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ──────────────────────────────────────────────────────────────────
# 1D-CNN Model
# ──────────────────────────────────────────────────────────────────
class CNN1D(nn.Module):
    def __init__(self, n_features, n_classes, n_filters=64, kernel_size=3, n_layers=2, dropout=0.3):
        super().__init__()
        layers = []
        in_channels = 1
        for i in range(n_layers):
            out_channels = n_filters * (2 ** i) if i < 3 else n_filters * 4
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, n_classes)
    
    def forward(self, x):
        # x: (batch, features) → (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


# ──────────────────────────────────────────────────────────────────
# LSTM Model
# ──────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden_size=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1, hidden_size=hidden_size, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_classes)
    
    def forward(self, x):
        # x: (batch, features) → (batch, features, 1) — treat each feature as a time step
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        # Take the last hidden state
        last_hidden = lstm_out[:, -1, :]
        x = self.dropout(last_hidden)
        return self.fc(x)
    
    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)


# ──────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────
def custom_macro_f1(y_true, y_pred, min_samples=MIN_SAMPLES_FOR_F1):
    classes = np.unique(y_true)
    f1_scores = []
    for c in classes:
        if (y_true == c).sum() >= min_samples:
            f1_c = f1_score(y_true == c, y_pred == c, average='binary', zero_division=0)
            f1_scores.append(f1_c)
    return np.mean(f1_scores) if f1_scores else 0.0


def apply_smote_selective(X_train, y_train, label_encoder):
    try:
        target_id = list(label_encoder.classes_).index(SMOTE_TARGET_CLASS_NAME)
    except ValueError:
        return X_train, y_train
    
    class_counts = pd.Series(y_train).value_counts()
    target_count = class_counts.get(target_id, 0)
    if target_count < 2:
        return X_train, y_train
    
    median_count = int(class_counts.median())
    new_count = min(median_count, max(target_count * 3, 50))
    if new_count <= target_count:
        return X_train, y_train
    
    try:
        k = min(5, target_count - 1)
        if k < 1:
            return X_train, y_train
        smote = SMOTE(sampling_strategy={target_id: new_count}, random_state=RANDOM_STATE, k_neighbors=k)
        return smote.fit_resample(X_train, y_train)
    except Exception:
        return X_train, y_train


def get_class_weights_tensor(y, n_classes):
    """Returns a class weight tensor for PyTorch CrossEntropyLoss."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1)  # avoid division by zero
    weights = len(y) / (n_classes * counts)
    return torch.FloatTensor(weights).to(DEVICE)


def train_pytorch_model(model, X_train, y_train, n_classes, lr=1e-3, epochs=50, batch_size=64):
    """Trains a PyTorch model and returns it."""
    X_tensor = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train).to(DEVICE)
    y_tensor = torch.LongTensor(y_train.values if hasattr(y_train, 'values') else y_train).to(DEVICE)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    class_weights = get_class_weights_tensor(
        y_train.values if hasattr(y_train, 'values') else y_train, n_classes
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    model.to(DEVICE)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
    
    return model


def get_predictions_proba(model, X, scaler=None, model_type='pytorch'):
    """Gets probability predictions from a model."""
    if model_type == 'xgboost':
        return model.predict_proba(X)
    else:
        X_data = X.values if hasattr(X, 'values') else X
        if scaler is not None:
            X_data = scaler.transform(X_data)
        X_tensor = torch.FloatTensor(X_data).to(DEVICE)
        model.eval()
        with torch.no_grad():
            proba = model.predict_proba(X_tensor)
        return proba.cpu().numpy()


def load_phase1_winner(level='level_1'):
    """Loads the winning configuration from Phase 1."""
    winner_path = os.path.join(REPORT_DIR, f'phase1_winner_{level}.json')
    if os.path.exists(winner_path):
        with open(winner_path, 'r') as f:
            return json.load(f)
    else:
        print(f"WARNING: Phase 1 winner not found at {winner_path}")
        print("Using default config (Noise OFF, AEF ON)")
        return {
            'exp_id': '1.2',
            'exp_name': 'Noise_OFF__AEF_ON',
            'noise_reduction': False,
            'aef_embeddings': True,
        }


def load_winner_data(winner_config, level='level_1', unnormalized=False):
    """Loads the exact data variant and columns that won Phase 1."""
    noise = winner_config.get('noise_reduction', False)
    aef = winner_config.get('aef_embeddings', True)
    
    feature_scope = 'full' if aef else 'baseline'
    augmented = '_augmented' if noise else ''
    unnorm_str = '_unnorm' if unnormalized else ''
    
    X_train = pd.read_csv(os.path.join(DATA_DIR, f'X_train_{feature_scope}{unnorm_str}{augmented}_{level}.csv'))
    X_test = pd.read_csv(os.path.join(DATA_DIR, f'X_test_{feature_scope}{unnorm_str}_{level}.csv'))
    
    if noise:
        y_train = pd.read_csv(os.path.join(DATA_DIR, f'y_train_{feature_scope}{augmented}_{level}.csv'))['target']
    else:
        y_train = pd.read_csv(os.path.join(DATA_DIR, f'y_train_{level}.csv'))['target']
    
    y_test = pd.read_csv(os.path.join(DATA_DIR, f'y_test_{level}.csv'))['target']
    
    # Filter columns to only what the winner used
    exp_id = winner_config.get('exp_id', '1.2')
    exp_config = EXPERIMENTS.get(exp_id)
    if exp_config:
        fg_name = exp_config['feature_group']
        fg_cols = FEATURE_GROUPS[fg_name]['cols']
        if fg_cols not in ['ALL', 'ALL+CSI']:
            usable_cols = [c for c in fg_cols if c in X_train.columns]
            X_train = X_train[usable_cols]
            X_test = X_test[usable_cols]
            
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────────────────────────
# Ensemble Experiments
# ──────────────────────────────────────────────────────────────────
def tune_cnn(X_train, y_train, n_features, n_classes, n_trials=30):
    """Bayesian tuning for the 1D-CNN base learner."""
    def objective(trial):
        params = {
            'n_filters': trial.suggest_int('n_filters', 32, 128),
            'kernel_size': trial.suggest_int('kernel_size', 3, 7, step=2),
            'n_layers': trial.suggest_int('n_layers', 2, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            scaler = StandardScaler()
            X_fold_train_scaled = scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler.transform(X_fold_val)
            
            model = CNN1D(n_features, n_classes, params['n_filters'],
                         params['kernel_size'], params['n_layers'], params['dropout'])
            model = train_pytorch_model(
                model, X_fold_train_scaled, y_fold_train, n_classes,
                lr=params['lr'], epochs=30, batch_size=params['batch_size']
            )
            
            proba = get_predictions_proba(model, X_fold_val_scaled, scaler=None, model_type='pytorch')
            y_pred = np.argmax(proba, axis=1)
            scores.append(custom_macro_f1(y_fold_val.values, y_pred))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=(n_jobs == 1))
    print(f"  [CNN] Best CV F1: {study.best_value:.4f}")
    return study.best_params


def tune_lstm(X_train, y_train, n_features, n_classes, n_trials=30):
    """Bayesian tuning for the LSTM base learner."""
    def objective(trial):
        params = {
            'hidden_size': trial.suggest_int('hidden_size', 32, 256),
            'n_layers': trial.suggest_int('n_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            scaler = StandardScaler()
            X_fold_train_scaled = scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler.transform(X_fold_val)
            
            model = LSTMClassifier(n_features, n_classes, params['hidden_size'],
                                   params['n_layers'], params['dropout'])
            model = train_pytorch_model(
                model, X_fold_train_scaled, y_fold_train, n_classes,
                lr=params['lr'], epochs=30, batch_size=params['batch_size']
            )
            
            proba = get_predictions_proba(model, X_fold_val_scaled, scaler=None, model_type='pytorch')
            y_pred = np.argmax(proba, axis=1)
            scores.append(custom_macro_f1(y_fold_val.values, y_pred))
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=(n_jobs == 1))
    print(f"  [LSTM] Best CV F1: {study.best_value:.4f}")
    return study.best_params


def run_ensemble_experiment(exp_id, meta_learner_type, include_lstm, 
                            X_train, X_test, y_train, y_test, 
                            le, xgb_params, cnn_params, lstm_params=None,
                            level='level_1'):
    """
    Runs a single ensemble experiment:
    1. Trains base learners (XGBoost, CNN, optionally LSTM)
    2. Stacks their probability outputs
    3. Trains a meta-learner (LogReg or MLP)
    4. Evaluates on test set
    """
    n_features = X_train.shape[1]
    n_classes = len(le.classes_)
    
    ensemble_type = 'XGB+CNN+LSTM' if include_lstm else 'XGB+CNN'
    exp_name = f'{ensemble_type}__{meta_learner_type}'
    
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT {exp_id}: {exp_name}")
    print(f"{'=' * 60}")
    
    start_time = time.time()
    
    # Apply SMOTE
    X_train_sm, y_train_sm = apply_smote_selective(X_train.copy(), y_train.copy(), le)
    
    # ── Train XGBoost ──
    print("\n  Training XGBoost base learner...")
    xgb_model = XGBClassifier(**xgb_params, tree_method='hist', random_state=RANDOM_STATE,
                              eval_metric='mlogloss', verbosity=0)
    class_weights = {}
    for c in np.unique(y_train_sm):
        class_weights[c] = len(y_train_sm) / (n_classes * (y_train_sm == c).sum())
    sample_weights = np.array([class_weights.get(c, 1.0) for c in y_train_sm])
    xgb_model.fit(X_train_sm, y_train_sm, sample_weight=sample_weights)
    
    xgb_train_proba = xgb_model.predict_proba(X_train_sm)
    xgb_test_proba = xgb_model.predict_proba(X_test)
    
    # ── Train CNN ──
    print("  Training CNN base learner...")
    # Scale exclusively for neural nets
    pt_scaler = StandardScaler()
    X_train_sm_scaled = pt_scaler.fit_transform(X_train_sm)
    X_test_scaled = pt_scaler.transform(X_test)
    
    cnn_model = CNN1D(n_features, n_classes, cnn_params['n_filters'],
                      cnn_params['kernel_size'], cnn_params['n_layers'], cnn_params['dropout'])
    cnn_model = train_pytorch_model(
        cnn_model, X_train_sm_scaled, y_train_sm, n_classes,
        lr=cnn_params['lr'], epochs=50, batch_size=cnn_params['batch_size']
    )
    cnn_train_proba = get_predictions_proba(cnn_model, X_train_sm_scaled, scaler=None, model_type='pytorch')
    cnn_test_proba = get_predictions_proba(cnn_model, X_test_scaled, scaler=None, model_type='pytorch')
    
    # ── Stack probabilities ──
    train_stack = np.hstack([xgb_train_proba, cnn_train_proba])
    test_stack = np.hstack([xgb_test_proba, cnn_test_proba])
    
    # ── Optionally train LSTM ──
    if include_lstm and lstm_params:
        print("  Training LSTM base learner...")
        lstm_model = LSTMClassifier(n_features, n_classes, lstm_params['hidden_size'],
                                    lstm_params['n_layers'], lstm_params['dropout'])
        lstm_model = train_pytorch_model(
            lstm_model, X_train_sm_scaled, y_train_sm, n_classes,
            lr=lstm_params['lr'], epochs=50, batch_size=lstm_params['batch_size']
        )
        lstm_train_proba = get_predictions_proba(lstm_model, X_train_sm_scaled, scaler=None, model_type='pytorch')
        lstm_test_proba = get_predictions_proba(lstm_model, X_test_scaled, scaler=None, model_type='pytorch')
        train_stack = np.hstack([train_stack, lstm_train_proba])
        test_stack = np.hstack([test_stack, lstm_test_proba])
    
    # ── Train meta-learner ──
    print(f"  Training {meta_learner_type} meta-learner...")
    y_train_np = y_train_sm.values if hasattr(y_train_sm, 'values') else y_train_sm
    
    if meta_learner_type == 'LogReg':
        meta = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, 
                                  class_weight='balanced', multi_class='multinomial')
    else:  # MLP
        meta = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500,
                           random_state=RANDOM_STATE, early_stopping=True,
                           validation_fraction=0.15)
    
    meta.fit(train_stack, y_train_np)
    
    # ── Evaluate ──
    y_pred = meta.predict(test_stack)
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    
    test_accuracy = accuracy_score(y_test_np, y_pred)
    test_macro_f1 = f1_score(y_test_np, y_pred, average='macro')
    test_custom_f1 = custom_macro_f1(y_test_np, y_pred)
    
    print(f"\n  Results:")
    print(f"    Test Accuracy:        {test_accuracy:.4f}")
    print(f"    Test Macro F1:        {test_macro_f1:.4f}")
    print(f"    Test Custom Macro F1: {test_custom_f1:.4f}")
    
    elapsed = time.time() - start_time
    
    # Save artifacts
    exp_dir = os.path.join(REPORT_DIR, f'phase2_{exp_id}_{exp_name}')
    os.makedirs(exp_dir, exist_ok=True)
    
    target_names = le.inverse_transform(sorted(np.unique(np.concatenate([y_test_np, y_pred]))))
    report = classification_report(y_test_np, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_test_np, y_pred)
    
    report_lines = [
        f"Experiment: {exp_id} — {exp_name}", f"Level: {level}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Runtime: {elapsed:.1f}s", "",
        f"Ensemble: {ensemble_type}", f"Meta-learner: {meta_learner_type}", "",
        f"Results:", f"  Test Accuracy:        {test_accuracy:.4f}",
        f"  Test Macro F1:        {test_macro_f1:.4f}",
        f"  Test Custom Macro F1: {test_custom_f1:.4f}",
        "", "Classification Report:", report,
    ]
    
    with open(os.path.join(exp_dir, f'report_{level}.txt'), 'w') as f:
        f.write('\n'.join(report_lines))
    
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    cm_df.to_csv(os.path.join(exp_dir, f'confusion_matrix_{level}.csv'))
    
    # Save models
    joblib.dump(xgb_model, os.path.join(exp_dir, f'xgboost_{level}.pkl'))
    joblib.dump(meta, os.path.join(exp_dir, f'meta_{meta_learner_type}_{level}.pkl'))
    torch.save(cnn_model.state_dict(), os.path.join(exp_dir, f'cnn_{level}.pt'))
    if include_lstm and lstm_params:
        torch.save(lstm_model.state_dict(), os.path.join(exp_dir, f'lstm_{level}.pt'))
    
    return {
        'exp_id': exp_id, 'exp_name': exp_name,
        'test_accuracy': test_accuracy, 'test_macro_f1': test_macro_f1,
        'test_custom_f1': test_custom_f1, 'runtime_s': elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Ensemble Architecture Testing')
    parser.add_argument('--level', default='level_1', choices=['level_1', 'level_2'])
    parser.add_argument('--n_trials', type=int, default=30, help='Bayesian trials for CNN/LSTM tuning')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of optuna search jobs')
    parser.add_argument('--unnormalized', action='store_true', help='Use unnormalized phase 1 winner data')
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"PHASE 2: ENSEMBLE ARCHITECTURE EXPERIMENTS")
    print(f"Level: {args.level} | CNN/LSTM Trials: {args.n_trials} | Jobs: {args.n_jobs} | Unnormalized: {args.unnormalized}")
    print("=" * 60)
    
    # Load Phase 1 winner config and data
    winner_config = load_phase1_winner(args.level)
    X_train, X_test, y_train, y_test = load_winner_data(winner_config, args.level, unnormalized=args.unnormalized)
    le = joblib.load(os.path.join(MODEL_DIR, f'label_encoder_{args.level}.pkl'))
    
    n_features = X_train.shape[1]
    n_classes = len(le.classes_)
    
    print(f"\nData: Train={X_train.shape}, Test={X_test.shape}, Classes={n_classes}")
    
    # Load best XGBoost params from Phase 1
    xgb_params = winner_config.get('best_params', {})
    # Remove non-XGB keys
    for key in ['tree_method', 'random_state', 'eval_metric', 'verbosity']:
        xgb_params.pop(key, None)
    
    # Tune CNN
    print(f"\n{'=' * 40}")
    print(f"Tuning CNN ({args.n_trials} trials, {args.n_jobs} jobs)...")
    cnn_params = tune_cnn(X_train, y_train, n_features, n_classes, args.n_trials, n_jobs=args.n_jobs)
    
    # Tune LSTM
    print(f"\n{'=' * 40}")
    print(f"Tuning LSTM ({args.n_trials} trials, {args.n_jobs} jobs)...")
    lstm_params = tune_lstm(X_train, y_train, n_features, n_classes, args.n_trials, n_jobs=args.n_jobs)
    
    # Run all 4 ensemble experiments
    results = []
    
    for exp_id, meta_type, use_lstm in [
        ('2.1', 'LogReg', False),
        ('2.2', 'MLP', False),
        ('2.3', 'LogReg', True),
        ('2.4', 'MLP', True),
    ]:
        result = run_ensemble_experiment(
            exp_id, meta_type, use_lstm,
            X_train, X_test, y_train, y_test,
            le, xgb_params, cnn_params, lstm_params if use_lstm else None,
            args.level
        )
        results.append(result)
    
    # Comparison report
    lines = [
        "=" * 70,
        f"PHASE 2 COMPARISON REPORT — {args.level.upper()}",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        f"{'Exp':<5} {'Config':<30} {'Test Acc':<10} {'Test F1':<10} {'Custom F1':<10} {'Time':<8}",
        "-" * 70,
    ]
    
    best = None
    for r in results:
        lines.append(
            f"{r['exp_id']:<5} {r['exp_name']:<30} "
            f"{r['test_accuracy']:<10.4f} {r['test_macro_f1']:<10.4f} "
            f"{r['test_custom_f1']:<10.4f} {r['runtime_s']:<8.1f}s"
        )
        if best is None or r['test_custom_f1'] > best['test_custom_f1']:
            best = r
    
    lines.extend([
        "-" * 70, "",
        f"WINNER: Experiment {best['exp_id']} ({best['exp_name']})",
        f"  Test Custom Macro F1: {best['test_custom_f1']:.4f}", "",
        "This configuration should be carried forward to Phase 3 (Cloud Masking).",
    ])
    
    report_path = os.path.join(REPORT_DIR, f'phase2_comparison_{args.level}.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    # Save winner for Phase 3
    winner_path = os.path.join(REPORT_DIR, f'phase2_winner_{args.level}.json')
    with open(winner_path, 'w') as f:
        json.dump(best, f, indent=2, default=str)
    
    print(f"\nComparison report: {report_path}")
    print(f"Winner config: {winner_path}")
    print("\nPHASE 2 COMPLETE")


if __name__ == '__main__':
    main()
