import os
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUGMENTED_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'analysis_data', 'augmented_data.csv')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
REPORT_FILE = os.path.join(PROJECT_DIR, 'reports', 'detailed_model_performance.txt')
CM_DIR = os.path.join(PROJECT_DIR, 'reports', 'confusion_matrices')

os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)

def generate_reports():
    print("Loading data...")
    df = pd.read_csv(AUGMENTED_CSV)
    
    # Needs the exact same feature names used during training
    META_COLS = ['system:index', 'SNo', 'target_class', 'year', '.geo', 'Level_1', 'Level_2', 'longitude', 'latitude']
    all_feats = [c for c in df.columns if c not in META_COLS]
    aef_feats = [c for c in all_feats if c.startswith('A') and len(c)==3 and c[1].isdigit()]
    base_feats = [c for c in all_feats if c not in aef_feats]
    
    feature_sets = {
        'baseline': base_feats,
        'full': all_feats
    }
    
    # Recreate the Label Encoder for Level_2 exactly as in train_models.py
    le_l2 = LabelEncoder()
    y_l2 = le_l2.fit_transform(df['Level_2'])
    
    # We also need a mapping from the real string names of Level 2 -> Level 1
    l2_to_l1 = dict(zip(df['Level_2'], df['Level_1']))
    
    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y_l2)
    df_test = df.loc[idx_test].copy()
    
    true_l2_strings = df_test['Level_2'].values
    true_l1_strings = df_test['Level_1'].values

    # Determine sorting orders for axes
    l2_classes = sorted(list(df['Level_2'].unique()))
    l1_classes = sorted(list(df['Level_1'].unique()))

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DETAILED MODEL PERFORMANCE REPORT (LEVEL 1 & LEVEL 2)")
    report_lines.append("=" * 80 + "\n")
    
    # Iterate over all models in the directory
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    
    for m_file in model_files:
        print(f"Processing model: {m_file}")
        model_name_full = m_file.replace('.pkl', '')
        
        # Determine feature set
        cols_to_use = feature_sets['full'] if 'full' in m_file.lower() else feature_sets['baseline']
        
        X_test = df_test[cols_to_use].values
        
        model_path = os.path.join(MODELS_DIR, m_file)
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
            
        # 1. Level 2 Predictions
        preds_l2_encoded = clf.predict(X_test)
        preds_l2_strings = le_l2.inverse_transform(preds_l2_encoded)
        
        # 2. Level 1 Roll-up Predictions
        preds_l1_strings = np.array([l2_to_l1[p] for p in preds_l2_strings])
        
        # --- Generate Classification Reports ---
        report_lines.append(f"===============================================================================")
        report_lines.append(f"Model: {model_name_full.upper()}")
        report_lines.append(f"===============================================================================\n")
        
        report_lines.append("--- LEVEL 1 AGGREGATED CLASSIFICATION REPORT ---")
        rep_l1 = classification_report(true_l1_strings, preds_l1_strings, labels=l1_classes, digits=4)
        report_lines.append(rep_l1)
        
        report_lines.append("\n--- LEVEL 2 DIRECT CLASSIFICATION REPORT ---")
        rep_l2 = classification_report(true_l2_strings, preds_l2_strings, labels=l2_classes, digits=4)
        report_lines.append(rep_l2)
        report_lines.append("\n\n")
        
        # --- Generate Confusion Matrices ---
        # Level 1 CM
        cm_l1 = confusion_matrix(true_l1_strings, preds_l1_strings, labels=l1_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_l1, annot=True, fmt='d', cmap='Blues', xticklabels=l1_classes, yticklabels=l1_classes)
        plt.title(f'{model_name_full} - Level 1 Aggregated\nConfusion Matrix')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(CM_DIR, f"{model_name_full}_cm_level_1.png"), dpi=150)
        plt.close()
        
        # Level 2 CM
        cm_l2 = confusion_matrix(true_l2_strings, preds_l2_strings, labels=l2_classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_l2, annot=True, fmt='d', cmap='Oranges', xticklabels=l2_classes, yticklabels=l2_classes)
        plt.title(f'{model_name_full} - Level 2 Direct\nConfusion Matrix')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(CM_DIR, f"{model_name_full}_cm_level_2.png"), dpi=150)
        plt.close()

    # Save final report text
    with open(REPORT_FILE, 'w') as f:
        f.write("\n".join(report_lines))
        
    print(f"Report saved to: {REPORT_FILE}")
    print(f"Confusion matrices saved to: {CM_DIR}")

if __name__ == '__main__':
    generate_reports()
