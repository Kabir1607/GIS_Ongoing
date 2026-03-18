import os
import joblib
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import f1_score, jaccard_score, cohen_kappa_score
from imblearn.over_sampling import KMeansSMOTE

def apply_kmeans_smote(X, y):
    smote = KMeansSMOTE(random_state=42, cluster_balance_threshold=0.1)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

class MLPMetaLearner(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLPMetaLearner, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def _run_pipeline(X_train, y_train, X_test, y_test, suffix, model_dir, report_dir):
    """
    Sub-routine to run the ensemble for a given feature set.
    """
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    evaluate_and_report(y_test, rf_preds, f"RandomForest{suffix}", f"{report_dir}/model_eval_rf{suffix}.txt")
    joblib.dump(rf, f"{model_dir}/random_forest{suffix}.pkl")
    
    # SVM
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    svm_preds = svm.predict(X_test)
    evaluate_and_report(y_test, svm_preds, f"SVM{suffix}", f"{report_dir}/model_eval_svm{suffix}.txt")
    joblib.dump(svm, f"{model_dir}/svm{suffix}.pkl")
    
    # Stacking
    estimators = [('rf', rf), ('svm', svm)]
    meta_learner = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    meta_learner.fit(X_train, y_train)
    
    meta_preds = meta_learner.predict(X_test)
    evaluate_and_report(y_test, meta_preds, f"StackingMetaLearner{suffix}", f"{report_dir}/model_eval_metalearner{suffix}.txt")
    joblib.dump(meta_learner, f"{model_dir}/meta_learner{suffix}.pkl")
    
    return meta_learner

def train_ensemble(X_train, y_train, X_test, y_test, gemini_columns=None, model_dir='models', report_dir='reports'):
    """
    Trains the base models (RF, SVM) and a stacking classifier.
    Validates performance With and Without Gemini Embedding Vectors.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    # 1. Train Baseline (No Gemini)
    print("Training Baseline Ensemble (Without Gemini Vectors)...")
    if gemini_columns is not None:
        X_train_base = X_train.drop(columns=gemini_columns, errors='ignore')
        X_test_base = X_test.drop(columns=gemini_columns, errors='ignore')
    else:
        X_train_base, X_test_base = X_train, X_test
        
    _run_pipeline(X_train_base, y_train, X_test_base, y_test, '_baseline', model_dir, report_dir)
    
    # 2. Train Experimental (With Gemini)
    if gemini_columns is not None:
        print("Training Experimental Ensemble (With Gemini Vectors)...")
        _run_pipeline(X_train, y_train, X_test, y_test, '_with_gemini', model_dir, report_dir)


def evaluate_and_report(y_true, y_pred, model_name, report_path):
    f1 = f1_score(y_true, y_pred, average='weighted')
    miou = jaccard_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    
    report_content = f"--- Evaluation Report for {model_name} ---\n"
    report_content += f"Weighted F1-Score: {f1:.4f}\n"
    report_content += f"Mean IoU: {miou:.4f}\n"
    report_content += f"Cohen's Kappa: {kappa:.4f}\n"
    
    with open(report_path, 'w') as f:
        f.write(report_content)
