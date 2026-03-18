import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from libpysal.weights import DistanceBand
from esda.moran import Moran

def calculate_morans_i(df, coords_cols=['longitude', 'latitude'], val_col='NDVI'):
    """
    Calculates Moran's I to determine the spatial autocorrelation range (phi).
    """
    coords = df[coords_cols].values
    y = df[val_col].values
    
    # Example distance thresholds to test
    thresholds = [0.01, 0.05, 0.1, 0.5] # In coordinate degrees, or convert to km
    results = {}
    
    for dist in thresholds:
        w = DistanceBand.from_array(coords, threshold=dist, silence_warnings=True)
        # Row-standardize the weights
        w.transform = 'r'
        mi = Moran(y, w)
        results[dist] = {'I': mi.I, 'p_value': mi.p_sim}
        
    return results

def knndm_partition(df, coords_cols=['longitude', 'latitude'], k=5):
    """
    k-fold Nearest Neighbor Distance Matching (kNNDM).
    Attempts to match the distance distribution between CV folds and the test set
    to the real prediction space.
    """
    # Placeholder for kNNDM logic: 
    # Proper kNNDM requires sampling the prediction space uniformly.
    # For now, implementing standard KFold and logging the distance distributions.
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    folds = list(kf.split(df))
    
    # Calculate NNDs between test folds and training folds
    coords = df[coords_cols].values
    fold_distances = []
    
    for train_idx, test_idx in folds:
        train_coords = coords[train_idx]
        test_coords = coords[test_idx]
        
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(train_coords)
        distances, _ = nn.kneighbors(test_coords)
        fold_distances.append(np.mean(distances))
        
    return folds, fold_distances

def adversarial_validation(train_df, test_df, features):
    """
    Performs Adversarial Validation (DAV) to check if the test set is 
    out-of-distribution compared to the training set.
    """
    # Label train as 0, test as 1
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['ADV_LABEL'] = 0
    test_df['ADV_LABEL'] = 1
    
    combined = pd.concat([train_df, test_df], axis=0).sample(frac=1, random_state=42)
    
    X = combined[features]
    y = combined['ADV_LABEL']
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # Simple cross-validation
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    
    return np.mean(scores)
