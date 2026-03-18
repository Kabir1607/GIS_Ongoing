import numpy as np

class LULCRules:
    CLASS_MAP = {
        'Alpine_Grassland': 1,
        'Wet_Paddy': 2,
        'Jhum': 3,
        'Urban': 4,
        'Water': 5,
        'Forest': 6,
        'Old_Growth_Forest': 7,
        'Bamboo': 8,
        'Scrub': 9
    }

def apply_elevation_envelopes(preds, elevation):
    """
    Applies elevation constraints.
    - Alpine Grassland: 3000-4500m. If < 3000m, fallback to Wet Paddy or Jhum Fallow (simulated here as 2/3).
    - Wet Paddy: Ceiling at 2500m. Above 2500m becomes Scrub.
    - Shifting Cultivation (Jhum): Ceiling at 3500m.
    """
    classes = LULCRules.CLASS_MAP
    
    # Alpine Grassland Constraint
    mask_alpine_low = (preds == classes['Alpine_Grassland']) & (elevation < 3000)
    preds[mask_alpine_low] = classes['Wet_Paddy'] # Simplification for fallback
    
    # Wet Paddy Constraint
    mask_paddy_high = (preds == classes['Wet_Paddy']) & (elevation > 2500)
    preds[mask_paddy_high] = classes['Scrub']
    
    # Jhum Constraint
    mask_jhum_high = (preds == classes['Jhum']) & (elevation > 3500)
    preds[mask_jhum_high] = classes['Scrub']
    
    return preds

def apply_slope_refinements(preds, slope):
    """
    Applies slope constraints.
    - Wet Paddy vs Jhum: Jhum < 5 deg reclassified to Paddy. Paddy > 20 deg reclassified to Jhum.
    - Urban Stability: Urban > 35 deg flagged/re-classified (to Bare Rock/Scrub).
    """
    classes = LULCRules.CLASS_MAP
    
    # Wet Paddy vs Jhum
    mask_jhum_flat = (preds == classes['Jhum']) & (slope < 5)
    preds[mask_jhum_flat] = classes['Wet_Paddy']
    
    mask_paddy_steep = (preds == classes['Wet_Paddy']) & (slope > 20)
    preds[mask_paddy_steep] = classes['Jhum']
    
    # Urban Stability
    mask_urban_unstable = (preds == classes['Urban']) & (slope > 35)
    preds[mask_urban_unstable] = classes['Scrub'] # Mapping bare rock to scrub for now
    
    return preds

def apply_temporal_logic(preds_t1, preds_t2):
    """
    Applies temporal constraints across two consecutive timeframes.
    - Irreversible Urban: Urban(t1) cannot become Forest or Water in t2.
    - Hydrological Stability: Water(t1) cannot jump to Jhum(t2) in one year.
    - Forest Succession: Old-Growth(t1) cannot jump to Wet Paddy(t2) without intermediate.
    """
    classes = LULCRules.CLASS_MAP
    
    # Urban Irreversibility
    mask_urban_rev = (preds_t1 == classes['Urban']) & ((preds_t2 == classes['Forest']) | (preds_t2 == classes['Water']))
    preds_t2[mask_urban_rev] = classes['Urban']
    
    # Water to Jhum
    mask_water_jump = (preds_t1 == classes['Water']) & (preds_t2 == classes['Jhum'])
    preds_t2[mask_water_jump] = classes['Water']
    
    # Old-Growth to Paddy
    mask_forest_jump = (preds_t1 == classes['Old_Growth_Forest']) & (preds_t2 == classes['Wet_Paddy'])
    preds_t2[mask_forest_jump] = classes['Old_Growth_Forest'] # Revert due to impossibility
    
    return preds_t2

def apply_proximity_rules(preds, hand, elevation, slope):
    """
    Applies rules using Distance to Stream / Height Above Nearest Drainage (HAND)
    - Shadow Correction: Water with HAND > 10m -> Forest.
    - Bamboo: Cannot be in high-altitude steep zones (> 3000m, > 30 deg).
    """
    classes = LULCRules.CLASS_MAP
    
    # Shadow Correction
    mask_shadow = (preds == classes['Water']) & (hand > 10)
    preds[mask_shadow] = classes['Forest']
    
    # Bamboo restricted
    mask_bamboo_alpine = (preds == classes['Bamboo']) & (elevation > 3000) & (slope > 30)
    preds[mask_bamboo_alpine] = classes['Scrub']
    
    return preds

def run_bfast_stability():
    """
    Placeholder for running the dense time-series R/Python BFAST algorithm to confirm breakpoints.
    Identifies "Stable Points" over 5 years.
    """
    pass

def post_process_pipeline(preds_series, elevation, slope, hand):
    """
    Main entry point for Stage 6 filtering.
    """
    # Assuming preds_series is a list of chronological prediction maps (numpy arrays)
    processed = []
    
    for i in range(len(preds_series)):
        preds = preds_series[i].copy()
        
        # Spatial Constraints
        preds = apply_elevation_envelopes(preds, elevation)
        preds = apply_slope_refinements(preds, slope)
        preds = apply_proximity_rules(preds, hand, elevation, slope)
        
        # Temporal Constraints
        if i > 0:
            preds = apply_temporal_logic(processed[i-1], preds)
            
        processed.append(preds)
        
    return processed
