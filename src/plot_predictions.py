import os
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import contextily as ctx

import warnings
warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUGMENTED_CSV = os.path.join(PROJECT_DIR, 'data', 'dataset_2', 'analysis_data', 'augmented_data.csv')
MODEL_FILE = os.path.join(PROJECT_DIR, 'models', 'XGBoost_full.pkl')
MAPS_DIR = os.path.join(PROJECT_DIR, 'reports', 'predicted_maps')

os.makedirs(MAPS_DIR, exist_ok=True)

# Define color palettes for strong visual interpretation
L1_COLORS = {
    'Agriculture': '#FFFF00',         # Yellow
    'Forest': '#228B22',              # Forest Green
    'Grassland / Shrub': '#9ACD32',   # YellowGreen
    'Water': '#1E90FF',               # DodgerBlue
    'Urban / Built-up': '#DC143C',    # Crimson (Red stands out well for urban areas)
    'Barren / Landslide': '#8B4513'   # SaddleBrown
}

L2_COLORS = {
    'Wet/Valley Agriculture (Rice)': '#FFD700',       # Gold
    'Shifting Cultivation (Jhum)': '#DAA520',         # GoldenRod
    'Tree-based/Perennial Plantation': '#006400',     # DarkGreen
    'Dense Canopy': '#2E8B57',                        # SeaGreen
    'Secondary/Degraded': '#8FBC8F',                  # DarkSeaGreen
    'Bamboo': '#7FFF00',                              # Chartreuse
    'Grassland / Shrub': '#9ACD32',                   # YellowGreen
    'Water': '#1E90FF',                               # DodgerBlue
    'Urban / Built-up': '#DC143C',                    # Crimson
    'Barren / Landslide': '#8B4513'                   # SaddleBrown
}

def plot_maps():
    print(f"Loading data from {AUGMENTED_CSV}")
    df = pd.read_csv(AUGMENTED_CSV)
    
    # We need to make sure longitude and latitude exist
    if 'longitude' not in df.columns or 'latitude' not in df.columns:
        lons, lats = [], []
        for geo_str in df['.geo']:
            try:
                g = json.loads(geo_str)
                lons.append(g['coordinates'][0])
                lats.append(g['coordinates'][1])
            except:
                lons.append(np.nan)
                lats.append(np.nan)
        df['longitude'] = lons
        df['latitude'] = lats
        
    # Drop rows with missing coordinates entirely just for mapping
    df = df.dropna(subset=['longitude', 'latitude'])
        
    META_COLS = ['system:index', 'SNo', 'target_class', 'year', '.geo', 'Level_1', 'Level_2', 'longitude', 'latitude']
    all_feats = [c for c in df.columns if c not in META_COLS]
    
    X_full = df[all_feats].values
    
    print(f"Loading model: {MODEL_FILE}")
    with open(MODEL_FILE, 'rb') as f:
        clf = pickle.load(f)
        
    print("Running predictions on the entire dataset (42,000+ points)...")
    preds_encoded = clf.predict(X_full)
    
    # Recreate the LabelEncoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(df['Level_2'])
    
    preds_l2 = le.inverse_transform(preds_encoded)
    
    # Map Level 2 predictions back to Level 1
    l2_to_l1 = dict(zip(df['Level_2'], df['Level_1']))
    preds_l1 = np.array([l2_to_l1[p] for p in preds_l2])
    
    df['Pred_Level_2'] = preds_l2
    df['Pred_Level_1'] = preds_l1
    
    print("Converting to GeoPandas and re-projecting for Basemap compatibility...")
    # 1. Convert to spatial geometry based on raw WGS84 GPS coords
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    
    # 2. Re-project to Web Mercator (EPSG:3857) to align flawlessly with Contextily map tiles
    gdf = gdf.to_crs(epsg=3857)
    
    # Shuffle for plot rendering to avoid Z-order bias
    gdf_shuffled = gdf.sample(frac=1, random_state=42)
    
    print("Generating Level 1 geographic map with Contextily Basemap...")
    fig, ax = plt.subplots(figsize=(20, 16))
    
    colors_l1 = [L1_COLORS.get(val, '#000000') for val in gdf_shuffled['Pred_Level_1']]
    # Plot points onto Web Mercator coordinate space
    gdf_shuffled.plot(ax=ax, color=colors_l1, markersize=8, alpha=0.7)
    
    # Add high resolution OpenStreetMap basemap underneath
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=8, alpha=0.8)
    except Exception as e:
        print(f"Warning: Could not fetch basemap. Plotting without it. Error: {e}")
    
    ax.set_title("Arunachal Pradesh LULC - Level 1 Predictions (XGBoost)\nwith Geographic Context", fontsize=22, fontweight='bold', pad=20)
    ax.set_axis_off() # Hide coordinate axes for a clean look
    
    # Legend
    handles = [mpatches.Patch(color=color, label=label) for label, color in L1_COLORS.items() if label in gdf_shuffled['Pred_Level_1'].unique()]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, 1), fontsize=14, frameon=True, facecolor='white', title="Level 1 Classification", title_fontsize='16')
    
    plt.tight_layout()
    l1_path = os.path.join(MAPS_DIR, "predicted_map_level_1.png")
    plt.savefig(l1_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generating Level 2 geographic map with Contextily Basemap...")
    fig, ax = plt.subplots(figsize=(20, 16)) 
    
    colors_l2 = [L2_COLORS.get(val, '#FFFFFF') for val in gdf_shuffled['Pred_Level_2']]
    gdf_shuffled.plot(ax=ax, color=colors_l2, markersize=8, alpha=0.7)
    
    # Add basemap
    try:
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=8, alpha=0.8)
    except Exception as e:
        pass
        
    ax.set_title("Arunachal Pradesh LULC - Level 2 Predictions (XGBoost)\nwith Geographic Context", fontsize=22, fontweight='bold', pad=20)
    ax.set_axis_off()
    
    # Legend
    handles2 = [mpatches.Patch(color=color, label=label) for label, color in L2_COLORS.items() if label in gdf_shuffled['Pred_Level_2'].unique()]
    ax.legend(handles=handles2, loc='upper right', bbox_to_anchor=(1, 1), fontsize=14, frameon=True, facecolor='white', title="Level 2 Classification", title_fontsize='16')
    
    plt.tight_layout()
    l2_path = os.path.join(MAPS_DIR, "predicted_map_level_2.png")
    plt.savefig(l2_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Maps flawlessly overlaid with real-world terrain and saved to: {MAPS_DIR}")

if __name__ == '__main__':
    plot_maps()
