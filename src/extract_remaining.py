"""
Extract Remaining Data Points
=============================
Extracts ONLY the data points that were NOT successfully extracted:
  - SNo > 40001 (the failed batch 40000-42500+)
  - SNo = 18391 (dropped due to NaN — skipped, still has NaN coords)
  
Fix applied: The original script crashed because Landsat 8 collections can be
empty for some points/years, causing `Image.select('SR_B2')` on an image with
no bands. The fix wraps Landsat 8 in a conditional check using ee.Algorithms.If
to provide a fallback empty image when no Landsat 8 data exists.
"""

import ee
import pandas as pd
import datetime
import os
import glob

def initialize_ee():
    """Initializes the Earth Engine Python API."""
    try:
        ee.Initialize(project='gis-hub-464402')
        print("Earth Engine initialized successfully.")
    except Exception as e:
        print("Failed to initialize Earth Engine. Please run src/authenticate_ee.py first.")
        raise e

def prepare_feature_collection(df):
    """
    Converts a Pandas DataFrame to an ee.FeatureCollection.
    Ensures 'date collected' is parsed into a 'year' property.
    """
    features = []
    
    if 'lon' not in df.columns or 'lat' not in df.columns:
        raise ValueError("Dataset must contain 'lon' and 'lat' columns")
        
    for index, row in df.iterrows():
        try:
            date_str = str(row['date collected'])
            date_obj = datetime.datetime.strptime(date_str, "%d/%m/%Y")
            year = date_obj.year
        except:
            year = 2018 
            
        geom = ee.Geometry.Point([row['lon'], row['lat']])
        
        properties = {
            'SNo': int(row['SNo']) if 'SNo' in row else index,
            'year': int(year),
            'target_class': row['class'] if 'class' in row else 'Unknown'
        }
        
        feature = ee.Feature(geom, properties)
        features.append(feature)
        
    return ee.FeatureCollection(features)

def _extract_point_data_safe(feature):
    """
    Server-side EE function — FIXED version.
    
    Key fix: Wraps Landsat 8 in ee.Algorithms.If to handle empty collections
    that have no bands. When no Landsat 8 imagery exists for a point/year,
    we create a constant fallback image with 0 values so the export doesn't crash.
    """
    year = ee.Number(feature.get('year'))
    geom = feature.geometry()
    
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = start_date.advance(1, 'year')
    
    # 1. Sentinel-2 Harmonized
    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geom) \
        .filterDate(start_date, end_date)
    
    # Guard against empty S2 collection too
    s2_fallback = ee.Image.constant([0, 0, 0, 0, 0, 0]).rename(
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    ).toFloat()
    
    s2_median = s2_col.median()
    s2_std = s2_median.select(
        ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    )
    # Use If to check if collection has images
    s2_safe = ee.Image(ee.Algorithms.If(
        s2_col.size().gt(0),
        s2_std,
        s2_fallback
    ))
    
    # Spectral indices
    ndvi = s2_safe.normalizedDifference(['nir', 'red']).rename('NDVI')
    ndwi = s2_safe.normalizedDifference(['nir', 'swir1']).rename('NDWI')
    gcvi = s2_safe.expression(
        '(NIR / GREEN) - 1',
        {'NIR': s2_safe.select('nir'), 'GREEN': s2_safe.select('green')}
    ).rename('GCVI')
    
    img_stack = s2_safe.addBands([ndvi, ndwi, gcvi])
    
    # 2. Landsat 8 — THE FIX: guard against empty collection
    l8_col = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(geom) \
        .filterDate(start_date, end_date)
    
    # Fallback: constant image with 0s for all 6 L8 bands
    l8_fallback = ee.Image.constant([0, 0, 0, 0, 0, 0]).rename(
        ['L8_blue', 'L8_green', 'L8_red', 'L8_nir', 'L8_swir1', 'L8_swir2']
    ).toFloat()
    
    l8_median = l8_col.median()
    l8_std = l8_median.select(
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
        ['L8_blue', 'L8_green', 'L8_red', 'L8_nir', 'L8_swir1', 'L8_swir2']
    )
    
    l8_safe = ee.Image(ee.Algorithms.If(
        l8_col.size().gt(0),
        l8_std,
        l8_fallback
    ))
    
    img_stack = img_stack.addBands(l8_safe)
    
    # 3. Precipitation (CHIRPS)
    chirps_col = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterBounds(geom) \
        .filterDate(start_date, end_date) \
        .select('precipitation')
    
    chirps_fallback = ee.Image.constant(0).rename('Precipitation').toFloat()
    chirps_safe = ee.Image(ee.Algorithms.If(
        chirps_col.size().gt(0),
        chirps_col.mean().rename('Precipitation'),
        chirps_fallback
    ))
    
    # 4. AEF V1
    aef_col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL") \
        .filterBounds(geom) \
        .filterDate(start_date, end_date)
    
    # AEF has 64 bands (A00-A63), create fallback with 64 zeros
    aef_band_names = [f'A{str(i).zfill(2)}' for i in range(64)]
    aef_fallback = ee.Image.constant([0] * 64).rename(aef_band_names).toFloat()
    
    aef_safe = ee.Image(ee.Algorithms.If(
        aef_col.size().gt(0),
        aef_col.mean(),
        aef_fallback
    ))
    
    # Merge all layers
    final_img = img_stack.addBands(chirps_safe).addBands(aef_safe)
    
    # Extract the values at the geometry
    reduced = final_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=10, 
        maxPixels=1e9
    )
    
    return feature.set(reduced)


def find_remaining_snos(cleaned_csv, extracted_dir):
    """Find SNo values that have NOT been extracted yet."""
    df_clean = pd.read_csv(cleaned_csv)
    
    # Collect all SNo values from already-extracted CSVs
    extracted_snos = set()
    csv_files = glob.glob(os.path.join(extracted_dir, '*.csv'))
    for f in csv_files:
        batch = pd.read_csv(f, usecols=['SNo'])
        extracted_snos.update(batch['SNo'].values)
    
    # Find remaining rows: valid coords AND not yet extracted
    df_remaining = df_clean[
        (~df_clean['SNo'].isin(extracted_snos)) & 
        (df_clean['lon'].notna()) & 
        (df_clean['lat'].notna())
    ]
    
    print(f"Total rows in cleaned dataset: {len(df_clean)}")
    print(f"Already extracted SNo count:    {len(extracted_snos)}")
    print(f"Rows with NaN coords (skipped): {df_clean['lon'].isna().sum()}")
    print(f"Remaining to extract:           {len(df_remaining)}")
    
    return df_remaining


def run_remaining_extraction():
    """Extract only the remaining (unextracted) data points."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cleaned_csv = os.path.join(project_dir, 'data', 'cleaned_dataset_2.csv')
    extracted_dir = os.path.join(project_dir, 'data', 'dataset_2', 'dataset_downloaded')
    
    # Step 1: Find what's missing
    df_remaining = find_remaining_snos(cleaned_csv, extracted_dir)
    
    if len(df_remaining) == 0:
        print("All data points have been extracted! Nothing remaining.")
        return
    
    # Show class breakdown of what we're about to extract
    print(f"\nClass breakdown of remaining {len(df_remaining)} points:")
    class_counts = df_remaining['class'].value_counts()
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")
    
    # Step 2: Initialize EE
    print("\nInitializing Google Earth Engine...")
    initialize_ee()
    
    # Step 3: Submit in chunks (smaller chunks for safety)
    chunk_size = 2500
    total_rows = len(df_remaining)
    print(f"\nSubmitting {total_rows} points in chunks of {chunk_size}...")
    
    for i in range(0, total_rows, chunk_size):
        chunk_df = df_remaining.iloc[i:i+chunk_size]
        sno_min = chunk_df['SNo'].min()
        sno_max = chunk_df['SNo'].max()
        
        print(f"\nPreparing chunk {i} to {i+len(chunk_df)} (SNo {sno_min}-{sno_max})...")
        fc = prepare_feature_collection(chunk_df)
        
        print("Mapping SAFE extraction logic (with empty-collection guards)...")
        extracted_fc = fc.map(_extract_point_data_safe)
        
        task_name = f'LULC_Remaining_SNo_{sno_min}_to_{sno_max}'
        
        try:
            task = ee.batch.Export.table.toDrive(
                collection=extracted_fc,
                description=task_name,
                fileNamePrefix=task_name,
                fileFormat='CSV'
            )
            task.start()
            print(f"Batch task '{task_name}' started. Check your GEE Task Manager.")
        except Exception as e:
            print(f"Task export failed for {task_name}: {e}")
    
    print(f"\nDone! {total_rows} remaining points submitted across "
          f"{(total_rows + chunk_size - 1) // chunk_size} batch tasks.")
    print("Monitor progress at: https://code.earthengine.google.com/tasks")


if __name__ == '__main__':
    run_remaining_extraction()
