import ee
import pandas as pd
import datetime
import os

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
    
    # Needs lat/lon fields, allow checking for typical column names
    lat_col = 'latitude' if 'latitude' in df.columns else 'lat'
    lon_col = 'longitude' if 'longitude' in df.columns else 'lon'
    
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Dataset must contain latitude and longitude columns. Found: {list(df.columns)}")
        
    for index, row in df.iterrows():
        try:
            date_str = str(row.get('date collected', '2025'))
            if '-' in date_str and 'T' in date_str:
                # e.g., 2025-11-29T09:30:43.987+0530
                year = int(date_str.split('-')[0])
            elif '/' in date_str:
                date_obj = datetime.datetime.strptime(date_str, "%d/%m/%Y")
                year = date_obj.year
            else:
                year = int(date_str)
        except Exception:
            year = 2025 
            
        geom = ee.Geometry.Point([float(row[lon_col]), float(row[lat_col])])
        
        properties = {
            'SNo': int(row.get('S.No', row.get('S.no.', index))),
            'year': int(year),
            'target_class': str(row.get('class', row.get('target_class', 'Unknown')))
        }
        
        feature = ee.Feature(geom, properties)
        features.append(feature)
        
    return ee.FeatureCollection(features)

def _extract_point_data(feature):
    """
    Server-side EE function to map over the Feature Collection.
    Calculates indices and AEF, excluding CHIRPS precipitation (to avoid circular arguments).
    """
    year = ee.Number(feature.get('year'))
    geom = feature.geometry()
    
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = start_date.advance(1, 'year')
    
    # 1. Base Sentinel-2/Landsat bands mapping to standard names
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geom) \
        .filterDate(start_date, end_date) \
        .median()
        
    s2_std = s2.select(
        ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    )
    
    ndvi = s2_std.normalizedDifference(['nir', 'red']).rename('NDVI')
    ndwi = s2_std.normalizedDifference(['nir', 'swir1']).rename('NDWI')
    gcvi = s2_std.expression('(NIR / GREEN) - 1', {'NIR': s2_std.select('nir'), 'GREEN': s2_std.select('green')}).rename('GCVI')
    
    img_stack = s2_std.addBands([ndvi, ndwi, gcvi])
    
    # 2. Landsat 8 Surface Reflectance mapped to standalone bands
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(geom) \
        .filterDate(start_date, end_date) \
        .median()
        
    l8_std = l8.select(
        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
        ['L8_blue', 'L8_green', 'L8_red', 'L8_nir', 'L8_swir1', 'L8_swir2']
    )
    img_stack = img_stack.addBands(l8_std)
    
    # 3. AEF V1
    aef = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL") \
        .filterBounds(geom) \
        .filterDate(start_date, end_date) \
        .mean()
        
    # Merge all layers (EXCLUDING CHIRPS)
    final_img = img_stack.addBands(aef)
    
    # Extract the values at the geometry
    reduced = final_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=10, 
        maxPixels=1e9
    )
    
    return feature.set(reduced)

def run_extraction(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    lat_col = 'latitude' if 'latitude' in df.columns else 'lat'
    lon_col = 'longitude' if 'longitude' in df.columns else 'lon'
    
    initial_len = len(df)
    df = df.dropna(subset=[lon_col, lat_col])
    if len(df) < initial_len:
        print(f"Removed {initial_len - len(df)} rows with invalid (NaN) coordinates.")
        
    print("Initializing Google Earth Engine...")
    initialize_ee()
    
    chunk_size = 2500
    total_rows = len(df)
    print(f"Total rows to process: {total_rows}. Submitting in chunks of {chunk_size}...")
    
    for i in range(0, total_rows, chunk_size):
        chunk_df = df.iloc[i:i+chunk_size]
        print(f"Preparing EE Feature Collection for chunk {i} to {i+len(chunk_df)}...")
        fc = prepare_feature_collection(chunk_df)
        
        print("Mapping extraction logic across data points (Server-side)...")
        extracted_fc = fc.map(_extract_point_data)
        
        task_name = f'Dataset3_LULC_Extraction_{i}_to_{i+len(chunk_df)}'
        
        try:
            task = ee.batch.Export.table.toDrive(
                collection=extracted_fc,
                description=task_name,
                fileNamePrefix=task_name,
                fileFormat='CSV'
            )
            task.start()
            print(f"Batch task '{task_name}' started. Check your Google Earth Engine Task Manager.")
        except Exception as e:
            print(f"Task export failed for {task_name}: {e}")

if __name__ == '__main__':
    data_file = '/home/Kdixter/Desktop/final_analysis/data/dataset_3/raw_dataset_3.csv'
    
    if os.path.exists(data_file):
        print(f"Launching Earth Engine extraction task for {data_file}...")
        run_extraction(data_file)
    else:
        print(f"Source file {data_file} not found.")
