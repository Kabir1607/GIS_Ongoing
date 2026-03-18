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
    
    # Check if we have the needed columns
    if 'lon' not in df.columns or 'lat' not in df.columns:
        raise ValueError("Dataset must contain 'lon' and 'lat' columns")
        
    for index, row in df.iterrows():
        try:
            # Parse year from 'date collected', assuming dd/mm/yyyy format based on sample
            date_str = str(row['date collected'])
            date_obj = datetime.datetime.strptime(date_str, "%d/%m/%Y")
            year = date_obj.year
        except:
            # Fallback if date parsing fails (e.g. malformed or missing)
            year = 2018 
            
        geom = ee.Geometry.Point([row['lon'], row['lat']])
        
        # We store original properties to keep track of the row
        properties = {
            'SNo': int(row['SNo']) if 'SNo' in row else index,
            'year': int(year),
            'target_class': row['class'] if 'class' in row else 'Unknown'
        }
        
        feature = ee.Feature(geom, properties)
        features.append(feature)
        
    return ee.FeatureCollection(features)

def _extract_point_data(feature):
    """
    Server-side EE function to map over the Feature Collection.
    For the specific year of the point, it calculates indices, precipitation, and AEF.
    """
    year = ee.Number(feature.get('year'))
    geom = feature.geometry()
    
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = start_date.advance(1, 'year')
    
    # 1. Base Sentinel-2/Landsat bands mapping to standard names
    # Sentinel-2 Harmonized
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(geom) \
        .filterDate(start_date, end_date) \
        .median()
        
    # Standardize bands for indices
    s2_std = s2.select(
        ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    )
    
    # 2. Add spectral indices via the previously defined module equations
    # For extraction compactness, we replicate the most critical indices directly:
    ndvi = s2_std.normalizedDifference(['nir', 'red']).rename('NDVI')
    ndwi = s2_std.normalizedDifference(['nir', 'swir1']).rename('NDWI')
    gcvi = s2_std.expression('(NIR / GREEN) - 1', {'NIR': s2_std.select('nir'), 'GREEN': s2_std.select('green')}).rename('GCVI')
    
    # Combine optical properties
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
    
    # 3. Precipitation (CHIRPS)
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterBounds(geom) \
        .filterDate(start_date, end_date) \
        .select('precipitation') \
        .mean().rename('Precipitation')
        
    # 4. AEF V1
    aef = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL") \
        .filterBounds(geom) \
        .filterDate(start_date, end_date) \
        .mean()
        
    # Merge all layers
    final_img = img_stack.addBands(chirps).addBands(aef)
    
    # Extract the values at the geometry
    reduced = final_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=10, 
        maxPixels=1e9
    )
    
    return feature.set(reduced)

def run_extraction(csv_path, output_path):
    """
    Main batch extraction pipeline. Split into chunks to respect EE memory limits.
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Clean dataset of invalid geometries that cause EE JSON payload failures
    initial_len = len(df)
    df = df.dropna(subset=['lon', 'lat'])
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
        
        task_name = f'LULC_Data_Extraction_{i}_to_{i+len(chunk_df)}'
        
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
    # Local executable path check
    data_file = 'data/cleaned_dataset_2.csv'
    out_file = 'data/extracted_features.csv'
    
    if os.path.exists(data_file):
        print(f"Launching Earth Engine extraction task for {data_file}...")
        run_extraction(data_file, out_file)
    else:
        print(f"Source file {data_file} not found.")
