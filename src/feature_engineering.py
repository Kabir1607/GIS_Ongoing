import ee
import pandas as pd

def compute_spectral_indices(image):
    """
    Computes 10 spectral indices and appends them to the input image.
    Requires input image to have standardized band names: 
    'blue', 'green', 'red', 'nir', 'swir1', 'swir2'
    """
    
    # 1. NDVI: (nir - red) / (nir + red)
    ndvi = image.normalizedDifference(['nir', 'red']).rename('NDVI')
    
    # 2. NDWI: (nir - swir1) / (nir + swir1)
    ndwi = image.normalizedDifference(['nir', 'swir1']).rename('NDWI')
    
    # 3. SAVI: 1.5 * (nir - red) / (0.5 + nir + red)
    savi = image.expression(
        '1.5 * (NIR - RED) / (0.5 + NIR + RED)',
        {
            'NIR': image.select('nir'),
            'RED': image.select('red')
        }
    ).rename('SAVI')
    
    # 4. PRI: (blue - green) / (blue + green)
    pri = image.normalizedDifference(['blue', 'green']).rename('PRI')
    
    # 5. CAI: swir2 / swir1
    cai = image.expression(
        'SWIR2 / SWIR1',
        {
            'SWIR2': image.select('swir2'),
            'SWIR1': image.select('swir1')
        }
    ).rename('CAI')
    
    # 6. EVI: 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)
    evi = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {
            'NIR': image.select('nir'),
            'RED': image.select('red'),
            'BLUE': image.select('blue')
        }
    ).rename('EVI')
    
    # 7. EVI2: 2.5 * (nir - red) / (nir + 2.4 * red + 1)
    evi2 = image.expression(
        '2.5 * (NIR - RED) / (NIR + 2.4 * RED + 1)',
        {
            'NIR': image.select('nir'),
            'RED': image.select('red')
        }
    ).rename('EVI2')
    
    # 8. HallCover: (-red * 0.017) - (nir * 0.007) - (swir2 * 0.079) + 5.22
    hallcover = image.expression(
        '(-RED * 0.017) - (NIR * 0.007) - (SWIR2 * 0.079) + 5.22',
        {
            'RED': image.select('red'),
            'NIR': image.select('nir'),
            'SWIR2': image.select('swir2')
        }
    ).rename('HallCover')
    
    # 9. HallHeigth: (-red * 0.039) - (nir * 0.011) - (swir1 * 0.026) + 4.13
    hallheigth = image.expression(
        '(-RED * 0.039) - (NIR * 0.011) - (SWIR1 * 0.026) + 4.13',
        {
            'RED': image.select('red'),
            'NIR': image.select('nir'),
            'SWIR1': image.select('swir1')
        }
    ).rename('HallHeigth')
    
    # 10. GCVI: (nir / green) - 1
    gcvi = image.expression(
        '(NIR / GREEN) - 1',
        {
            'NIR': image.select('nir'),
            'GREEN': image.select('green')
        }
    ).rename('GCVI')
    
    return image.addBands([ndvi, ndwi, savi, pri, cai, evi, evi2, hallcover, hallheigth, gcvi])


def get_precipitation(roi, year):
    """
    Extracts mean annual precipitation from CHIRPS Daily for the given year.
    """
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .select('precipitation')
        
    return chirps.mean().rename('Precipitation')


def get_aef_embeddings(roi, year):
    """
    Extracts AEF V1 Foundation Embeddings (64-D).
    """
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    aef = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL") \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)
        
    return aef.mean()


def extract_features_for_dataset(csv_path):
    """
    Main extraction pipeline dynamically syncing to the year per point.
    """
    df = pd.read_csv(csv_path)
    # Required columns: 'latitude', 'longitude', 'year'
    
    # For each row, construct EE Point geometry
    # Get standard bands + indices for that year
    # Get Precipitation for that year
    # Get AEF embeddings for that year
    # Get Gemini Embedding 2 via external API (mocked here if pure GEE)

    # Note: In an actual execution, we would batch this using ee.FeatureCollection
    # grouping by year to prevent over-fetching. 
    pass
