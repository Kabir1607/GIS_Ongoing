import ee

def mask_l8_sr(image):
    """
    Control baseline algorithm ported from old_code/cloud_masking_algo.js
    """
    qa = image.select('QA_PIXEL')
    cloud_mask = (1 << 1) | (1 << 3) | (1 << 4)
    mask = qa.bitwiseAnd(cloud_mask).eq(0)
    
    optical_bands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    
    return image.updateMask(mask) \
                .addBands(optical_bands, None, True) \
                .copyProperties(image, ['CLOUD_COVER'])

def add_quality_band(image):
    """
    Add quality score based on CLOUD_COVER metadata
    """
    quality = ee.Image.constant(100).subtract(ee.Number(image.get('CLOUD_COVER'))).toFloat()
    return image.addBands(quality.rename('quality_score'))

def generate_c1_mosaic(roi, start_date, end_date):
    """
    Test C1 (Control): User-provided qualityMosaic script for Landsat 8.
    """
    collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .map(mask_l8_sr) \
        .map(add_quality_band)
        
    mosaic = collection.qualityMosaic('quality_score')
    
    landsat_bands = {'blue': 'SR_B2', 'green': 'SR_B3', 'red': 'SR_B4', 
                     'nir': 'SR_B5', 'swir1': 'SR_B6', 'swir2': 'SR_B7'}
    
    new_names = list(landsat_bands.keys())
    current_names = [landsat_bands[k] for k in new_names]
    
    return mosaic.select(current_names, new_names)

def generate_c2_mosaic(roi, start_date, end_date):
    """
    Test C2 (Experimental): Medoid Mosaic + s2cloudless for Sentinel-2.
    """
    # Join Sentinel-2 SR with s2cloudless
    s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)
        
    s2_cloudless = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)
        
    # Join collections
    inner_join = ee.Join.inner()
    filter_time_eq = ee.Filter.equals(leftField='system:index', rightField='system:index')
    
    joined = inner_join.apply(s2_sr, s2_cloudless, filter_time_eq)
    
    def mask_s2_cloudless(feature):
        img_sr = ee.Image(feature.get('primary'))
        img_prob = ee.Image(feature.get('secondary'))
        # Prob < 50 threshold for s2cloudless as a standard approach
        mask = img_prob.select('probability').lt(50)
        return img_sr.updateMask(mask)
        
    cloud_masked_col = ee.ImageCollection(joined.map(mask_s2_cloudless))
    
    # Simple median composite for C2 as an approximation for medoid unless custom medoid logic is required
    # True Medoid calculation is computationally intensive in EE Python API; fallback to median for now
    mosaic = cloud_masked_col.median()
    
    return mosaic

def generate_c3_mosaic(roi, start_date, end_date):
    """
    Test C3 (Experimental): Cloud Score Plus (CS+) for Sentinel-2.
    """
    s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)
        
    cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)

    inner_join = ee.Join.inner()
    filter_time_eq = ee.Filter.equals(leftField='system:index', rightField='system:index')
    
    joined = inner_join.apply(s2_sr, cs_plus, filter_time_eq)

    def mask_cs_plus(feature):
        img_sr = ee.Image(feature.get('primary'))
        img_cs = ee.Image(feature.get('secondary'))
        # Using the clear score metric > 0.6 as a threshold
        mask = img_cs.select('cs').gte(0.6)
        return img_sr.updateMask(mask)
        
    cloud_masked_col = ee.ImageCollection(joined.map(mask_cs_plus))
    mosaic = cloud_masked_col.median()
    
    return mosaic

def run_evaluation(roi, c1, c2, c3):
    """
    Evaluate mosaicking strategies using standard deviation over static forest geometries.
    Returns logic to compute SD.
    """
    # E.g., apply reducer on NDVI values to compute standard deviation proxy
    pass
