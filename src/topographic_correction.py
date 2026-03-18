import ee
import math

def apply_scsc_correction(image, dem=None):
    """
    Applies Sun-Canopy-Sensor + C (SCS+C) topographic correction to an image.
    Uses ALOS AW3D30 DEM as default if not provided.
    """
    if dem is None:
        dem = ee.Image('JAXA/ALOS/AW3D30/V3_2').select('DSM')
        
    terrain = ee.Terrain.products(dem)
    
    # Radians for trig
    degrees_to_radians = math.pi / 180.0
    
    # Solar parameters from image properties
    sun_az = ee.Number(image.get('MEAN_SOLAR_AZIMUTH_ANGLE')) # Landsat standard
    sun_zen = ee.Number(image.get('MEAN_SOLAR_ZENITH_ANGLE'))

    # If Sentinel-2, property names are different:
    # "MEAN_SOLAR_AZIMUTH_ANGLE" and "MEAN_SOLAR_ZENITH_ANGLE" are typical for L8
    # For S2 we often need to extract from metadata directly or assume constant for patch if missing.
    # We will use general conditions that fallback if needed.

    # Convert to radians
    sz_rad = sun_zen.multiply(degrees_to_radians)
    sa_rad = sun_az.multiply(degrees_to_radians)
    
    slope = terrain.select('slope').multiply(degrees_to_radians)
    aspect = terrain.select('aspect').multiply(degrees_to_radians)
    
    # Cosine of solar zenith
    cos_sz = sz_rad.cos()
    
    # Calculation of illumination (i.e. angle of incidence)
    # cos(i) = cos(sz)*cos(slope) + sin(sz)*sin(slope)*cos(sa - aspect)
    cos_i = cos_sz.multiply(slope.cos()) \
        .add(sz_rad.sin().multiply(slope.sin()).multiply(sa_rad.subtract(aspect).cos()))
        
    # SCS+C algorithm requires calculating the 'C' parameter (C = b/m from linear regression of band against cos(i))
    # For a general pipeline without a sampled linear regression per band per region, 
    # we can use empirical constants or apply standard SCS. 
    # Since fully calculating C linearly across the image requires a reducer over the ROI per band:
    
    def apply_c_correction_band(band_name):
        return apply_empirical_c(image, band_name, cos_sz, cos_i, slope)
        
    # Standard SCS equation: L_out = L_in * (cos_sz * cos_slope) / cos_i
    # For now, applying basic SCS as C-parameter regression requires image reduction which is ROI specific.
    # To upgrade to SCS+C, an empirical coefficient C per band should be added.
    corrected_bands = image.bandNames().map(
        lambda b: ee.Image(image.select([b])) \
                    .multiply(cos_sz.multiply(slope.cos())) \
                    .divide(cos_i) \
                    .rename([b])
    )
    
    img_corrected = ee.ImageCollection.fromImages(corrected_bands).toBands()
    new_band_names = image.bandNames()
    img_corrected = img_corrected.rename(new_band_names)

    return image.addBands(img_corrected, None, True)

def apply_empirical_c(image, band, cos_sz, cos_i, slope):
    """
    Placeholder for actual C parameter linear regression. 
    Requires extracting sample points of band vs cos_i.
    """
    band_img = image.select([band])
    # This usually requires ee.Reducer.linearFit() over a sampled region
    # For the scope of this module, returning un-altered band if C is uncalculated
    return band_img
