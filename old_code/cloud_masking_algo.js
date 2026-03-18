// =================================================================================
//  3. CREATE THE PREDICTOR IMAGE
// =================================================================================
function maskL8sr(image) {
    var qa = image.select('QA_PIXEL');
    var cloudMask = (1 << 1) | (1 << 3) | (1 << 4);
    var mask = qa.bitwiseAnd(cloudMask).eq(0);
    var opticalBands = image.select('SR_B.*').multiply(0.0000275).add(-0.2);
    return image.updateMask(mask)
        .addBands(opticalBands, null, true)
        .copyProperties(image, ['CLOUD_COVER']);
}

var addQualityBand = function (image) {
    var quality = ee.Image.constant(100)
        .subtract(ee.Number(image.get('CLOUD_COVER')))
        .toFloat();
    return image.addBands(quality.rename('quality_score'));
};

var filters = pathRowList.map(function (pathRow) { return ee.Filter.and(ee.Filter.eq('WRS_PATH', pathRow[0]), ee.Filter.eq('WRS_ROW', pathRow[1])); });
var pathRowFilters = ee.Filter.or.apply(null, filters);
var startDate = ee.Date.fromYMD(year, month, 1);
var endDate = startDate.advance(1, 'month');

var mosaic = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filter(pathRowFilters)
    .filterDate(startDate, endDate)
    .map(maskL8sr)
    .map(addQualityBand)
    .qualityMosaic('quality_score');

var landsatBands = { 'blue': 'SR_B2', 'green': 'SR_B3', 'red': 'SR_B4', 'nir': 'SR_B5', 'swir1': 'SR_B6', 'swir2': 'SR_B7' };
var newNames = Object.keys(landsatBands);
var currentNames = newNames.map(function (key) { return landsatBands[key]; });
var mosaicRenamed = mosaic.select(currentNames, newNames);