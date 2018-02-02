#coding: utf-8




import numpy as np 
import pyximport; pyximport.install()
pyximport.install(setup_args={"include_dirs":np.get_include()},
                  reload_support=True)
from extractors import get_data_by_coordinate

from conf import DATA_PATTERNS, PREDICTOR_LOADERS
from osgeo import gdal

  
def get_bio_data(lats,lons, name):
    if name not in DATA_PATTERNS:
        raise BaseException("Couldn't find the <name> name in the declared datasets")
    data = gdal.Open(DATA_PATTERNS[name]['filename'])
    print('the data is opened')
    geoinfo = data.GetGeoTransform()
    xmin = geoinfo[0]
    xres = geoinfo[1]
    ymax = geoinfo[3]
    yrot = geoinfo[4]
    xrot = geoinfo[2]
    yres = geoinfo[-1]
    xsize = data.RasterXSize
    ysize = data.RasterYSize
    band = data.GetRasterBand(1)
    nodata = np.float64(band.GetNoDataValue())
    if not np.isclose(xrot, 0) or not np.isclose(yrot, 0):
        raise BaseException("xrot and yrot should be 0")
    array = data.ReadAsArray()
    del data
    return get_data_by_coordinate(np.array(lats), np.array(lons), np.array(array), xmin, xres, ymax, yres,
                                  xsize, ysize, nodata)


def get_predictor_data(lats, lons, name='BIO1', sources = ('BIO1', )):
    '''
    Extract data for specified latitudes and longitudes; 
    '''
    
    if name not in PREDICTOR_LOADERS:
        raise BaseException("Couldn't find registered extractor function for this name")
   
    
    if PREDICTOR_LOADERS[name] in globals():
        return globals()[PREDICTOR_LOADERS[name]](lats, lons, name)
    else:
        raise BaseException("The method for computation of %s isn't defined" % name)
        
    
    
