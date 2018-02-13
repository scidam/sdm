#coding: utf-8

import numpy as np
import os
from functools import lru_cache

# This Cython utility module have been deprecated in flavour of numpy indexing
#import pyximport; pyximport.install()
#pyximport.install(setup_args={"include_dirs":np.get_include()},
#                 reload_support=True)
#from extractors import get_data_by_coordinate

from .conf import DATA_PATTERNS, PREDICTOR_LOADERS, LARGE_VALUE
from osgeo import gdal


def get_data_by_coordinate_np(lats, lons, array, xmin, xres, ymax, yres):
    lat_inds = ((lats - ymax) / yres).astype(np.int16)
    lon_inds = ((lons - xmin) / xres).astype(np.int16)
    array = array[lat_inds, lon_inds]
    array[np.abs(array) > LARGE_VALUE] = np.nan
    return array

def get_bio_data(lats,lons, name):
    if name not in DATA_PATTERNS:
        raise BaseException("Couldn't find the <%s> name in the declared datasets" % name)
    data = gdal.Open(DATA_PATTERNS[name]['filename'])
    geoinfo = data.GetGeoTransform()
    xmin = geoinfo[0]
    xres = geoinfo[1]
    ymax = geoinfo[3]
    yrot = geoinfo[4]
    xrot = geoinfo[2]
    yres = geoinfo[-1]
    if not np.isclose(xrot, 0) or not np.isclose(yrot, 0):
        raise BaseException("xrot and yrot should be 0")
    array = data.ReadAsArray()
    del data
    return get_data_by_coordinate_np(np.array(lats, dtype=np.float64),
                                  np.array(lons, dtype=np.float64),
                                  np.array(array, dtype=np.float64),
                                  xmin, xres, ymax, yres)


def get_kiras_indecies(lats, lons, name):
    '''Retruns the Family of Kira's indecies.
    '''
    if 'WKI' in name:
        t = float(name.replace('WKI', ''))
    elif 'CKI' in name:
        t = -float(name.replace('CKI', ''))
    else:
        raise BaseException("Illegal name of Kira's index")
    result = np.zeros(np.shape(lats))
    mask = result.copy().astype(np.bool)
    if t >= 0:
        # Warm indecies
        for k in range(1, 13):
            vals = get_bio_data(lats, lons, name='TAVG' + str(k))
            result[vals > t] = result[vals > t] + vals[vals > t] - t
            mask += np.isnan(vals)
    else:
        # Cold indecies
        for k in range(1, 13):
            vals = get_bio_data(lats, lons, name='TAVG' + str(k))
            result[vals < t] = result[vals < t] + vals[vals < t] - t
            mask += np.isnan(vals)
        result = -result
    result[mask] = np.nan
    return result


def get_precipitation_kiras(lats, lons, name):
    '''Returns Kira's based indecies of precipitation amount
    '''

    if 'PWKI' in name:
        t = float(name.replace('PWKI', ''))
    elif 'PCKI' in name:
        t = -float(name.replace('PCKI', ''))
    else:
        raise BaseException("Illegal name of Kira's index")
    result = np.zeros(np.shape(lats))
    mask = result.copy().astype(np.bool)
    if 'PWKI' in name:
        for k in range(1, 13):
            vals = get_bio_data(lats, lons, name='TAVG' + str(k))
            if np.any(vals > t):
                precs = get_bio_data(lats, lons, name='PREC' + str(k))
                result[vals > t] = result[vals > t] + precs[vals > t]
                mask += np.isnan(precs)
    else:
        for k in range(1, 13):
            vals = get_bio_data(lats, lons, name='TAVG' + str(k))
            if np.any(vals < t):
                precs = get_bio_data(lats, lons, name='PREC' + str(k))
                result[vals < t] = result[vals < t] + precs[vals < t]
                mask += np.isnan(precs)
    result[mask] = np.nan
    return result


def get_extreme_temperatures(lats, lons, name):
    if name == 'TMINM':
        func = np.minimum
    elif name == 'TMAXM':
        func = np.maximum
    result = np.zeros(np.shape(lats))
    mask = result.copy().astype(np.bool)
    for k in range(1, 13):
        vals = get_bio_data(lats, lons, name='TAVG' + str(k))
        result = func(result, vals)
        mask += np.isnan(vals)
    result[mask] = np.nan
    return result

def get_IC(lats, lons, name):
    vals_min = get_extreme_temperatures(lats, lons, 'TMINM')
    vals_max = get_extreme_temperatures(lats, lons, 'TMAXM')
    return vals_max - vals_min


def get_EXTCM(lats, lons, name):
    if 'TMAX' in name:
        t = 'TMAX'
    else:
        t = 'TMIN'
    result = np.zeros(np.shape(lats))
    mask = result.copy().astype(np.bool)
    result = get_bio_data(lats, lons, name=t + '1')
    vals_avg = get_bio_data(lats, lons, name='TAVG1')
    for k in range(2, 13):
        _ = get_bio_data(lats, lons, name='TAVG' + str(k))
        inds = _ < vals_avg
        vals_max = get_bio_data(lats, lons, name=t + str(k))
        result[inds] =  vals_max[inds]
        mask += np.isnan(vals_max)
    result[mask] = np.nan
    return result


def get_IT(lats, lons, name):
    tmin = get_EXTCM(lats, lons, "TMINCM")
    tmax = get_EXTCM(lats, lons, 'TMAXCM')
    tavg = get_bio_data(lats, lons, 'BIO1')
    return tmin + tmax + tavg

def get_IO(lats, lons, name):
    prec_warm = get_precipitation_kiras(lats, lons, 'PWKI0')
    prec_cold = get_precipitation_kiras(lats, lons, 'PCKI0')
    return prec_warm / prec_cold

@lru_cache(maxsize=200)
def get_predictor_data(lats, lons, name='BIO1', sources = ('BIO1', )):
    '''
    Extract data for specified latitudes and longitudes;
    '''
    lats = np.array(lats)
    lons = np.array(lons)
    if name not in PREDICTOR_LOADERS:
        raise BaseException("Couldn't find registered extractor function for this name")

    if PREDICTOR_LOADERS[name] in globals():
        return globals()[PREDICTOR_LOADERS[name]](lats, lons, name)
    else:
        raise BaseException("The method for computation of %s isn't defined" % name)



