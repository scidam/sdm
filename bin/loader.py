# -*- coding:utf-8 -*-
"""
Created Date: Thursday October 31st 2019
Author: Dmitry Kislov
E-mail: kislov@easydan.com
-----
Last Modified: Friday, November 1st 2019, 9:19:02 am
Modified By: Dmitry Kislov
-----
Copyright (c) 2019
"""


import numpy as np
from functools import lru_cache
import re

from .chelsa_conf import DATA_PATTERNS, PREDICTOR_LOADERS, LARGE_VALUE
from osgeo import gdal
from osgeo import osr

def array_to_raster(array, lats, lons,  fname):
    """Array > Raster
    Save a raster from a C order array.

    :param array: ndarray
    """
    # You need to get those values like you did.

    SourceDS = gdal.Open(DATA_PATTERNS['BIO1']['filename'], gdal.GA_ReadOnly)
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    x_pixels, y_pixels = array.shape
    XPIXEL_SIZE = (lons[1] - lons[0]) / float(x_pixels)
    YPIXEL_SIZE = (lats[1] - lats[0]) / float(y_pixels)
    x_min = np.min(lons)
    y_max = np.max(lats)
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        fname,
        y_pixels,
        x_pixels,
        1,
        gdal.GDT_Float32)

    dataset.SetGeoTransform((
        x_min,    # 0
        abs(XPIXEL_SIZE),  # 1
        0,                      # 2
        y_max,    # 3
        0,                      # 4
        -abs(YPIXEL_SIZE)))
    dataset.SetProjection(Projection.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.
    return 0


def get_data_by_coordinate_np(lats, lons, array, xmin, xres, ymax, yres):
    lat_inds = ((lats - ymax) / yres).astype(np.int32)
    lon_inds = ((lons - xmin) / xres).astype(np.int32)
    array = array[lat_inds, lon_inds]
    array[np.abs(array) > LARGE_VALUE] = np.nan
    return array


@lru_cache(maxsize=200)
def get_bio_data(lats, lons, name):
    if name not in DATA_PATTERNS:
        raise BaseException("Couldn't find the <%s> name in the declared datasets" % name)
    data = gdal.Open(DATA_PATTERNS[name]['filename'])
    geoinfo = data.GetGeoTransform()
    xmin, xres, ymax, yrot, xrot, yres = (geoinfo[0], geoinfo[1], geoinfo[3],
                                          geoinfo[4], geoinfo[2], geoinfo[-1])
    if not np.isclose(xrot, 0) or not np.isclose(yrot, 0):
        raise BaseException("xrot and yrot should be 0")
    array = data.ReadAsArray()
    del data
    result = get_data_by_coordinate_np(np.array(lats, dtype=np.float64),
                                       np.array(lons, dtype=np.float64),
                                       np.array(array, dtype=np.float16),
                                       xmin, xres, ymax, yres)
    #if (('TMIN' in name) or ('TMAX' in name)) and ('_' in name):
    #    return result / 10.0
    #else:
    return result


def get_avg_temperature(lats, lons, name, postfix=''):
    if postfix:
        month = re.findall(r'(\d+)', name)[0]
        vals_min = get_bio_data(lats, lons, 'TMIN' + month + postfix)
        vals_max = get_bio_data(lats, lons, 'TMAX' + month + postfix)
        return 0.5 * (vals_min + vals_max)
    else:
        return get_bio_data(lats, lons, name)


def get_kiras_indecies(lats, lons, name, postfix=''):
    '''Retruns the Family of Kira's indecies.
    '''
    if 'WKI' in name:
        t = 10.0 * float(name.replace('WKI', ''))
    elif 'CKI' in name:
        t = -10.0 * float(name.replace('CKI', '')) 
        # needs to be checked for correctness !!!
    else:
        raise BaseException("Illegal name of Kira's index")
    result = np.zeros(np.shape(lats))
    mask = result.copy().astype(np.bool)
    if t >= 0:
        # Warm indecies
        for k in range(1, 13):
            vals = get_avg_temperature(lats, lons, name='TAVG' + str(k), postfix=postfix)
            result[vals > t] = result[vals > t] + vals[vals > t] - t
            mask += np.isnan(vals)
    else:
        # Cold indecies
        for k in range(1, 13):
            vals = get_avg_temperature(lats, lons, name='TAVG' + str(k), postfix=postfix)
            result[vals < t] = result[vals < t] + vals[vals < t] - t
            mask += np.isnan(vals)
        result = -result
    result[mask] = np.nan
    return result


def get_precipitation_kiras(lats, lons, name, postfix=''):
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
            vals = get_avg_temperature(lats, lons, name='TAVG' + str(k), postfix=postfix)
            if np.any(vals > t):
                precs = get_bio_data(lats, lons, name='PREC' + str(k) + postfix)
                result[vals > t] = result[vals > t] + precs[vals > t]
                mask += np.isnan(precs)
    else:
        for k in range(1, 13):
            vals = get_avg_temperature(lats, lons, name='TAVG' + str(k), postfix=postfix)
            if np.any(vals < t):
                precs = get_bio_data(lats, lons, name='PREC' + str(k) + postfix)
                result[vals < t] = result[vals < t] + precs[vals < t]
                mask += np.isnan(precs)
    result[mask] = np.nan
    return result


def get_extreme_temperatures(lats, lons, name, postfix=''):
    if name == 'TMINM':
        func = np.minimum
    elif name == 'TMAXM':
        func = np.maximum
    result = np.zeros(np.shape(lats))
    mask = result.copy().astype(np.bool)
    for k in range(1, 13):
        vals = get_avg_temperature(lats, lons, name='TAVG' + str(k), postfix=postfix)
        result = func(result, vals)
        mask += np.isnan(vals)
    result[mask] = np.nan
    return result

def get_IC(lats, lons, name, postfix):
    vals_min = get_extreme_temperatures(lats, lons, 'TMINM', postfix=postfix)
    vals_max = get_extreme_temperatures(lats, lons, 'TMAXM', postfix=postfix)
    return vals_max - vals_min


def get_EXTCM(lats, lons, name, postfix=''):
    if 'TMAX' in name:
        t = 'TMAX'
    else:
        t = 'TMIN'
    result = np.zeros(np.shape(lats))
    mask = result.copy().astype(np.bool)
    result = get_bio_data(lats, lons, name=t + '1' + postfix)
    vals_avg = get_avg_temperature(lats, lons, name='TAVG1', postfix=postfix)
    for k in range(2, 13):
        _ = get_avg_temperature(lats, lons, name='TAVG' + str(k), postfix=postfix)
        inds = _ < vals_avg
        vals_max = get_bio_data(lats, lons, name=t + str(k) + postfix)
        result[inds] =  vals_max[inds]
        mask += np.isnan(vals_max)
    result[mask] = np.nan
    return result


def get_IT(lats, lons, name, postfix=''):
    tmin = get_EXTCM(lats, lons, "TMINCM", postfix=postfix)
    tmax = get_EXTCM(lats, lons, 'TMAXCM', postfix=postfix)
    tavg = get_bio_data(lats, lons, 'BIO1' + postfix)
    return tmin + tmax + tavg


def get_IO(lats, lons, name, postfix=''):
    prec_warm = get_precipitation_kiras(lats, lons, 'PWKI0', postfix=postfix)
    prec_cold = get_precipitation_kiras(lats, lons, 'PCKI0', postfix=postfix)
    return prec_warm / prec_cold


def get_predictor_data(lats, lons, name='BIO1', postfix=''):
    '''
    Extract data for specified latitudes and longitudes;
    '''
    if name not in PREDICTOR_LOADERS:
        raise BaseException("Couldn't find registered extractor function for this name: %s" % name)

    if PREDICTOR_LOADERS[name] in globals():
        
        try:
            result = globals()[PREDICTOR_LOADERS[name]](lats, lons, name, postfix=postfix)
        except TypeError:
            result = globals()[PREDICTOR_LOADERS[name]](lats, lons, name+postfix)
        finally:
            return result
    else:
        raise BaseException("The method for computation of %s isn't defined" % name)
