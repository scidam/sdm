# coding: utf-8


import os


LARGE_VALUE = 10000.0

DATA_PATH = './geodata'

COM_PREFIX = 'trim_wc2_30s_%s_'
DATA_EXT = '.tif'

DENSITY_UNIT = 0.1 # degrees,

DATA_PREFIXES = ('tmin', 'tmax', 'tavg', 'prec')

DATA_PATTERNS = dict()
for pref in DATA_PREFIXES:
    DATA_PATTERNS.update({
                          pref.upper() + str(k):  {'filename':  os.path.join(DATA_PATH, pref, COM_PREFIX % pref + '{:02d}'.format(k)) + DATA_EXT } for k in range(1, 13)
                          })

BIO_PREFIX = 'trim_wc2_bio_30s_'
DATA_PATTERNS.update({
                       'BIO' + str(k):  {'filename':  os.path.join(DATA_PATH, 'bio', BIO_PREFIX + '{:02d}'.format(k)) + DATA_EXT } for k in range(1, 20)
                })


PREDICTOR_LOADERS = dict()
PREDICTOR_LOADERS.update({'BIO' + str(k): 'get_bio_data' for k in range(1, 20)})
PREDICTOR_LOADERS.update({'TMIN' + str(k): 'get_bio_data' for k in range(1, 13)})
PREDICTOR_LOADERS.update({'PREC' + str(k): 'get_bio_data' for k in range(1, 13)})
PREDICTOR_LOADERS.update({'TMAX' + str(k): 'get_bio_data' for k in range(1, 13)})
PREDICTOR_LOADERS.update({'TAVG' + str(k): 'get_bio_data' for k in range(1, 13)})
PREDICTOR_LOADERS.update({'WKI' + str(k): 'get_kiras_indecies' for k in range(11)})
PREDICTOR_LOADERS.update({'CKI' + str(k): 'get_kiras_indecies' for k in range(11)})
PREDICTOR_LOADERS.update({'PWKI' + str(k): 'get_precipitation_kiras' for k in range(10)})
PREDICTOR_LOADERS.update({'PCKI' + str(k): 'get_precipitation_kiras' for k in range(10)})
PREDICTOR_LOADERS.update({'TMINM': 'get_extreme_temperatures'})
PREDICTOR_LOADERS.update({'TMAXM': 'get_extreme_temperatures'})
PREDICTOR_LOADERS.update({'IC': 'get_IC'})
PREDICTOR_LOADERS.update({'TMINCM': 'get_EXTCM'})
PREDICTOR_LOADERS.update({'TMAXCM': 'get_EXTCM'})
PREDICTOR_LOADERS.update({'IT': 'get_IT'})
PREDICTOR_LOADERS.update({'IO': 'get_IO'})




