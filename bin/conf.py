# coding: utf-8
import os
from itertools import product

LARGE_VALUE = 9000.0

DATA_PATH = './sourcegeo/geodata1.4/'

COM_PREFIX = '%s_' #'#'trim_wc2_30s_%s_'
DATA_EXT = '.tif'

#DENSITY_UNIT = 0.1 # degrees,

DATA_PREFIXES = ('tmin', 'tmax', 'tavg', 'prec', 'wind')

DATA_PATTERNS = dict()
for pref in DATA_PREFIXES:
    DATA_PATTERNS.update({
                          pref.upper() + str(k):  {'filename':  os.path.join(DATA_PATH, pref, COM_PREFIX % pref + '{:1d}'.format(k)) + DATA_EXT } for k in range(1, 13)
                          })

BIO_PREFIX = 'bio_' #trim_wc2_bio_30s
DATA_PATTERNS.update({
                       'BIO' + str(k):  {'filename':  os.path.join(DATA_PATH, 'bio', BIO_PREFIX + '{:1d}'.format(k)) + DATA_EXT } for k in range(1, 20)
                })

# ------------- Data from the future --------------------
FUTURE_POSTFIX = '.tif_tr.tif'
FUTURE_PATH_PATTERN = './sourcegeo/future/%s/%s/%s/%s/'
FUTURE_FILE_PATTERN = '%s%s%s%s%s' + FUTURE_POSTFIX
FUTURE_VARS_MAPPING = {'pr':'PREC', 'bi': 'BIO',
                       'tx':'TMAX', 'tn':'TMIN'}
FUTURE_YEARS = ['50', '70']
FUTURE_VARS = ['bi', 'tn', 'tx', 'pr']
FUTURE_PATHS = ['26', '45', '85']
FUTURE_MODELS = ['cc', 'mc']
FUTURE_MONTHS = map(str, range(1, 13))
for model, year, path, var, month  in product(FUTURE_MODELS,
                                              FUTURE_YEARS,
                                              FUTURE_PATHS,
                                              FUTURE_VARS,
                                              FUTURE_MONTHS):
    path_pat = FUTURE_PATH_PATTERN % (model, path, year, var)
    file_pat = FUTURE_FILE_PATTERN % (model, path, var, year, month)
    DATA_PATTERNS.update({
        FUTURE_VARS_MAPPING[var]+month+'_' + year + model + path: {'filename': os.path.join(path_pat, file_pat)}
                         })
    if var == 'bi':
        for m in range(13, 20):
            file_pat = FUTURE_FILE_PATTERN % (model, path, var, year, m)
            DATA_PATTERNS.update({
            FUTURE_VARS_MAPPING[var]+str(m) + '_' + year + model + path: {
                'filename': os.path.join(path_pat, file_pat)}
            })

    # append wind feature
    DATA_PATTERNS.update({
        'WIND' + month + '_' + year + model + path: {
            'filename': DATA_PATTERNS['WIND' + month]['filename']}
    })

# -------------------------------------------------------


# ------------- Data from the past --------------------
PAST_VARS_MAPPING = FUTURE_VARS_MAPPING
PAST_POSTFIX = '_.tif'
PAST_PATH_PATTERN = './sourcegeo/past/%s/%s/'
PAST_FILE_PATTERN = '%s%s%s%s' + PAST_POSTFIX

PAST_PERIOD = ['lgm', 'mid']
PAST_VARS = ['bi', 'tn', 'tx', 'pr']
PAST_MODELS = ['cc', 'mc']
PAST_MONTHS = map(str, range(1, 13))
for model, period, var, month  in product(PAST_MODELS,
                                          PAST_PERIOD,
                                          PAST_VARS,
                                          PAST_MONTHS):
    path_pat = PAST_PATH_PATTERN % (model, period)
    file_pat = PAST_FILE_PATTERN % (model, period, var, month)
    DATA_PATTERNS.update({
        PAST_VARS_MAPPING[var]+month+'_' + model + period: {'filename': os.path.join(path_pat, file_pat)}
                         })
    if var == 'bi':
        for m in range(13, 20):
            file_pat = PAST_FILE_PATTERN % (model, period, var, m)
            DATA_PATTERNS.update({
                PAST_VARS_MAPPING[var]+str(m) + '_' + model+period: {
                'filename': os.path.join(path_pat, file_pat)}
            })

# ----------------------------------------------------





# ------------ Present & future & past data predictors ------------------

PREDICTOR_LOADERS = dict()
PREDICTOR_LOADERS.update({'BIO' + str(k): 'get_bio_data' for k in range(1, 20)})
PREDICTOR_LOADERS.update({'WIND' + str(k): 'get_bio_data' for k in range(1, 13)})
PREDICTOR_LOADERS.update({'TMIN' + str(k): 'get_bio_data' for k in range(1, 13)})
PREDICTOR_LOADERS.update({'PREC' + str(k): 'get_bio_data' for k in range(1, 13)})
PREDICTOR_LOADERS.update({'TMAX' + str(k): 'get_bio_data' for k in range(1, 13)})
PREDICTOR_LOADERS.update({'TAVG' + str(k): 'get_avg_temperature' for k in range(1, 13)})
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

# -------------------------------------------------------
