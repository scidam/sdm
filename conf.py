# coding: utf-8


import os



DATA_PATH = './geodata'

COM_PREFIX = 'trim_wc2_30s_%s_'
DATA_EXT = '.tif'



DATA_PREFIXES = ('tmin', 'tmax', 'tavg')

DATA_PATTERNS = dict()
for pref in DATA_PREFIXES:
    DATA_PATTERNS.update({
                          pref.upper() + str(k):  {'filename':  os.path.join(DATA_PATH, pref, COM_PREFIX % pref + '{:02d}'.format(k)) + DATA_EXT } for k in range(1, 13)
                          })

BIO_PREFIX = COM_PREFIX % 'bio'
DATA_PATTERNS.update({
                       BIO_PREFIX + str(k):  {'filename':  os.path.join(DATA_PATH, 'bio', BIO_PREFIX + '{:02d}'.format(k)) + DATA_EXT } for k in range(1, 20)
                })






PREDICTOR_LOADERS = dict()
PREDICTOR_LOADERS.update({'BIO' + str(k): 'get_bio_data' for k in range(1, 20)})
PREDICTOR_LOADERS.update({'TMIN' + str(k): 'get_bio_data' for k in range(1, 13)})
PREDICTOR_LOADERS.update({'PREC' + str(k): 'get_bio_data' for k in range(1, 13)})
PREDICTOR_LOADERS.update({'TMAX' + str(k): 'get_bio_data' for k in range(1, 13)})
PREDICTOR_LOADERS.update({'TAVG' + str(k): 'get_bio_data' for k in range(1, 13)})
PREDICTOR_LOADERS.update({'WKI5': 'get_kiras_indecies'})
PREDICTOR_LOADERS.update({'WKI7': 'get_kiras_indecies'})
PREDICTOR_LOADERS.update({'WKI0': 'get_kiras_indecies'})






