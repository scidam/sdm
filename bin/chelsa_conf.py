# coding: utf-8

import os
from itertools import product

LARGE_VALUE = 9000.0  # used to indicate large value in geotiff files (NA- values)
DATA_PATH = '../chelsa'

DATA_EXT = '.tif'

#DENSITY_UNIT = 0.1 # degrees,


# ------------- Data from the present --------------------

PRESENT_DATA_PREFIXES = ('tmin', 'tmax', 'tavg', 'prec')
PRESENT_DIRS = ('CURRENT_min_temp', 'CURRENT_max_temp', 'CURRENT_mean_temp', 'CURRENT_prec')

def present_var_mapper(varname):
    if varname == 'tavg':
        return 'temp10'
    elif varname == 'prec':
        return varname
    else:
        return varname + '10'

PRESENT_FILE_TEMLATE = 'CHELSA_{pref}_{month}_{prec_flag}V1.2_land'
prec_flag = '1979-2013_'

DATA_PATTERNS = dict()
for pdir, pref in zip(PRESENT_DIRS, PRESENT_DATA_PREFIXES):
    DATA_PATTERNS.update({
                        pref.upper() + str(k): {'filename':  os.path.join(DATA_PATH, pdir,
                                                 PRESENT_FILE_TEMLATE.format(pref=present_var_mapper(pref), month=\
                                                 '{:02d}'.format(k), prec_flag='' if pref == 'prec' else prec_flag)) + DATA_EXT
                                                 } for k in range(1, 13)
                        })
# --------------------------------------------------------

# ------------- Data from the future --------------------

FUTURE_FILE_TEMPLATE1 = "CHELSA_{pref}_mon_{model}_{path}_r1i1p1_g025.nc_{month}_{year}_V1.2"
FUTURE_FILE_TEMPLATE2 = "CHELSA_{pref}_mon_{model}_{path}_r1i1p1_g025.nc_{month}_{year}"


FUTURE_DIRS = ('FUTURE_prec', 'FUTURE_tmax', 'FUTURE_tmean', 'FUTURE_tmin')
FUTURE_VARS = ('prec', 'tmax', 'tavg', 'tmin')

def future_var_mapper(varname):
    if varname == 'tavg':
        return 'tas'
    elif varname == 'tmax':
        return 'tasmax'
    elif varname == 'tmin':
        return 'tasmin'
    elif varname == 'prec':
        return 'pr'
    raise ValueError("Value not allowed: %s" % varname)

FUTURE_YEARS = ['2041-2060', '2061-2080']
FUTURE_PATHS = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
FUTURE_MODELS = ['MIROC-ESM', 'MRI-CGCM3', 'CCSM4']

FUTURE_MONTHS = list(map(str, range(1, 13)))
ind = 1

for pdir, var in zip(FUTURE_DIRS, FUTURE_VARS):
    for month in FUTURE_MONTHS:
        for model, year, path in product(FUTURE_MODELS, FUTURE_YEARS, FUTURE_PATHS):
            path_name = os.path.join(DATA_PATH, pdir,  model, year, path)
            if var != 'prec':
                FUTURE_FILE_TEMPLATE = FUTURE_FILE_TEMPLATE1
            else:
                FUTURE_FILE_TEMPLATE = FUTURE_FILE_TEMPLATE2
            file_name = FUTURE_FILE_TEMPLATE.format(model=model, year=year, path=path,
                                pref=future_var_mapper(var), month=str(month)) + DATA_EXT
            ind += 1
            DATA_PATTERNS.update({
                var.upper() + month +'_' + year + model + path: {'filename': os.path.join(path_name, file_name)}
            })
        
# -----------------------------------------------------

# ------------- Data from the past --------------------
PAST_MODELS = ['MIROC-ESM', 'MRI-CGCM3', 'CCSM4']
PAST_DIRS = ('LGM_max_temp', 'LGM_min_temp', 'LGM_mean_temp', 'LGM_prec')
PAST_VARS = ('tmax', 'tmin', 'tavg', 'prec')
PAST_FILE_TEMPLATE = "CHELSA_PMIP_{model}_{pref}_{month}_1"
PAST_MONTHS = map(str, range(1, 13))

for pdir, var in zip(PAST_DIRS, PAST_VARS):
    for model in PAST_MODELS:
        for month in PAST_MONTHS:
            path_name = os.path.join(DATA_PATH, pdir, model)
            file_pat = PAST_FILE_TEMPLATE.format(model=model, pref=var, month=month) + DATA_EXT
            DATA_PATTERNS.update({
                var.upper() + month+'_' + model: {'filename': os.path.join(path_name, file_pat)}
                                })
# ----------------------------------------------------

print("Total length: ", len(DATA_PATTERNS))
print("Checking source data files...")
for k, v in DATA_PATTERNS.items():
    if not os.path.exists(v['filename']):
        print("File for %s not found: %s" % (k, v['filename']))
       
print("All done.")


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
