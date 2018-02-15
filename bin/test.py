from .loader import get_predictor_data, get_kiras_indecies
import numpy as np
from pylab import *


RESOLUTION = 2000
lats, lons = np.meshgrid(np.linspace(25, 65, RESOLUTION),
                         np.linspace(100, 165, RESOLUTION))

valuesf = get_predictor_data(tuple(lats.ravel()), tuple(lons.ravel()),
                            'WKI5', postfix='_50cc26')
print("The number of negative occurences:", np.sum(valuesf<0.0))
print("The number of nan-occurences: ", np.sum(np.isnan(valuesf)))

figure()
contourf(lons, lats, valuesf.reshape(RESOLUTION, RESOLUTION))
colorbar()
title('WKI10 future')


values = get_predictor_data(tuple(lats.ravel()), tuple(lons.ravel()),
                            'WKI5', postfix='')
print("The number of negative occurences:", np.sum(values<0.0))
print("The number of nan-occurences: ", np.sum(np.isnan(values)))

figure()
contourf(lons, lats, values.reshape(RESOLUTION, RESOLUTION))
colorbar()
title('WKI10 present')


show()
