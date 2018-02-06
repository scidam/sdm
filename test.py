from loader import get_predictor_data, get_kiras_indecies
import numpy as np
from pylab import *


RESOLUTION = 2000
lats, lons = np.meshgrid(np.linspace(25, 65, RESOLUTION),
                         np.linspace(100, 165, RESOLUTION))


values = get_predictor_data(lats.ravel(), lons.ravel(), 'WKI10')

print("The number of negative occurences:", np.sum(values<0.0))
print("The number of nan-occurences: ", np.sum(np.isnan(values)))

figure()
contourf(lons, lats, values.reshape(RESOLUTION, RESOLUTION))
colorbar()
title('WKI10')


show()
