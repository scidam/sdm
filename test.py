from loader import get_predictor_data, get_kiras_indecies
import numpy as np
from pylab import *



lats, lons = np.meshgrid(np.linspace(25, 65, 5000), np.linspace(100, 165, 5000))



values = get_predictor_data(lats.ravel(), lons.ravel(), 'WKI0')
figure()
contourf(lons, lats, values.reshape(5000,5000))
colorbar()
title('WKI0 heatmap')
print("Hey")
values = get_predictor_data(lats.ravel(), lons.ravel(), 'WKI5')
figure()
contourf(lons, lats, values.reshape(5000,5000))
colorbar()
title('WKI5 heatmap')
show()
