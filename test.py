from loader import get_predictor_data
import numpy as np
from pylab import *



lats, lons = np.meshgrid(np.linspace(25, 65, 5000), np.linspace(100, 165, 5000))


#values = get_predictor_data(lats.ravel(), lons.ravel(), 'TMAX6')

#contourf(lons, lats, values.reshape(1000,1000))
#colorbar()
#show()



import cartopy.io.shapereader as shpreader
import shapely.vectorized
from shapely.ops import unary_union
from shapely.prepared import prep



land_shp_fname = shpreader.natural_earth(resolution='50m',
                                       category='physical', name='land')

land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))


mask = shapely.vectorized.contains(land_geom, lons, lats)
print("Total number of points belonging to the Main Land:", np.sum(mask))

print("Getting the values...")    
values = get_predictor_data(lats[mask].ravel(), lons[mask].ravel(), 'TMAX6')

print('Values are obtained:', len(values))
