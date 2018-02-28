from .loader import get_predictor_data, get_kiras_indecies
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


NAME='TMIN6'
RESOLUTION = 500
lats, lons = np.meshgrid(np.linspace(25, 65, RESOLUTION),
                         np.linspace(100, 165, RESOLUTION))

valuesf = get_predictor_data(tuple(lats.ravel()), tuple(lons.ravel()),
                             NAME, postfix='_cclgm')
print("The number of negative occurences:", np.sum(valuesf<0.0))
print("The number of nan-occurences: ", np.sum(np.isnan(valuesf)))


plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
plt.contourf(lons, lats, valuesf.reshape(RESOLUTION, RESOLUTION),
         transform=ccrs.PlateCarree()
         )

plt.colorbar()
plt.title(NAME + 'past')


values = get_predictor_data(tuple(lats.ravel()), tuple(lons.ravel()),
                            NAME, postfix='')
print("The number of negative occurences:", np.sum(values<0.0))
print("The number of nan-occurences: ", np.sum(np.isnan(values)))

plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
plt.contourf(lons, lats, values.reshape(RESOLUTION, RESOLUTION),
         transform=ccrs.PlateCarree())
plt.colorbar()
plt.title(NAME + ' present')


plt.show()
