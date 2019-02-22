from .loader import get_predictor_data, get_kiras_indecies
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


NAME='WKI5'
RESOLUTION = 1000
lats, lons = np.meshgrid(np.linspace(25, 65, RESOLUTION),
                         np.linspace(100, 165, RESOLUTION))

valuesf = get_predictor_data(tuple(lats.ravel()), tuple(lons.ravel()),
                             NAME, postfix='')
#print("The number of negative occurences:", np.sum(valuesf<0.0))
#print("The number of nan-occurences: ", np.sum(np.isnan(valuesf)))


# VLADIVOSTOK


# names = ['TAVG%s' % k for k in range(1, 13)]

# name='TAVG7'
# for model in ['50cc26','50cc85','50cc45', '70cc26', '70cc85', '70cc45', 'cclgm', 'ccmid']:
#         print(name, model, get_predictor_data((43.137299,), (131.946541,),
#           name, postfix='_'+model))


# wef


plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
plt.contourf(lons, lats, valuesf.reshape(RESOLUTION, RESOLUTION),
         transform=ccrs.PlateCarree()
         )

plt.colorbar()
plt.title(NAME + 'past')

plt.show()

# sdfsdf

# values = get_predictor_data(tuple(lats.ravel()), tuple(lons.ravel()),
#                             NAME, postfix='')
# print("The number of negative occurences:", np.sum(values<0.0))
# print("The number of nan-occurences: ", np.sum(np.isnan(values)))

# plt.figure()
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax.coastlines()
# plt.contourf(lons, lats, values.reshape(RESOLUTION, RESOLUTION),
#          transform=ccrs.PlateCarree())
# plt.colorbar()
# plt.title(NAME + ' present')


# plt.show()
