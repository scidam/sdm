# -*- coding:utf-8 -*-
"""
Created Date: Tuesday October 29th 2019
Author: Dmitry Kislov
E-mail: kislov@easydan.com
-----
Last Modified: Friday, November 1st 2019, 9:50:54 am
Modified By: Dmitry Kislov
-----
Copyright (c) 2019
"""


from .loader import get_predictor_data, get_kiras_indecies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cartopy.crs as ccrs

NAME='TMIN5'
RESOLUTION = 500
lats, lons = np.meshgrid(np.linspace(42, 44, RESOLUTION),
                         np.linspace(130, 134, RESOLUTION))

valuesf = get_predictor_data(tuple(lats.ravel()), tuple(lons.ravel()), NAME, postfix='')
print("The number of negative occurences:", np.sum(valuesf<0.0))
print("The number of nan-occurences: ", np.sum(np.isnan(valuesf)))
print("Mean value", np.nanmean(valuesf))



# VLADIVOSTOK

# names = ['TAVG%s' % k for k in range(1, 13)]


# for model in ['50cc26','50cc85','50cc45', '70cc26', '70cc85','70cc45', 'cclgm', 'ccmid']:
#         name='TAVG7'
#         print(name, model, get_predictor_data((43.137299,), (131.946541,),
#           name, postfix='_'+model))


# print('WKI5=',  get_predictor_data((43.137299,), (131.946541,),
#           'WKI5', postfix=''))


fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()
plt.contourf(lons, lats, valuesf.reshape(RESOLUTION, RESOLUTION),
         transform=ccrs.PlateCarree())
plt.colorbar()
plt.title(NAME)
plt.savefig('output.png', dpi=300)


# values = get_predictor_data(tuple(lats.ravel()), tuple(lons.ravel()),
#                             NAME, postfix='')
# print("The number of negative occurences:", np.sum(values<0.0))
# print("The number of nan-occurences: ", np.sum(np.isnan(values)))

# plt.figure()

# 
# plt.contourf(lons, lats, values.reshape(RESOLUTION, RESOLUTION),
#          transform=ccrs.PlateCarree())
# plt.colorbar()
# plt.title(NAME + ' present')


# plt.show()
