
# Cython module

import numpy as np
cimport numpy as np

cpdef np.ndarray get_data_by_coordinate(np.ndarray lats,
                                       np.ndarray lons,
                                       np.ndarray dataset,
                                       double xmin,
                                       double xres,
                                       double ymax,
                                       double yres,
                                       int RasterXSize,
                                       int RasterYSize,
                                       double nodata
                                       ):

    cdef long int data_len = lats.shape[0]
    cdef np.ndarray[double, ndim=1] values = np.empty(data_len, dtype=np.float64)
    cdef int i
    for i in range(data_len):
        if lats[i] <= ymax and lats[i] > (ymax + RasterYSize * yres) and lons[i] >= xmin and lons[i] < (xmin + RasterXSize * xres):
            value =  dataset[int((lats[i] - ymax) / yres), int((lons[i] - xmin) / xres)]
            if np.isclose(value, nodata):
                value = np.nan
        else:
            value = np.nan
        values[i] = value
    return values
