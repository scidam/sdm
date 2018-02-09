# Cython module

import numpy as np
cimport numpy as np

cdef double large_value = 10000.0

# Deprecated in flavour of numpy indexing...
cpdef np.ndarray[double, ndim=1] get_data_by_coordinate(np.ndarray [double, ndim=1] lats,
                                       np.ndarray [double, ndim=1] lons,
                                       np.ndarray [double, ndim=2] dataset,
                                       double xmin,
                                       double xres,
                                       double ymax,
                                       double yres,
                                       long int RasterXSize,
                                       long int RasterYSize,
                                       ):

    cdef long int data_len = lats.shape[0]
    cdef np.ndarray[double, ndim=1] values = np.empty(data_len, dtype=np.float64)
    cdef int i
    cdef double value
    for i in range(data_len):
        if lats[i] <= ymax and lats[i] > (ymax + RasterYSize * yres) and lons[i] >= xmin and lons[i] < (xmin + RasterXSize * xres):
            value =  dataset[int((lats[i] - ymax) / yres), int((lons[i] - xmin) / xres)]
            if abs(value) > large_value:
                value = np.nan
        else:
            value = np.nan
        values[i] = value
    return values
