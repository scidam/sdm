
# Basic model utilities

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

from absence import data
from conf import DENSITY_UNIT
import pandas as pd
import numpy as np


class DensityTweaker(BaseEstimator, TransformerMixin):
    def __init__(self, density=10):
        '''
        Decrease density of the train dataset according to the provided density.

        :param

          density -- a float value, allowed density in a 0.1x0.1 (lat x lon, in degrees)
                     square
        '''
        self.density_ = density

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df


class FillPseudoAbsenceData(BaseEstimator, TransformerMixin):
    def __init__(self, density=10):
        '''
        Fill data frame with pseudo-absence data
        '''
        self.density_ = density

    def fit(self, df, y=None):
        return self


    def update_df(self, df, ar, sp):
        size = abs((ar[0] - ar[-2]) * (ar[-1] - ar[1]))
        num = (size / DENSITY_UNIT) * self.density_
        lats = np.random.uniform(ar[0], ar[2], num)
        lons = np.random.uniform(ar[1], ar[-1], num)
        res = pd.concat([df, pd.DataFrame({'species': [sp] * len(lats),
                                            'latitude': lats,
                                            'longitude': lons,
                                            'absence': [True] * len(lats)})
        return res

    def transform(self, df, y=None):
        res = df
        for sp in df.species.unique:
            if sp in data:
                for ar in data[sp]:
                    res = self.update_df(res, ar, sp)

        if 'all' in data:
            for sp in df.species.unqique:
                for ar in data['all']:
                    res = self.update_df(res, ar, sp)
        return res
