
# Basic model utilities

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from .absence import data as absence_data
from .conf import DENSITY_UNIT
import pandas as pd
import numpy as np
from .loader import get_predictor_data



class PreprocessingMixin(BaseEstimator, TransformerMixin):

    def fit(self, df, y=None):
        return self

    def transform(self, df, y=None):
        return df

class DensityTweaker(PreprocessingMixin):
    def __init__(self, density=10):
        '''
        Decrease density of the train dataset according to the provided density.

        :param

          density -- a float value, allowed density in a 0.1x0.1 (lat x lon, in degrees)
                     square
        '''
        self.density_ = density


class PruneSuspiciousCoords(PreprocessingMixin):
    '''
    Removes all records having suspicious precision of positioning.
    If latitute and longitude of a record have the number
    of floating digits less than <digits> ,it is treated as suspicious.
    '''

    def __init__(self, digits=0):
        self.digits_ = digits

    def transform(self, df, y=None):
        val = 10 ** self.digits_
        lats_ind = np.equal(np.mod(df.latitude.values * val, 1), 0)
        lons_ind = np.equal(np.mod(df.longitude.values * val, 1), 0)
        return df.copy()[lats_ind * lons_ind]


class ExpertFeatureSelector(PreprocessingMixin):
    '''
    Feature selection utility function.

    Select only those featrues, that were selected by an expert.
    '''

    def __init__(self, variables=None):
        self.vartiables_ = variables

    def transform(self, df, y=None):
        if not self.variables_: return df

        filtered = []
        for name in self.variables_:
            if name in df:
                filtered.append(name)
        return df[filtered]


class FillPseudoAbsenceData(PreprocessingMixin):
    def __init__(self, density=0.1):
        '''
        Fill data frame with pseudo-absence data
        '''
        self.density_ = float(density)

    def update_df(self, df, ar, sp):
        size = abs((ar[0] - ar[-2]) * (ar[-1] - ar[1]))
        num = int((size / DENSITY_UNIT) * self.density_)
        lats = np.random.uniform(ar[0], ar[2], num)
        lons = np.random.uniform(ar[1], ar[-1], num)
        res = pd.concat([df, pd.DataFrame({'species': [sp] * len(lats),
                                            'latitude': lats,
                                            'longitude': lons,
                                            'absence': [True] * len(lats)})])
        return res

    def transform(self, df, y=None):
        res = df.copy()
        assert len(df.species.unique()) == 1, "DataFrame should contain only one species"
        res['absence'] = False
        sp = df.species.unique()[-1]
        if sp in absence_data:
            for ar in absence_data[sp]:
                res = self.update_df(res, ar, sp)
        if 'all' in absence_data:
            for ar in absence_data['all']:
                res = self.update_df(res, ar, sp)
        return res


class FillEnvironmentalData(PreprocessingMixin):

    def __init__(self, variables=None):
        self.variables_ = variables

    def transform(self, df, y=None):
        df_ = df.copy()
        for var in self.variables_:
            values = get_predictor_data(df_['latitude'].values, df_['longitude'].values, var)
            df_[var] = values
        return df_.dropna().reset_index(drop=True)


class SelectSpecies(PreprocessingMixin):

    def __init__(self, species):
        self.species_ = species

    def transform(self, df, y=None):
        df_ = df.copy()
        df_ = df_[df.species.str.contains(self.species_)]
        df_.species = self.species_
        return df_.reset_index(drop=True)

class TreeFeatureImportance(PreprocessingMixin):

    def __init__(self, variables=[], iterations=10, nest=200, pabs_density=0.1):
        self.iterations_=iterations
        self.nest_ = nest
        self.variables_ = tuple(variables)
        self.pabs_density_ = pabs_density

    def transform(self, df, y=None):
        self.df_ = df.copy()
        variables = list(self.variables_)
        common = Pipeline([('fillabsence', FillPseudoAbsenceData(density=self.pabs_density_)),
                           ('fillenvironmental', FillEnvironmentalData(variables)),
                           ])
        forest = ExtraTreesClassifier(n_estimators=self.nest_,
                                      random_state=0)
        importances = []
        print("Performing iterations of the feature selection procedure.")
        for i in range(self.iterations_):
            print('Current iteration is ', i)
            new_df = common.fit_transform(self.df_)
            y = new_df['absence'].values
            X = new_df[variables].values
            forest.fit(X, y)
            importances.append(forest.feature_importances_)
        print("Iterations finished.")
        self.feature_importances_ = np.mean(np.array(importances), axis=0).tolist()
        return df





class RFECV_FeatureSelector(PreprocessingMixin):

    def __init__(self, clfs=[('MaxEnt', LogisticRegression()),
                             ], cv=StratifiedKFold(5),
                 score_func='accuracy'):
        self.clfs_ = clfs
        self.score_func = score_func
        self.cv_ = cv

    def transform(self, df, y=None):
        result = dict()
        for name, clf in zip(clfs):
            estimator = RFECV(clf, step=1, cv=self.cv_,
                              njobs=3)






