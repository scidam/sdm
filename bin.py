import numpy as np
import pandas as pd
import gc
import os
from IPython.core.display import display, HTML
from bin.model import *
from sklearn.pipeline import Pipeline
from bin.conf import PREDICTOR_LOADERS
from bin.loader import get_predictor_data
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.manifold import MDS
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from collections import defaultdict
import matplotlib



MAP_RESOLUTION = 500 # 5000 is default
SOURCE_DATA_PATH = './data' # relative (or absolute) path to the data directory
CSV_SEPARATOR = r';' # separator used in csv data files
DATA_FILE_NAMES = ['Picea_jezoensis.csv',
                   'Pinus_koraiensis.csv' 
                   #'all_species_final.csv',# all data files should be in the same format
                   #'new_species.csv',
                   #'Filipendula.csv',
                   #'Giant_herbs.csv',
                   #'Petasites.csv',
                   #'gbif_new.csv',
                   #'giant_herb_fulldata.csv',
                   #'Species_FINAL.csv',
                   #'heracleum_addition.csv',
                   #'points_giant_herb.csv'
                   ]
ALLOWED_COLUMNS = ['species', 'latitude', 'longitude'] # only these columns will be retained for computations
COLUMNS_DTYPES = [np.str, np.float64, np.float64] # Should have the same length as ALLOWED_COLUMNS
#CLIMATIC_MODELS = #['50cc26','50cc85','50cc45', '70cc26', '70cc85','70cc45']
CLIMATIC_MODELS = ['70cc26', '70cc85']
# CLIMATIC_MODELS = CLIMATIC_MODELS + list(map(lambda x: x.replace('cc', 'mc'), CLIMATIC_MODELS))
CLIMATIC_MODELS = list(map(lambda x: '_' + x, CLIMATIC_MODELS))
CLIMATIC_MODELS += ['_cclgm', ]
MODEL_SPECIES = [
           #   'filipendula',
           #   'senecio',
           #   'petasites',
           #   'angelica',
           #   'heracleum',
           #   'reynoutria'
               'picea jezoensis',
               'pinus koraiensis'

#                 'giant'
                 #'quercus mongolica',
                 #'kalopanax septemlobus',
                 #'quercus',
                 #'quercus crispula',
                 #'fraxinus mandshurica',
                 #'carpinus cordata',
                 #'juglans mandshurica',
                 #'phellodendron amurense',
                 #'ulmus davidiana',
                 #'acer mono',
                 #'ulmus laciniata',
               #  'pinus koraiensis',
                 #'tilia amurensis'
                ] # all  species should be given in lowercase format

# Initial set of variables (see conf.py: PREDICTOR_LOADERS parameter for details)
VARIABLE_SET = ('WKI5', 'PCKI0','PWKI0', 'CKI5', 'IC')
#VARIABLE_SET += tuple(['WIND' + str(k) for k in range(1, 13)])#
#VARIABLE_SET = ('BIO1',)
#VARIABLE_SET += tuple(['WKI' + str(k) for k in range(2, 7)])
#VARIABLE_SET += tuple(['CKI' + str(k) for k in range(2, 7)])
#VARIABLE_SET += tuple(['BIO' + str(k) for k in range(1, 4)])
#VARIABLE_SET += ('PWKI0', 'PCKI0','IT', 'IC', 'TMINM', 'TMAXM')
#VARIABLE_SET += tuple(['PREC' + str(k) for k in range(1, 13)])
#VARIABLE_SET += tuple(['TAVG' + str(k) for k in range(1, 13)])
#VARIABLE_SET += tuple(['TMIN' + str(k) for k in range(1, 13)])
#VARIABLE_SET += tuple(['TMAX' + str(k) for k in range(1, 13)])
VARIABLE_SET = tuple(set(VARIABLE_SET)) # remove duplicate variables if they are exist

CLASSIFIERS = [# ('tree', DecisionTreeClassifier(random_state=10)),
                #('NB', GaussianNB()),
                #('MaxEnt', LogisticRegression()),
                ('RF_100', RandomForestClassifier(n_estimators=100, random_state=10)),
              #  ('ada', AdaBoostClassifier(DecisionTreeClassifier(max_depth=7),
              #                             n_estimators=200, random_state=10))


                #('SVM', SVC(kernel='linear'))
                #('LDA', LinearDiscriminantAnalysis())
              ]
KFOLDS_NUMBER = 20
PSEUDO_ABSENCE_DENSITY = 0.02

original_presence_data = pd.DataFrame({col: [] for col in ALLOWED_COLUMNS}) #initialize dataframe-accumulator

for filename in DATA_FILE_NAMES:
    try:
        # data loading procedure
        data = pd.read_csv(os.path.join(SOURCE_DATA_PATH, filename),
                           sep=CSV_SEPARATOR, dtype={a:b for a,b in zip(ALLOWED_COLUMNS, COLUMNS_DTYPES)})
    except IOError:
        print("Couldn't read the file %s." % filename)
    if any(data):
        print('The file %s succesfully loaded.' % filename)
        print('=' * 50)
    # data concatenation procedure
    original_presence_data = pd.concat([original_presence_data, data[ALLOWED_COLUMNS]], ignore_index=True)



# make species names lowercased and stripped
original_presence_data['species'] = original_presence_data['species'].apply(str.lower).apply(str.strip)


# remove duplicate rows and nan values
original_presence_data = original_presence_data.dropna().drop_duplicates(ALLOWED_COLUMNS).reset_index(drop=True)
print("Unique species: ", np.unique(original_presence_data.species))

ind = 0
for species in MODEL_SPECIES:
    print("Processing: sp = {}".format(species))
    classifier_stats_acc, classifier_stats_auc = [], []
    model = Pipeline([('select_species', SelectSpecies(species)),
                      ('select_within_area', SelectDataWithinArea(bbox=[22, 100, 65, 169])),
                     # ('prune_suspicious', PruneSuspiciousCoords()),
                      #('dtweak', DensityTweaker(density=40)),
                      ('fill_absence', FillPseudoAbsenceData(density=0.3, area=[22, 100, 65, 169])),
                      ('fill_env', FillEnvironmentalData(VARIABLE_SET)),
                     # ('exclude_by_corr', CorrelationPruner(threshold=0.999, variables=VARIABLE_SET))
                      ])

    aux_result = model.fit_transform(original_presence_data)
    current_variable_set = set(VARIABLE_SET).intersection(
        set(aux_result.columns.values))
    print("Removed correlated features: ", set(VARIABLE_SET) - current_variable_set)
    print("Leaved features: ", current_variable_set)
    current_variable_set = list(current_variable_set)
    X, y = aux_result[current_variable_set].values, list(
        map(int, ~aux_result.absence))
    print("Dataset is formed.")

    optimal_vars = current_variable_set
    X, y = aux_result[optimal_vars].values, np.array(list(map(int, ~aux_result.absence)))
    print("The number of absence ponts: ", (y==0).sum())
    print("The number of presence ponts: ", (y==1).sum())

    minmax_key = {var: (X[:, i].min(), X[:, i].max()) for i, var in enumerate(optimal_vars)}
    response = {var: [] for var in optimal_vars}
    response.update({'probs': []})
    for name, clf in CLASSIFIERS:
        ind += 1
        print("Using classifier: ", name)
        std_clf = TweakedPipeline([('scaler', StandardScaler()),
                                   ('classificator', clf)])
        cv_auc = cross_val_score(std_clf, X, y, cv=10, scoring='roc_auc')
        print("AUC:", cv_auc)
        cf_matrices = []
        femp = []
        for train_index, test_index in StratifiedKFold(n_splits=20,
                                                        shuffle=True,
                                                        random_state=10).split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            std_clf.fit(X_train, y_train)
            cf_matrix = confusion_matrix(y_test, std_clf.predict(X_test))
            cf_matrix_p = (cf_matrix.T / np.sum(cf_matrix, axis=1)).T
            cf_matrices.append([cf_matrix_p[0][0], cf_matrix_p[1][1]])

            femp.append(list(std_clf.feature_importances_))


            # store data for response curve
            probs = std_clf.predict_proba(X_test[:100])
            response['probs'].append(probs[:, 1].T.tolist())
            for i, var in enumerate(optimal_vars):
                response[var].append(X_test[:100, i].T.tolist())
        print("Features:", optimal_vars)
        print('Feature importances:', np.array(femp).mean(axis=0), np.array(femp).std(axis=0), species)
        print('Confusion matrices:', np.mean(cf_matrices, axis=0), np.std(cf_matrices, axis=0))

        
        std_clf.fit(X, y)
        fig1, ax = plot_map([22, 67], [100, 169], MAP_RESOLUTION, std_clf,
                            optimal_vars, train_df=aux_result,
                            name=species + '_' + str(ind), postfix='')
        
        ax.set_xlabel('CF_diag: %s +/- %s'%(np.mean(cf_matrices, axis=0), np.std(cf_matrices, axis=0)))
        ax.set_ylabel(';'.join(['%s=%s'%(key,val) for key,val in grid.items()]))
        fig1.set_size_inches(18.5, 10.5)
        fig1.savefig('_'.join([species,  name]) + '_' + str(ind) + '.png', dpi=300)
        plt.close(fig1)
        gc.collect()

        for cm in CLIMATIC_MODELS:
            print("CURRENT MODEL:", cm)
            fig2, ax = plot_map([22, 67], [100, 169], MAP_RESOLUTION, std_clf,
                            optimal_vars, train_df=None,
                            name='_'.join([species, cm, name, str(ind), 'AUC=%0.2f +/- %0.2f' % (np.mean(cv_auc), np.std(cv_auc))]),
                            postfix=cm)
            ax.set_xlabel('CF_diag: %s +/- %s'%(np.mean(cf_matrices, axis=0), np.std(cf_matrices, axis=0)))
            fig2.set_size_inches(18.5, 10.5)
            fig2.savefig(cm + '_'.join([species, name]) + '_' + str(ind) + '.png', dpi=300)
            plt.close(fig2)
            gc.collect()

