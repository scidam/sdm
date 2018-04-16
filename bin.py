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
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage
from collections import defaultdict

SOURCE_DATA_PATH = './data' # relative (or absolute) path to the data directory
CSV_SEPARATOR = r';' # separator used in csv data files
DATA_FILE_NAMES = ['all_species_final.csv',# all data files should be in the same format
                   'new_species.csv',
                   'Filipendula.csv',
                   'Giant_herbs.csv',
                   'Petasites.csv',
                   'gbif_new.csv',
                   'giant_herb_fulldata.csv',
                   'Species_FINAL.csv'
                   ]
ALLOWED_COLUMNS = ['species', 'latitude', 'longitude'] # only these columns will be retained for computations
COLUMNS_DTYPES = [np.str, np.float64, np.float64] # Should have the same length as ALLOWED_COLUMNS
#CLIMATIC_MODELS = #['50cc26','50cc85','50cc45', '70cc26', '70cc85','70cc45']
CLIMATIC_MODELS = ['70cc26', '70cc85']
# CLIMATIC_MODELS = CLIMATIC_MODELS + list(map(lambda x: x.replace('cc', 'mc'), CLIMATIC_MODELS))
CLIMATIC_MODELS = list(map(lambda x: '_' + x, CLIMATIC_MODELS))
CLIMATIC_MODELS += ['_cclgm', ]
MODEL_SPECIES = [
              #'filipendula',
               # 'senecio',
              #'petasites',
              #'angelica',
              #'heracleum',
              'reynoutria'

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
        print('File overview:')
        data.info()
        print('='*50)
    # data concatenation procedure
    original_presence_data = pd.concat([original_presence_data, data[ALLOWED_COLUMNS]], ignore_index=True)

# make species names lowercased and stripped
original_presence_data['species'] = original_presence_data['species'].apply(str.lower).apply(str.strip)

# remove duplicate rows and nan values
original_presence_data = original_presence_data.dropna().drop_duplicates(ALLOWED_COLUMNS).reset_index(drop=True)
original_presence_data.info()

print("Unique species: ", np.unique(original_presence_data.species))


parameter_grid_search = [
                {'ps_density': 4,
                 'density': 3,
                 'similarity': 0.0,
                             },
                #{'ps_density': 4,
                 #'density': 2,
                 #'similarity': 0.0,
                 #},
                #{'ps_density': 4,
                 #'density': 0.5,
                 #'similarity': 0.0,
                 #},
                  #{'ps_density': 4,
                   #'density': 4,
                   #'similarity': 0.0,
                   #},
                 ]


def make_response(edges, xdata,  ydata, sigma=4):
    newx, newy = [], []
    xdata = np.array(xdata)
    ydata = np.array(ydata)
    for k in range(len(edges))[:-1]:
        ids = (xdata > edges[k]) * (xdata <= edges[k + 1])
        newx.append((edges[k] + edges[k + 1]) / 2.0)
        if any(ids):
            newy.append(ydata[ids].max())
        else:
            newy.append(0.0)
    newx = ndimage.gaussian_filter1d(newx, sigma)
    newy = ndimage.gaussian_filter1d(newy, sigma)
    return newx, newy




for ind, grid in enumerate(parameter_grid_search):
    for species in MODEL_SPECIES:
        print("Processing: sp = %s" % species)
        classifier_stats_acc, classifier_stats_auc = [], []
        model = Pipeline([('select_species', SelectSpecies(species)),
                          ('select_within_area', SelectDataWithinArea(bbox=[22, 100, 65, 169])),
                          ('prune_suspicious', PruneSuspiciousCoords()),
                          ('dtweak', DensityTweaker(density=40)),
                          ('ps_absence', FillPseudoAbsenceData(density=grid['ps_density'])),
                          ('fill_absence', FillPseudoAbsenceData(density=grid['density'],
                                                 area=[22, 100, 65, 169])),
                          ('fill_env', FillEnvironmentalData(VARIABLE_SET)),
                          # ('fill_by_cond',
                          #  FillPseudoAbsenceByConditions(species=species,
                          #                                similarity=grid['similarity'],
                          #                                density=grid['density'], #0.1 for trees
                          #                                area=[(22, 100),
                          #                                      (65, 169)])),
                          ('exclude_by_corr', CorrelationPruner(threshold=0.9999,
                                                                variables=VARIABLE_SET))
                          ])

        print("Constructing the dataset...")
        aux_result = model.fit_transform(original_presence_data)
        current_variable_set = set(VARIABLE_SET).intersection(
            set(aux_result.columns.values))
        print("Removed correlated features: ",
              set(VARIABLE_SET) - current_variable_set)
        print("Leaved features: ", current_variable_set)
        current_variable_set = list(current_variable_set)
        X, y = aux_result[current_variable_set].values, list(
            map(int, ~aux_result.absence))
        print("Dataset is formed.")
        # for name, clf in CLASSIFIERS:
        #     std_clf = TweakedPipeline([('scaler', StandardScaler()),
        #                                ('classificator', clf)])
        #     print("Preforming recursive feature ellimination for the <%s> classifier..." % name)
        #     rfecv_acc = RFECV(estimator=std_clf, step=1,
        #                       cv=StratifiedKFold(KFOLDS_NUMBER, shuffle=True),
        #                       scoring='accuracy')
        #     rfecv_acc.fit(X, y)
        #     acc_score = np.array(rfecv_acc.grid_scores_)[
        #         np.argmax(rfecv_acc.grid_scores_)]
        #     rfecv_auc = RFECV(estimator=std_clf, step=1,
        #                       cv=StratifiedKFold(KFOLDS_NUMBER, shuffle=True),
        #                       scoring='roc_auc')
        #     rfecv_auc.fit(X, y)
        #     auc_score = np.array(rfecv_auc.grid_scores_)[
        #         np.argmax(rfecv_auc.grid_scores_)]
        #     classifier_stats_acc.append(
        #         (name, acc_score, std_clf, rfecv_acc.support_))
        #     classifier_stats_auc.append(
        #         (name, auc_score, std_clf, rfecv_auc.support_))
        # acc_optimal_name, acc_optimal_score, acc_optimal_clf, acc_optimal_mask = tuple(
        #     classifier_stats_acc[
        #         np.argmax(list(map(lambda x: x[1], classifier_stats_acc)))])
        # auc_optimal_name, auc_optimal_score, auc_optimal_clf, auc_optimal_mask = tuple(
        #     classifier_stats_auc[
        #         np.argmax(list(map(lambda x: x[1], classifier_stats_auc)))])
        # display(HTML(
        #     '<h5> --------------- Summary for %s: --------------- </h5>' % species))
        # print("The best classifier is %s. Its accuracy score is %s." % (
        # acc_optimal_name, acc_optimal_score))
        # print("Optimal predictor set (acc): ",
        #       np.array(current_variable_set)[acc_optimal_mask])
        # print("The best classifier is %s. Its roc/auc score is %s." % (
        # auc_optimal_name, auc_optimal_score))
        # print("Optimal predictor set (auc): ",
        #       np.array(current_variable_set)[auc_optimal_mask])
        # print("Statistic over all classifiers: ")
        # print("AUC/ROC - Case:")
        # df = pd.DataFrame(
        #     {n[0]: [n[1], np.array(current_variable_set)[n[-1]][:5]] for n in
        #      classifier_stats_auc})
        # print(df)
        # print("Precision - Case:")
        # df = pd.DataFrame(
        #     {n[0]: [n[1], np.array(current_variable_set)[n[-1]][:5]] for n in
        #      classifier_stats_acc})
        # print(df)
        # print(HTML('<h5> %s </h5>' % ("~" * 90,)))
        optimal_vars = current_variable_set
        X, y = aux_result[optimal_vars].values, np.array(list(map(int, ~aux_result.absence)))
        print("The number of absence ponts: ", (y==0).sum())
        print("The number of presence ponts: ", (y==1).sum())

        minmax_key = {var: (X[:, ind].min(), X[:, ind].max()) for ind, var in enumerate(optimal_vars)}
        response = {var: [] for var in optimal_vars}
        response.update({'probs': []})
        for name, clf in CLASSIFIERS:
            print("Using classifier: ", name)
            std_clf = TweakedPipeline([('scaler', StandardScaler()),
                                       ('classificator', clf)])
            cv_auc = cross_val_score(std_clf, X, y, cv=10, scoring='roc_auc')
            print("AUC:", cv_auc)
            cf_matrices = []
            for train_index, test_index in StratifiedKFold(n_splits=10,
                                                           shuffle=True,
                                                           random_state=10).split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                std_clf.fit(X_train, y_train)
                cf_matrix = confusion_matrix(y_test, std_clf.predict(X_test))
                cf_matrix_p = (cf_matrix.T / np.sum(cf_matrix, axis=1)).T
                cf_matrices.append([cf_matrix_p[0][0], cf_matrix_p[1][1]])

                # store data for response curve
                probs = std_clf.predict_proba(X_test)
                response['probs'].append(probs[:, 1].T.tolist())
                for ind, var in enumerate(optimal_vars):
                    response[var].append(X_test[:, ind].T.tolist())

            std_clf.fit(X, y)
            fig1, ax = plot_map([22, 67], [100, 169], 500, std_clf,
                                optimal_vars, train_df=aux_result,
                                name=species + '_' + str(ind), postfix='')

            ax.set_xlabel('CF_diag: %s +/- %s'%(np.mean(cf_matrices, axis=0), np.std(cf_matrices, axis=0)))
            ax.set_ylabel(';'.join(['%s=%s'%(key,val) for key,val in grid.items()]))
            fig1.set_size_inches(18.5, 10.5)
            fig1.savefig('_'.join([species,  name]) + '_' + str(ind) + '.png', dpi=600)
            plt.close(fig1)
            gc.collect()

            # plot response curves
            keys = list(response.keys())
            keys.remove('probs')
            for key in keys:
                minx = minmax_key[key][0]
                maxx = minmax_key[key][1]
                resps = []
                for a, b in zip(response[key], response['probs']):
                    xdata, ydata = make_response(np.linspace(minx, maxx, 100), a, b)
                    resps.append(ydata)
                resps = np.array(resps)
                ydata_med = np.percentile(resps, 50, axis=0)
                ydata_l = resps.min(axis=0)
                ydata_u = resps.max(axis=0)
                figr = plt.figure()
                figr.set_size_inches(10, 10)
                axr = figr.add_subplot(111)
                axr.plot(xdata, ydata_med, '-r', xdata, ydata_l, '-b', xdata, ydata_u, '-b')
                figr.savefig('_'.join([species,  name, 'reponse', key]) + '_' + str(ind)  + '.png', dpi=600)
                plt.close(figr)

            for cm in CLIMATIC_MODELS:
                print("CURRENT MODEL:", cm)
                fig2, ax = plot_map([22, 67], [100, 169], 500, std_clf,
                                optimal_vars, train_df=None,
                                name='_'.join([species, cm, name, str(ind), 'AUC=%0.2f +/- %0.2f' % (np.mean(cv_auc), np.std(cv_auc))]),
                                postfix=cm)
                ax.set_xlabel('CF_diag: %s +/- %s'%(np.mean(cf_matrices, axis=0), np.std(cf_matrices, axis=0)))
                fig2.set_size_inches(18.5, 10.5)
                fig2.savefig(cm + '_'.join([species, name]) + '_' + str(ind) + '.png', dpi=600)
                plt.close(fig2)
                gc.collect()
