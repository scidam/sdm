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
               'Picea jezoensis',
               'Pinus koraiensis'

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
                 'density': 4,
                 'similarity': 0.0,
                 'ms':  ['giant', 'filipendula']
                             },
                {'ps_density': 4,
                 'density': 2,
                 'similarity': 0.0,
                 'ms':  ['reynoutria',]
                 },
                {'ps_density': 4,
                 'density': 3,
                 'similarity': 0.0,
                 'ms':  ['petasites', 'heracleum', 'angelica']
                 },
                # {'ps_density': 4,
                #  'density': 4,
                #  'similarity': 0.0,
                #  'ms':  ['all',]
                #              },
                
                #{'ps_density': 4,
                 #'density': 0.5,
                 #'similarity': 0.0,
                 #},
                  #{'ps_density': 4,
                   #'density': 4,
                   #'similarity': 0.0,
                   #},
                 ]


def make_response(edges, xdata,  ydata, sigma=7):
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



##================= make  LDA - plot ====================

#def add_convex_hull(ax, points, color='r', label=''):
    #from scipy.spatial import ConvexHull
    #from matplotlib.patches import Polygon
    #hull = ConvexHull(points)
    
    #ax.plot(np.append(points[hull.vertices,0], points[hull.vertices[0],0]),
            #np.append(points[hull.vertices,1],points[hull.vertices[0],1]), '%s--o'%color, lw=2, label=label)
    

#model = Pipeline([('select_species', SelectSpeciesList(MODEL_SPECIES, overwrite=True)),
                  #('select_within_area', SelectDataWithinArea(bbox=[22, 100, 65, 169])),
                  #('prune_suspicious', PruneSuspiciousCoords()),
                  #('dtweak', DensityTweaker(density=40)),
                  #('fill_env', FillEnvironmentalData(VARIABLE_SET)),
                 #])

#prj_clfs = ( # ('LDA',  LinearDiscriminantAnalysis(n_components=2)),
             #('PCA',  PCA(n_components=2)),
##             ('MDS',  MDS(n_components=2)),
##             ('NMDS', MDS(n_components=2, metric=False))
            #)
            

#labels = LabelEncoder()
#aux_result = model.fit_transform(original_presence_data)

#y = labels.fit_transform(aux_result.species.values)
#matplotlib.rcParams.update({'legend.fontsize': 14, 'xtick.labelsize':16,
                            #'ytick.labelsize': 16, 'font.size': 14,
                            #'axes.linewidth': 2,
                            #'xtick.major.width':1.5
                            #})

#for clf_name, clf in prj_clfs:
    #proj = clf.fit_transform(StandardScaler().fit_transform(aux_result.loc[:, VARIABLE_SET]), y)
    #fig = plt.figure()
    #fig.set_size_inches(10.5, 10.5)
    #ax = fig.add_subplot(111)

    #if 'DS' not in clf_name:
        ## creating arrows,
        #unity_values = [1.5] * len(VARIABLE_SET)
        #x0, y0 = clf.transform([StandardScaler().fit_transform(aux_result.loc[:, VARIABLE_SET]).mean(axis=0)])[0]

        #for key, u in zip(VARIABLE_SET, unity_values):
            #x1, y1 = clf.transform([((np.array(VARIABLE_SET) == key).astype(int)*u).tolist()])[0]
            #if key == 'WKI5':
                #angle = np.arcsin(y1 / np.sqrt(x1 * x1 + y1 * y1))
                #rotmat = np.array([[np.cos(angle), np.sin(angle)],
                                   #[-np.sin(angle), np.cos(angle)]])

        #for key, u in zip(VARIABLE_SET, unity_values):
            #x1, y1 = clf.transform([((np.array(VARIABLE_SET) == key).astype(int)*u).tolist()])[0]
            #x1_, y1_ = (rotmat @ np.array([x1, y1]).T).T.tolist()
            #ax.arrow(x0, y0, x1_, y1_, head_width=0.1, head_length=0.2, fc='k', ec='k')
            #ax.text(x1_+0.1, y1_, key)

    #for sp, m, i,c in zip(np.unique(y), ['o', 'v', 's', 'd', '^'], range(5), 'rbgkc'):
        ##ax.scatter(proj[y==sp,0], proj[y==sp,1], marker=m, s=30, label=labels.inverse_transform(i),
                ##facecolors='none', edgecolors=c)
        #add_convex_hull(ax, (rotmat @ proj[y==sp,:].T).T, color=c, label=sp)

    #ax.legend()
    #ax.set_title(clf_name + 'with aligned components')
    #ax.set_xlabel('PC1')
    #ax.set_ylabel('PC2')
    #fig.savefig(clf_name + '.png', dpi=300)
    #plt.close(fig)
    #gc.collect()
    #print(clf.explained_variance_ratio_)
    #tot = clf.explained_variance_[0]/clf.explained_variance_ratio_[0]
    #print('Main axis1', np.var((rotmat @ proj[y==sp,:].T).T[:,0],ddof=1)/tot)
    #print('Main axis2', np.var((rotmat @ proj[y==sp,:].T).T[:,1],ddof=1)/tot)

## #=======================================================
#sdjfkl

# Make for all species.. 
# original_presence_data = SelectSpeciesList(MODEL_SPECIES, overwrite=True).fit_transform(original_presence_data)
# MODEL_SPECIES = ['all']
# original_presence_data.species = 'all'


for ind, grid in enumerate(parameter_grid_search):
    # if grid['ms'][0] == 'all':
    #     original_presence_data = SelectSpeciesList(['filipendula',
    #           # 'senecio',
    #           'petasites',
    #           'angelica',
    #           'heracleum',
    #           'reynoutria'], overwrite=True).fit_transform(original_presence_data)
    #     original_presence_data.species = 'all'
    MODEL_SPECIES = grid['ms']
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

        minmax_key = {var: (X[:, i].min(), X[:, i].max()) for i, var in enumerate(optimal_vars)}
        response = {var: [] for var in optimal_vars}
        response.update({'probs': []})
        for name, clf in CLASSIFIERS:
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

            # save responses
            #respname = '%s' % '_'.join([species, name, str(ind)]) + '.dat'
            #with open(respname, 'wb') as f:
                #pickle.dump([response, minmax_key], f)
            #
            #
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
                ydata_l = np.percentile(resps, 2.5, axis=0)
                ydata_u = np.percentile(resps, 97.5, axis=0)
                figr = plt.figure()
                figr.set_size_inches(10, 10)
                axr = figr.add_subplot(111)
                axr.plot(xdata, ydata_med, '-r')
                axr.fill_between(xdata, ydata_l, ydata_u, facecolor='gray', alpha=0.5)
                figr.savefig('_'.join([species,  name, 'reponse', key]) + '_' + str(ind)  + '.png', dpi=300)
                plt.close(figr)

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

