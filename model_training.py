from sklearn.model_selection import train_test_split
import xgboost as xgb
from utility import run_xgboost, run_surprise
from surprise import BaselineOnly, KNNBaseline, SVD, SVDpp
import joblib

def train_models(reg_data, models_evaluation, surprise_data):
    models = []
    x = reg_data.drop(['user','movie','rating'], axis=1)
    y = reg_data['rating']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

    first_xgb = xgb.XGBRegressor(silent=False, n_jobs=13, random_state=15, n_estimators=100)
    results = run_xgboost(models, first_xgb, x_train, y_train, x_test, y_test, 'first_xgb')

    models_evaluation['first_algo'] = results
##################################################################
    bsl_options = {'method': 'sgd', 'learning_rate': .001}
    bsl_algo = BaselineOnly(bsl_options=bsl_options)
    
    results = run_surprise(models, bsl_algo, surprise_data, 'bsl_options')
    models_evaluation['bsl_algo'] = results
##################################################################
    x_train['bslpr'] = models_evaluation['bsl_algo']['predictions']

    xgb_bsl = xgb.XGBRegressor(silent=False, n_jobs=13, random_state=15, n_estimators=100)
    results = run_xgboost(models, xgb_bsl, x_train, y_train, x_test, y_test, 'xgb_bsl')

    models_evaluation['xgb_bsl'] = results
##################################################################
    sim_options = {'user_based' : True, 'name': 'pearson_baseline', 'shrinkage': 100, 'min_support': 2}
    bsl_options = {'method': 'sgd'} 

    knn_bsl_u = KNNBaseline(k=40, sim_options = sim_options, bsl_options = bsl_options)
    results = run_surprise(models, knn_bsl_u, surprise_data, 'knn_bsl_u')

    models_evaluation['knn_bsl_u'] = results
###################################################################
    sim_options = {'user_based' : False, 'name': 'pearson_baseline', 'shrinkage': 100, 'min_support': 2} 
    bsl_options = {'method': 'sgd'} 

    knn_bsl_m = KNNBaseline(k=40, sim_options = sim_options, bsl_options = bsl_options)
    results = run_surprise(models, knn_bsl_m, surprise_data, 'knn_bsl_u')

    models_evaluation['knn_bsl_u'] = results
###################################################################
    x_train['knn_bsl_u'] = models_evaluation['knn_bsl_u']['predictions']
    x_train['knn_bsl_m'] = models_evaluation['knn_bsl_m']['predictions']

    xgb_knn_bsl = xgb.XGBRegressor(n_jobs=10, random_state=15)
    results = run_xgboost(models, xgb_knn_bsl, x_train, y_train, x_test, y_test, 'xgb_knn_bsl')

    models_evaluation['xgb_knn_bsl'] = results
#####################################################################
    svd = SVD(n_factors=100, biased=True, random_state=15, verbose=True)
    results = run_surprise(models, svd, surprise_data, 'svd')

    models_evaluation['svd'] = results
####################################################################
    svdpp = SVDpp(n_factors=50, random_state=15, verbose=True)
    results = run_surprise(models, svdpp, surprise_data, 'svdpp')

    models_evaluation['svdpp'] = results
####################################################################
    x_train['svd'] = models_evaluation['svd']['predictions']
    x_train['svdpp'] = models_evaluation['svdpp']['predictions']

    xgb_final = xgb.XGBRegressor(n_jobs=10, random_state=15)
    results = run_xgboost(models, xgb_final, x_train, y_train, x_test, y_test, 'xgb_final')

    models_evaluation['xgb_final'] = results
#####################################################################
    x_train = x_train[['knn_bsl_u', 'knn_bsl_m', 'svd', 'svdpp']]
    x_test = x_test[['knn_bsl_u', 'knn_bsl_m', 'svd', 'svdpp']]

    xgb_all_models = xgb.XGBRegressor(n_jobs=10, random_state=15)
    results = run_xgboost(models, xgb_all_models, x_train, y_train, x_test, y_test, 'xgb_all_models')

    models_evaluation['xgb_all_models'] = results
#########################################################################
    return models