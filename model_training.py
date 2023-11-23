from sklearn.model_selection import train_test_split
import xgboost as xgb
from utility import run_xgboost, run_surprise
from surprise import BaselineOnly, KNNBaseline, SVD, SVDpp
import joblib

def train_models(reg_train, reg_test, models_evaluation_train, models_evaluation_test, trainset, testset):

    x_train = reg_train.drop(['user','movie','rating'], axis=1)
    y_train = reg_train['rating']

    x_test = reg_test.drop(['user','movie','rating'], axis=1)
    y_test = reg_test['rating']

    first_xgb = xgb.XGBRegressor(n_jobs=13, random_state=15, n_estimators=100)
    train_results, test_results = run_xgboost(first_xgb, x_train, y_train, x_test, y_test, 'first_xgb')

    models_evaluation_train['first_xgb'] = train_results
    models_evaluation_test['first_xgb'] = test_results
########################################################################
    bsl_options = {'method': 'sgd',
                'learning_rate': .001
                }
    bsl_algo = BaselineOnly(bsl_options=bsl_options)
    bsl_train_results, bsl_test_results = run_surprise(bsl_algo, trainset, testset, 'bsl_algo')

    models_evaluation_train['bsl_algo'] = bsl_train_results 
    models_evaluation_test['bsl_algo'] = bsl_test_results
############################################################################
    x_train['bslpr'] = models_evaluation_train['bsl_algo']['predictions']
    x_test['bslpr']  = models_evaluation_test['bsl_algo']['predictions']

    xgb_bsl = xgb.XGBRegressor(n_jobs=13, random_state=15, n_estimators=100)
    train_results, test_results = run_xgboost(xgb_bsl, x_train, y_train, x_test, y_test, 'xgb_bsl')

    models_evaluation_train['xgb_bsl'] = train_results
    models_evaluation_test['xgb_bsl'] = test_results    
###################################################################################
    sim_options = {'user_based' : True,
                'name': 'pearson_baseline',
                'shrinkage': 100,
                'min_support': 2
                } 
    bsl_options = {'method': 'sgd'} 
    knn_bsl_user = KNNBaseline(k=40, sim_options = sim_options, bsl_options = bsl_options)
    knn_bsl_user_train_results, knn_bsl_user_test_results = run_surprise(knn_bsl_user, trainset, testset, 'knn_bsl_user')

    models_evaluation_train['knn_bsl_user'] = knn_bsl_user_train_results 
    models_evaluation_test['knn_bsl_user'] = knn_bsl_user_test_results
########################################################################
    sim_options = {'user_based' : False,
                'name': 'pearson_baseline',
                'shrinkage': 100,
                'min_support': 2
                } 
    bsl_options = {'method': 'sgd'}
    knn_bsl_movie = KNNBaseline(k=40, sim_options = sim_options, bsl_options = bsl_options)

    knn_bsl_movie_train_results, knn_bsl_movie_test_results = run_surprise(knn_bsl_movie, trainset, testset, 'knn_bsl_movie')

    models_evaluation_train['knn_bsl_movie'] = knn_bsl_movie_train_results 
    models_evaluation_test['knn_bsl_movie'] = knn_bsl_movie_test_results
###############################################################################
    x_train['knn_bsl_user'] = models_evaluation_train['knn_bsl_user']['predictions']
    x_train['knn_bsl_movie'] = models_evaluation_train['knn_bsl_movie']['predictions']

    x_test['knn_bsl_user'] = models_evaluation_test['knn_bsl_user']['predictions']
    x_test['knn_bsl_movie'] = models_evaluation_test['knn_bsl_movie']['predictions']

    xgb_knn_bsl = xgb.XGBRegressor(n_jobs=10, random_state=15)
    train_results, test_results = run_xgboost(xgb_knn_bsl, x_train, y_train, x_test, y_test, 'xgb_knn_bsl')
    
    models_evaluation_train['xgb_knn_bsl'] = train_results
    models_evaluation_test['xgb_knn_bsl'] = test_results

#############################################################################
    svd = SVD(n_factors=100, biased=True, random_state=15)
    svd_train_results, svd_test_results = run_surprise(svd, trainset, testset, 'svd')

    models_evaluation_train['svd'] = svd_train_results 
    models_evaluation_test['svd'] = svd_test_results

###################################################################
    svdpp = SVDpp(n_factors=50, random_state=15)
    svdpp_train_results, svdpp_test_results = run_surprise(svdpp, trainset, testset, 'svdpp')

    models_evaluation_train['svdpp'] = svdpp_train_results 
    models_evaluation_test['svdpp'] = svdpp_test_results
######################################################################
    x_train['svd'] = models_evaluation_train['svd']['predictions']
    x_train['svdpp'] = models_evaluation_train['svdpp']['predictions']

    x_test['svd'] = models_evaluation_test['svd']['predictions']
    x_test['svdpp'] = models_evaluation_test['svdpp']['predictions']

    xgb_final = xgb.XGBRegressor(n_jobs=10, random_state=15)
    train_results, test_results = run_xgboost(xgb_final, x_train, y_train, x_test, y_test, 'xgb_final')

    models_evaluation_train['xgb_final'] = train_results
    models_evaluation_test['xgb_final'] = test_results
##########################################################################
    x_train = x_train[['knn_bsl_user', 'knn_bsl_movie', 'svd', 'svdpp']]
    x_test = x_test[['knn_bsl_user', 'knn_bsl_movie', 'svd', 'svdpp']]

    xgb_all_models = xgb.XGBRegressor(n_jobs=10, random_state=15)
    train_results, test_results = run_xgboost(xgb_all_models, x_train, y_train, x_test, y_test, 'xgb_all_models')

    models_evaluation_train['xgb_all_models'] = train_results
    models_evaluation_test['xgb_all_models'] = test_results
