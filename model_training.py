import pandas as pd
import xgboost as xgb
from utility import run_xgboost, run_surprise
from surprise import BaselineOnly, KNNBaseline, SVD, SVDpp

def train_models(reg_train, reg_test, trainset, testset):
    models = {}
    x_train = reg_train.drop(['user','movie','rating'], axis=1)
    y_train = reg_train['rating']

    x_test = reg_test.drop(['user','movie','rating'], axis=1)
    y_test = reg_test['rating']

    xgbst = xgb.XGBRegressor(n_jobs=13, random_state=15, n_estimators=100)
    run_xgboost(xgbst, x_train, y_train, x_test, y_test, 'XGboost', models)

########################################################################
    bsl_options = {'method': 'sgd',
                'learning_rate': .001
                }
    bsl_algo = BaselineOnly(bsl_options=bsl_options)
    run_surprise(bsl_algo, trainset, testset, 'bsl_algo', models)

###################################################################################
    sim_options = {'user_based' : True,
                'name': 'pearson_baseline',
                'shrinkage': 100,
                'min_support': 2
                } 
    bsl_options = {'method': 'sgd'} 
    knn_bsl_user = KNNBaseline(k=40, sim_options = sim_options, bsl_options = bsl_options)
    run_surprise(knn_bsl_user, trainset, testset, 'knn_bsl_user', models)

########################################################################
    sim_options = {'user_based' : False,
                'name': 'pearson_baseline',
                'shrinkage': 100,
                'min_support': 2
                } 
    bsl_options = {'method': 'sgd'}
    knn_bsl_movie = KNNBaseline(k=40, sim_options = sim_options, bsl_options = bsl_options)

    run_surprise(knn_bsl_movie, trainset, testset, 'knn_bsl_movie', models)

#############################################################################
    svd = SVD(n_factors=100, biased=True, random_state=15)
    run_surprise(svd, trainset, testset, 'svd', models)

###################################################################
    svdpp = SVDpp(n_factors=50, random_state=15)
    run_surprise(svdpp, trainset, testset, 'svdpp', models)
###################################################################
    return models
