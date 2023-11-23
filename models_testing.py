import joblib
from utility import evaluate_model_XGB, evaluate_model_SRP

def test_models(reg_test, models_evaluation_test, testset):
    
    x_test = reg_test.drop(['user','movie','rating'], axis=1)
    y_test = reg_test['rating']

    first_xgb = joblib.load('first_xgb.joblib')
    y_pred = first_xgb.predict(x_test)
    evaluate_model_XGB(y_test, y_pred, 'first_xgb', models_evaluation_test)
    
    #################################################
    bsl_algo = joblib.load('bsl_algo.joblib')
    test_preds = bsl_algo.test(testset)
    evaluate_model_SRP(test_preds, 'bsl_algo', models_evaluation_test)
    #################################################
    x_test['bsl_algo'] = models_evaluation_test['bsl_algo']['predictions']

    xgb_bsl = joblib.load('xgb_bsl.joblib')
    y_pred = xgb_bsl.predict(x_test)
    evaluate_model_XGB(y_test, y_pred, 'xgb_bsl', models_evaluation_test)
    ##################################################
    knn_bsl_user = joblib.load('knn_bsl_user.joblib')
    test_preds = knn_bsl_user.test(testset)
    evaluate_model_SRP(test_preds, 'knn_bsl_user', models_evaluation_test)
    ###################################################
    knn_bsl_movie = joblib.load('knn_bsl_movie.joblib')
    test_preds = knn_bsl_movie.test(testset)
    evaluate_model_SRP(test_preds, 'knn_bsl_movie', models_evaluation_test)
    ###################################################
    x_test['knn_bsl_user'] = models_evaluation_test['knn_bsl_user']['predictions']
    x_test['knn_bsl_movie'] = models_evaluation_test['knn_bsl_movie']['predictions']
    
    xgb_knn_bsl = joblib.load('xgb_knn_bsl.joblib')
    y_pred = xgb_knn_bsl.predict(x_test)
    evaluate_model_XGB(y_test, y_pred, 'xgb_knn_bsl', models_evaluation_test)
    #####################################################
    svd = joblib.load('svd.joblib')
    test_preds = svd.test(testset)
    evaluate_model_SRP(test_preds, 'svd', models_evaluation_test)
    ###################################################
    svdpp = joblib.load('svdpp.joblib')
    test_preds = svdpp.test(testset)
    evaluate_model_SRP(test_preds, 'svdpp', models_evaluation_test)
    #########################################
    x_test['svd'] = models_evaluation_test['svd']['predictions']
    x_test['svdpp'] = models_evaluation_test['svdpp']['predictions']

    xgb_final = joblib.load('xgb_final.joblib')
    y_pred = xgb_final.predict(x_test)
    evaluate_model_XGB(y_test, y_pred, 'xgb_final', models_evaluation_test)
    ###########################################
    x_test = x_test[['knn_bsl_user', 'knn_bsl_movie', 'svd', 'svdpp']]

    xgb_all_models = joblib.load('xgb_all_models.joblib')
    y_pred = xgb_all_models.predict(x_test)
    evaluate_model_XGB(y_test, y_pred, 'xgb_all_models', models_evaluation_test)
    ##########################################
    
