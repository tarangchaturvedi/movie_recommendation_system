import joblib
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from surprise import Dataset, Reader
import random
from scipy import sparse

def read_csv(file_path):
    print("-------Reading from csv file-------")
    df = pd.read_csv(file_path)
    return df

def sort_datetime(df):
    print("--------sorting datetime values--------")
    df.date = pd.to_datetime(df.date)
    df.sort_values(by = 'date', inplace = True)
    return df

def to_sparse_matrix(df):
    print("--------converting dataframe to sparse matrix--------")
    df = csr_matrix((df.rating.values, (df.user.values, df.movie.values)))
    return df

def clean_data(df):
    print("-------cleaning data------")
    df.dropna(inplace = True)
    df.drop_duplicates(inplace = True)
    return df

# get the user averages in dictionary (key: user_id/movie_id, value: avg rating)
def get_average_ratings(sparse_matrix, of_users):
    # average ratings of user/axes
    ax = 1 if of_users else 0 # 1 - User axes,0 - Movie axes

    sum_of_ratings = sparse_matrix.sum(axis=ax).A1
    # Boolean matrix of ratings ( whether a user rated that movie or not)
    is_rated = sparse_matrix!=0
    no_of_ratings = is_rated.sum(axis=ax).A1
    # max_user  and max_movie ids in sparse matrix 
    u,m = sparse_matrix.shape
    average_ratings = { i : sum_of_ratings[i]/no_of_ratings[i] for i in range(u if of_users else m) if no_of_ratings[i] !=0}

    return average_ratings

def adding_average_ratings(sparse_matrix, averages):
    print("------adding average rating values------")
    averages['global'] = sparse_matrix.sum()/ sparse_matrix.count_nonzero()
    averages['user'] = get_average_ratings(sparse_matrix, of_users=True)
    averages['movie'] =  get_average_ratings(sparse_matrix, of_users=False)

def get_sample_sparse_matrix(sparse_matrix, no_users, no_movies):
    print("------------------------------------------")
    row_ind, col_ind, ratings = sparse.find(sparse_matrix)
    users = np.unique(row_ind)
    movies = np.unique(col_ind)

    print("Original Matrix : (users, movies) -- ({} {})".format(len(users), len(movies)))
    print("Original Matrix : Ratings -- {}\n".format(len(ratings)))

    np.random.seed(15)
    sample_users = np.random.choice(users, no_users, replace=False)
    sample_movies = np.random.choice(movies, no_movies, replace=False)
    # get the boolean mask or these sampled_items in originl row/col_inds..
    mask = np.logical_and( np.isin(row_ind, sample_users), np.isin(col_ind, sample_movies) )
    
    sample_sparse_matrix = csr_matrix((ratings[mask], (row_ind[mask], col_ind[mask])), shape=(max(sample_users)+1, max(sample_movies)+1))

    print("Sampled Matrix : (users, movies) -- ({} {})".format(len(sample_users), len(sample_movies)))
    print("Sampled Matrix : Ratings --", format(ratings[mask].shape[0]))
    print("------------------------------------------")
    return sample_sparse_matrix

def surprise_data_creation(reg_train, reg_test):
    print("---------creating data for surprise model----------\n")
    reader = Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(reg_train[['user', 'movie', 'rating']], reader)
    trainset = data.build_full_trainset() 
    testset = list(zip(reg_test.user.values, reg_test.movie.values, reg_test.rating.values))

    return trainset, testset

# to get rmse and mape given actual and predicted ratings..
def get_error_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean([ (y_true[i] - y_pred[i])**2 for i in range(len(y_pred)) ]))
    mape = np.mean(np.abs( (y_true - y_pred)/y_true )) * 100
    return rmse, mape

def evaluate_model_XGB(y_true, y_pred, model_name, models_evaluation_test):
    test = {}
    print(f"Evaluating {model_name} model :")
    rmse_test, mape_test = get_error_metrics(y_true, y_pred)
    test['rmse'] = rmse_test
    test['mape'] = mape_test
    test['predictions'] = y_pred
    print('RMSE : ', rmse_test)
    print('MAPE : ', mape_test)
    print('-'*30)
    models_evaluation_test[f"{model_name}"] = test
    
def evaluate_model_SRP(test_preds, model_name, models_evaluation_test):
    test = {}
    print(f"Evaluating {model_name} model :")
    test_rmse, test_mape = get_errors(test_preds)
    test_actual_ratings, test_pred_ratings = get_ratings(test_preds)
    print("RMSE : {}\nMAPE : {}".format(test_rmse, test_mape))
    print('-'*30)
    test['rmse'] = test_rmse
    test['mape'] = test_mape
    test['predictions'] = test_pred_ratings
    
    models_evaluation_test[f"{model_name}"] = test
###################################################################
def run_xgboost(algo, x_train, y_train, x_test, y_test, model_name):
    train_results = dict()
    test_results = dict()
    print(f'Training the {model_name} model...')
    start =datetime.now()
    
    algo.fit(x_train, y_train, eval_metric = 'rmse')
    joblib.dump(algo, model_name + '.joblib')

    print('Done. Time taken : {}'.format(datetime.now()-start))

    y_train_pred = algo.predict(x_train)
    rmse_train, mape_train = get_error_metrics(y_train.values, y_train_pred)
    
    train_results = {'rmse': rmse_train,
                    'mape' : mape_train,
                    'predictions' : y_train_pred}
    
    y_test_pred = algo.predict(x_test) 
    rmse_test, mape_test = get_error_metrics(y_true=y_test.values, y_pred=y_test_pred)
    
    test_results = {'rmse': rmse_test,
                    'mape' : mape_test,
                    'predictions':y_test_pred}
    
    print('Validation Data:-')
    print('RMSE : ', rmse_test)
    print('MAPE : ', mape_test)
    print('-'*30 + '\n')

    return train_results, test_results
    #############################################################################
# it is just to makesure that all of our algorithms should produce same results
# everytime they run...
my_seed = 15
random.seed(my_seed)
np.random.seed(my_seed)

def get_ratings(predictions):
    actual = np.array([pred.r_ui for pred in predictions])
    pred = np.array([pred.est for pred in predictions])
    
    return actual, pred

def get_errors(predictions, print_them=False):

    actual, pred = get_ratings(predictions)
    rmse = np.sqrt(np.mean((pred - actual)**2))
    mape = np.mean(np.abs(pred - actual)/actual)

    return rmse, mape*100
###################################################################
def run_surprise(algo, trainset, testset, model_name): 
    
    train = dict()
    test = dict()
    
    st = datetime.now()
    print(f'Training the {model_name} model...')
    algo.fit(trainset)
    joblib.dump(algo, model_name + '.joblib')
    print('Done. time taken : {} '.format(datetime.now()-st))
    
    # ---------------- Evaluating train data--------------------#
    train_preds = algo.test(trainset.build_testset())
    # get predicted ratings from the train predictions..
    train_actual_ratings, train_pred_ratings = get_ratings(train_preds)
    # get ''rmse'' and ''mape'' from the train predictions.
    train_rmse, train_mape = get_errors(train_preds)
    
    train['rmse'] = train_rmse
    train['mape'] = train_mape
    train['predictions'] = train_pred_ratings
    
    #------------ Evaluating Test data---------------#
    test_preds = algo.test(testset)
    # get the predicted ratings from the list of predictions
    test_actual_ratings, test_pred_ratings = get_ratings(test_preds)
    # get error metrics from the predicted and actual ratings
    test_rmse, test_mape = get_errors(test_preds)
    
    print('Validation Data:-')
    print("RMSE : {}\nMAPE : {}".format(test_rmse, test_mape))
    print('-'*15 + '\n')

    test['rmse'] = test_rmse
    test['mape'] = test_mape
    test['predictions'] = test_pred_ratings
    
    return train, test

def comparing_models(models_evaluation):
    print("-------comparing models : using RMSE--------")
    df_comparison = pd.DataFrame(models_evaluation)
    print(df_comparison.loc['rmse'].sort_values())
    print('-'*30)
