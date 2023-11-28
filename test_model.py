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

def evaluate_model_XGB(x_test,y_test, model_path):
    model = joblib.load(model_path)
    #print(f"Evaluating {model_name} model :")
    print("------Evaluation Report on Test Data:------")
    print('-'*30)
    y_pred = model.predict(x_test)
    #test_predicitons[f"{model_name}"] = y_pred
    rmse_test, mape_test = get_error_metrics(y_test, y_pred)
    print('RMSE : ', rmse_test)
    print('MAPE : ', mape_test)
    print('-'*30)
    
def evaluate_model_SRP(testset, model_path):
    model = joblib.load(model_path)
    #print(f"Evaluating {model_name} model :")
    print("------Evaluation Report on Test Data:------")
    print('-'*30)
    test_preds = model.test(testset)
    test_rmse, test_mape = get_errors(test_preds)
    test_actual_ratings, test_pred_ratings = get_ratings(test_preds)
    #test_predicitons[f"{model_name}"] = test_pred_ratings
    print("RMSE : {}\nMAPE : {}".format(test_rmse, test_mape))
    print('-'*30)

###################################################################
def run_xgboost(algo, x_train, y_train, x_test, y_test, model_name, models):
    model_name.upper()
    print(f'Training the {model_name} model...')
    start =datetime.now()
    path = model_name + '.joblib'

    algo.fit(x_train, y_train, eval_metric = 'rmse')
    joblib.dump(algo, path)
    print('Done. Time taken : {}'.format(datetime.now()-start))
    print('-'*30)

    y_pred = algo.predict(x_test)
    rmse, mape = get_error_metrics(y_test.values, y_pred)
    model_dict = {'name': model_name, 'model_algo': algo,
                   'path': path, 'rmse': rmse, 'mape': mape, 'predictions': y_pred}
    
    models[f"{model_name}"] = model_dict
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

def get_errors(predictions):

    actual, pred = get_ratings(predictions)
    rmse = np.sqrt(np.mean((pred - actual)**2))
    mape = np.mean(np.abs(pred - actual)/actual)

    return rmse, mape*100
###################################################################
def run_surprise(algo, trainset, testset, model_name, models):
    model_name.upper()
    st = datetime.now()
    path = model_name + '.joblib'
    print(f'Training the {model_name} model...')
    algo.fit(trainset)
    joblib.dump(algo, model_name + '.joblib')
    print('Done. time taken : {} '.format(datetime.now()-st))
    print('-'*30)
    
    test_preds = algo.test(testset)
    y_pred = np.array([pred.est for pred in test_preds])
    rmse, mape = get_errors(test_preds)
    model_dict = {'name': model_name, 'model_algo': algo,
                   'path': path, 'rmse': rmse, 'mape': mape, 'predictions': y_pred}
    
    models[f"{model_name}"] = model_dict
####################################################################
def comparing_models(models):
    print("-------comparing models : using RMSE--------")
    df_comparison = pd.DataFrame.from_dict(models, orient='index')
    print(df_comparison['rmse'])
    df_comparison.to_csv('prediction_results.csv', index = False)
    print('-'*30)
