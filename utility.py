import joblib
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from surprise import Dataset, Reader
import random
from scipy import sparse

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def sort_datetime(df):
    print("sorting datetime values___________")
    df.date = pd.to_datetime(df.date)
    df.sort_values(by = 'date', inplace = True)
    return df

def to_sparse_matrix(df):
    print("converting dataframe to sparse matrix__________")
    df = csr_matrix((df.rating.values, (df.user.values, df.movie.values)))
    return df

def clean_data(df):
    print("cleaning data____________")
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
    print("adding average rating values_____________________")
    averages['global'] = sparse_matrix.sum()/ sparse_matrix.count_nonzero()
    averages['user'] = get_average_ratings(sparse_matrix, of_users=True)
    averages['movie'] =  get_average_ratings(sparse_matrix, of_users=False)

def get_sample_sparse_matrix(sparse_matrix, no_users, no_movies):
    
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

    print("________________________________________________________________________________________")
    print("Sampled Matrix : (users, movies) -- ({} {})".format(len(sample_users), len(sample_movies)))
    print("Sampled Matrix : Ratings --", format(ratings[mask].shape[0]))
    
    return sample_sparse_matrix

def surprise_data_creation(reg_data, type):
    print("creating data for surprise model")
    if (type == 'train'):
        reader = Reader(rating_scale=(1,5))    
        data = Dataset.load_from_df(reg_data[['user', 'movie', 'rating']], reader)
        dataset = data.build_full_trainset() 
    else:
        dataset = list(zip(reg_data.user.values, reg_data.movie.values, reg_data.rating.values))

    return dataset

# to get rmse and mape given actual and predicted ratings..
def get_error_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean([ (y_true[i] - y_pred[i])**2 for i in range(len(y_pred)) ]))
    mape = np.mean(np.abs( (y_true - y_pred)/y_true )) * 100
    return rmse, mape

def evaluate_model_XGB(y_true, y_pred, model_name, models_evaluation_test):
    test = {}
    print(f"evaluating {model_name}" + '-'*30)
    rmse_test, mape_test = get_error_metrics(y_true, y_pred)
    test['rmse'] = rmse_test
    test['mape'] = mape_test
    test['predictions'] = y_pred
    print('RMSE : ', rmse_test)
    print('MAPE : ', mape_test)

    models_evaluation_test[f"{model_name}"] = test
    
def evaluate_model_SRP(test_preds, model_name, models_evaluation_test):
    test = {}
    print(f"evaluating {model_name}" + '-'*30)
    test_rmse, test_mape = get_errors(test_preds)
    test_pred_ratings = get_ratings(test_preds)
    print("RMSE : {}\n\nMAPE : {}\n".format(test_rmse, test_mape))
    test['rmse'] = test_rmse
    test['mape'] = test_mape
    test['predictions'] = test_pred_ratings
    
    models_evaluation_test[f"{model_name}"] = test
###################################################################
def run_xgboost(models, algo, x_train, y_train, x_test, y_test, model_name):    
    test_results = dict()
    print('Training the model..')
    start =datetime.now()
    algo.fit(x_train, y_train, eval_metric = 'rmse')
    joblib.dump(algo, model_name + '.joblib')
    models.append(algo)
    print('Done. Time taken : {}\n'.format(datetime.now()-start))
    print('Done \n')

    print('Evaluating validation data')
    y_test_pred = algo.predict(x_test) 
    rmse_test, mape_test = get_error_metrics(y_true=y_test.values, y_pred=y_test_pred)
    # store them in our test results dictionary.
    test_results = {'rmse': rmse_test,
                    'mape' : mape_test,
                    'predictions':y_test_pred}
    print('-'*30)
    print('RMSE : ', rmse_test)
    print('MAPE : ', mape_test)
        
    return test_results
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

def run_surprise(models, algo, trainset, model_name):
    train = dict()
    
    st = datetime.now()
    print('Training the model...')
    algo.fit(trainset)
    joblib.dump(algo, model_name + '.joblib')
    models.append(algo)
    print('Done. time taken : {} \n'.format(datetime.now()-st))
    
    train_preds = algo.test(trainset.build_testset())
    
    train_actual_ratings, train_pred_ratings = get_ratings(train_preds)
    
    train_rmse, train_mape = get_errors(train_preds)
    
    train['rmse'] = train_rmse
    train['mape'] = train_mape
    train['predictions'] = train_pred_ratings
    
    return train
#########################################################

def comparing_models(models_evaluation):
    print("comparing models_____________")
    df_comparison = pd.DataFrame(models_evaluation)
    print(df_comparison.loc['rmse'].sort_values())

