import numpy as np
import warnings
from model_training import train_models
from sklearn.model_selection import train_test_split
from utility import read_csv, clean_data, sort_datetime, get_sample_sparse_matrix, adding_average_ratings, surprise_data_creation, comparing_models, to_sparse_matrix, get_error_metrics
from data_featurizing import data_featurizing
from sklearn.metrics import mean_squared_error

def train_model(file_path):
    warnings.simplefilter(action='ignore')    
    df = read_csv(file_path)

    df = clean_data(df)
    df = sort_datetime(df)

    sparse_matrix = to_sparse_matrix(df)

    #choose sample data set: no_users and no_movies
    no_users, no_movies = 20000, 2000
    sample_sparse_matrix = get_sample_sparse_matrix(sparse_matrix, no_users, no_movies)

    sample_averages = {}
    adding_average_ratings(sample_sparse_matrix, sample_averages)

    reg_data = data_featurizing(sample_sparse_matrix, sample_averages)
    reg_train, reg_test = train_test_split(reg_data, test_size = 0.2, random_state=42)
    trainset, testset = surprise_data_creation(reg_train, reg_test)
    models = train_models(reg_train, reg_test, trainset, testset)

    comparing_models(models)
    ensembled_predictions = np.zeros(reg_test.shape[0])

    for model in models:
        ensembled_predictions = ensembled_predictions + models[model]['predictions']
    
    ensembled_predictions = ensembled_predictions*(1.0/6.0)
    ensembled_rmse, ensembled_mape = get_error_metrics(reg_test['rating'].values, ensembled_predictions)

    print("-----Evaluation Report on Validation Data:------")
    print('-'*30)
    print("RMSE: ", ensembled_rmse)
    print("MAPE :", ensembled_mape)
    print('-'*30)
