import warnings
from model_training import train_models
from sklearn.model_selection import train_test_split
from utility import read_csv, clean_data, sort_datetime, to_sparse_matrix, adding_average_ratings, comparing_models, get_sample_sparse_matrix
from data_featurizing import data_featurizing
from models_testing import test_models

def test_model(file_path):
    warnings.simplefilter(action='ignore')    
    df = read_csv(file_path)

    df = clean_data(df)
    df = sort_datetime(df)

    sparse_matrix = to_sparse_matrix(df)
    
    #choose sample data set: no_users and no_movies
    no_users, no_movies = 1000, 100
    sample_sparse_matrix = get_sample_sparse_matrix(sparse_matrix, no_users, no_movies)
    sample_averages = {}
    adding_average_ratings(sample_sparse_matrix, sample_averages)

    reg_test = data_featurizing(sample_sparse_matrix, sample_averages)
    
    testset = list(zip(reg_test.user.values, reg_test.movie.values, reg_test.rating.values))
    
    models_evaluation_test = {}

    test_models(reg_test, models_evaluation_test, testset)

    comparing_models(models_evaluation_test)
