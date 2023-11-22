import warnings
from model_training import train_models
from sklearn.model_selection import train_test_split
from utility import read_csv, clean_data, sort_datetime, get_sample_sparse_matrix, adding_average_ratings, surprise_data_creation, comparing_models, to_sparse_matrix
from data_featurizing import data_featurizing


def train_model(file_path):
    warnings.simplefilter(action='ignore')    
    df = read_csv(file_path)

    df = clean_data(df)
    df = sort_datetime(df)

    sparse_matrix = to_sparse_matrix(df)

    #choose sample data set: no_users and no_movies
    no_users, no_movies = 10000, 1000
    sample_sparse_matrix = get_sample_sparse_matrix(sparse_matrix, no_users, no_movies)

    sample_averages = {}
    adding_average_ratings(sample_sparse_matrix, sample_averages)

    reg_data = data_featurizing(sample_sparse_matrix, sample_averages)
    #reg_train, reg_test = train_test_split(reg_data,test_size=0.2
    
    surprise_data = surprise_data_creation(reg_data, type = 'train')

    models_evaluation = {}
    train_models(reg_data, models_evaluation, surprise_data)

    comparing_models(models_evaluation)