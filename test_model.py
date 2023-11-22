import warnings
from model_training import train_models
from sklearn.model_selection import train_test_split
from utility import read_csv, clean_data, sort_datetime, surprise_data_creation, to_sparse_matrix, adding_average_ratings, comparing_models
from data_featurizing import data_featurizing
from models_testing import test_models

def test_models(file_path):
    warnings.simplefilter(action='ignore')    
    df = read_csv(file_path)

    df = clean_data(df)
    df = sort_datetime(df)

    sparse_matrix = to_sparse_matrix(df)
    
    sample_averages = {}
    adding_average_ratings(sparse_matrix, sample_averages)

    reg_data = data_featurizing(sparse_matrix, sample_averages)
    
    surprise_data = surprise_data_creation(reg_data, type = 'test')
    models_evaluation_test = {}
    
    test_models(reg_data, models_evaluation_test, surprise_data)

    comparing_models(models_evaluation_test)