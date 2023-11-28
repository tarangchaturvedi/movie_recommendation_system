import warnings
from utility import read_csv, clean_data, sort_datetime, to_sparse_matrix, adding_average_ratings, comparing_models, get_sample_sparse_matrix, evaluate_model_XGB, evaluate_model_SRP
from data_featurizing import data_featurizing

def test_model(file_path, model_path):
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
    
    x_test = reg_test.drop(['user','movie','rating'], axis=1)
    y_test = reg_test['rating']
    
    test_predictions = {}
    if('X' in model_path):
        evaluate_model_XGB(x_test,y_test, model_path)
    else:
        testset = list(zip(reg_test.user.values, reg_test.movie.values, reg_test.rating.values))
        evaluate_model_SRP(testset, model_path)
