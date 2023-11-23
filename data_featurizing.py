import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

def data_featurizing(sample_sparse_matrix, sample_averages):
    sample_users, sample_movies, sample_ratings = sparse.find(sample_sparse_matrix)

    print('--------preparing {} tuples for the dataset--------'.format(len(sample_ratings)))
    with open('reg_data.csv', mode='w') as reg_data_file:
        count = 0
        st = datetime.now()
        for (user, movie, rating)  in zip(sample_users, sample_movies, sample_ratings):
        
            #--------------------- Ratings of "movie" by similar users of "user" --------------------
            user_sim = cosine_similarity(sample_sparse_matrix[user], sample_sparse_matrix).ravel()
            top_sim_users = user_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.
            # get the ratings of most similar users for this movie
            top_ratings = sample_sparse_matrix[top_sim_users, movie].toarray().ravel()
            # we will make it's length "5" by adding movie averages to .
            top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5])
            top_sim_users_ratings.extend([sample_averages['movie'][movie]]*(5 - len(top_sim_users_ratings)))
        
            #--------------------- Ratings by "user"  to similar movies of "movie" ---------------------
            movie_sim = cosine_similarity(sample_sparse_matrix[:,movie].T, sample_sparse_matrix.T).ravel()
            top_sim_movies = movie_sim.argsort()[::-1][1:] # we are ignoring 'The User' from its similar users.
            # get the ratings of most similar movie rated by this user..
            top_ratings = sample_sparse_matrix[user, top_sim_movies].toarray().ravel()
            # we will make it's length "5" by adding user averages to.
            top_sim_movies_ratings = list(top_ratings[top_ratings != 0][:5])
            top_sim_movies_ratings.extend([sample_averages['user'][user]]*(5-len(top_sim_movies_ratings))) 

            #-----------------prepare the row to be stores in a file-----------------#
            row = list()
            row.append(user)
            row.append(movie)
        
            row.append(sample_averages['global']) # first feature
            # next 5 features are similar_users "movie" ratings
            row.extend(top_sim_users_ratings)
            # next 5 features are "user" ratings for similar_movies
            row.extend(top_sim_movies_ratings)
            # Avg_user rating
            row.append(sample_averages['user'][user])
            # Avg_movie rating
            row.append(sample_averages['movie'][movie])
            # finalley, The actual Rating of this user-movie pair...
            row.append(rating)
            count = count + 1

            reg_data_file.write(','.join(map(str, row)))
            reg_data_file.write('\n')        
            if (count)%10000 == 0:
                print("Done for {} rows----- {}".format(count, datetime.now() - st))

    print("-------reading from csv file--------")
    print('-'*30 + '\n')
    reg_data = pd.read_csv('reg_data.csv', names = ['user', 'movie', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4', 'sur5','smr1', 'smr2', 'smr3', 'smr4', 'smr5', 'UAvg', 'MAvg', 'rating'], header=None)

    return reg_data
