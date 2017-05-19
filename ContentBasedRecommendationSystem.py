# coding: utf-8

# Recommendation systems
# A content-based recommendation algorithm.

from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile


def tokenize_string(my_string):
    """ 
	Tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.
    """
    ###TODO

    movies['tokens'] = [tokenize_string(genre) for genre in movies['genres']]

    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term.
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i
    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int.
    """
    ###TODO

    #creating a vocab of all the unique genres
    vocab = {movie_tokens:idx for idx, movie_tokens in enumerate(sorted(np.unique(np.concatenate(movies.tokens))))}

    # creating df
    df = defaultdict(int)
    for movie_genre in movies.tokens:
        for genre in vocab:
            if genre in movie_genre:
                df[genre]+=1


    #print(sorted(df.items(), key = lambda x: -x[1]))

    #for every movie how many times the genre appears

    all_csr = []
    for idx, movie in enumerate(movies.tokens):
        #print(movie)
        colmn, data, row = [], [], []
        tf = Counter(movie)     # tf
        max_k = tf.most_common(1)[0][1]
        #print(max_k)# max_k
        for genre, freq in tf.items():
            if genre in vocab:
                #row.append(0)
                colmn.append(vocab[genre])
                data.append((freq/max_k)*math.log10(len(movies)/df[genre])) # tf-idf
                X = csr_matrix((np.asarray(data), (np.zeros(shape=(len(data))), np.asarray(colmn))), shape=(1, len(vocab)))

        all_csr.append(X)

    movies['features'] = all_csr

    #print(movies['features'])

    #print(movies.features.head())


    return movies, vocab
    #pass


def train_test_split(ratings):
    """
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    #print(a, b)
    #print(a.shape, b.shape)
    #print(np.dot(a, b.T))
    #print(np.sum(np.square(a)), np.sum(np.square(b)))
    a = a.toarray()
    b = b.toarray()
    return (np.dot(a,b.T)) / (np.sqrt(np.sum(np.square(a))) * np.sqrt(np.sum(np.square(b))))



def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.
    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.
    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.
    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO

    # for every user in Test Set, get the rating from the Train Set
    predictions = []
    for test_userid, test_movieid in zip(ratings_test.userId, ratings_test.movieId):
        # got the test userid & test movieid
        #print("Getting for", test_userid, test_movied)
        weight_ratings = []
        weights = []
        target_user_ratings = []
        for idx, train_user in ratings_train.loc[ratings_train.userId == test_userid, 'movieId': 'rating'].iterrows():
            # got the ratings and movieId for the test userId
            # print(rating_val.movieId, rating_val.rating)
            # print(int(train_user.movieId), int(test_movieid))
            # print(movies.loc[movies.movieId == int(train_user.movieId)].features.values)
            # print(movies.loc[movies.movieId == int(test_movieid)].features.values)

            cos_sim_weight = cosine_sim(movies.loc[movies.movieId == int(train_user.movieId)].features.values[0],
                                        movies.loc[movies.movieId == int(test_movieid)].features.values[0])
            #print(cos_sim_weight)
            weight_ratings.append(train_user.rating * cos_sim_weight)
            weights.append(cos_sim_weight)
            target_user_ratings.append(train_user.rating)


        if np.count_nonzero(weights) > 0:
            #weighted_average = np.sum(weight_ratings)/np.sum(weights)
            predictions.append(np.sum(weight_ratings)/np.sum(weights))
            #print(np.sum(weights))
            #print(weighted_average)
        else:
            predictions.append(ratings_train.loc[ratings_train.userId == test_userid, 'rating'].mean())
            #predictions.append(np.mean(target_user_ratings))

            #print(ratings_train.loc[ratings_train.userId == test_userid, 'rating'].mean())



    return np.asarray(predictions)



def mean_absolute_error(predictions, ratings_test):
    """
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()