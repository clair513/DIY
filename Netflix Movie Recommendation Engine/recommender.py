## Importing required Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('dark')
plt.rcParams['figure.figsize'] = (15.0, 8.0)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['patch.force_edgecolor'] = True

import os
import random

from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from surprise import Reader, Dataset, BaselineOnly, KNNBaseline, SVD, SVDpp
from surprise.model_selection import GridSearchCV


## Cleansing & compiling CSV dataset:
if not os.path.isfile('./data/netflix_ratings.csv'):
    with open('./data/netflix_ratings.csv', 'w') as data:
        files = ['./data/combined_data_1.txt', './data/combined_data_2.txt', './data/combined_data_3.txt', './data/combined_data_4.txt']
        for i in files:
            with open(i) as f:
                for line in f:
                    line = line.strip()
                    if line.endswith(':'):
                        movieID = line.replace(':','')
                    else:
                        row = []
                        row = [x for x in line.split(',')]
                        row.insert(0, movieID)
                        data.write(','.join(row))
                        data.write('\n')


## Loading combined CSV file & storing a DataFrame copy of it:
if not os.path.isfile('./data/netflix_data.pkl'):
    col_names = ['MovieID','CustID', 'Ratings', 'Date']
    compiled_data = pd.read_csv('./data/netflix_ratings.csv', sep=',', names=col_names)
    compiled_data['Date'] = pd.to_datetime(compiled_data['Date'])
    compiled_data.sort_values(by='Date', inplace=True)
    compiled_data.to_pickle('./data/netflix_data.pkl')
else:
    compiled_data = pd.read_pickle('./data/netflix_data.pkl')


## Splitting Dataset & savig them separately:
if not os.path.isfile('./data/training_data.pkl'):
    compiled_data.iloc[ :int(compiled_data.shape[0]*0.8)].to_pickle('./data/training_data.pkl')
    training_data = pd.read_pickle('./data/training_data.pkl')
    training_data.reset_index(drop=True, inplace=True)
else:
    training_data = pd.read_pickle('./data/training_data.pkl')
    training_data.reset_index(drop=True, inplace=True)

if not os.path.isfile('./data/testing_data.pkl'):
    compiled_data.iloc[int(compiled_data.shape[0]*0.8): ].to_pickle('./data/testing_data.pkl')
    testing_data = pd.read_pickle('./data/testing_data.pkl')
    testing_data.reset_index(drop=True, inplace=True)
else:
    testing_data = pd.read_pickle('./data/testing_data.pkl')
    testing_data.reset_index(drop=True, inplace=True)


## EDA on Training data:
training_data['Day_of_Week'] = training_data.Date.dt.weekday_name
#sns.countplot(x='Ratings', data=training_data)
#training_data.resample('M', on='Date')['Ratings'].count().plot()
movies_rated_per_user = training_data.groupby(by='CustID')['Ratings'].count().sort_values(ascending=False)
#sns.kdeplot(movies_rated_per_user.values, shade=True, cumulative=True)
movie_ratings_count = training_data.groupby(by='MovieID')['Ratings'].count().sort_values(ascending=False)
#plt.plot(movie_ratings_count.values)
#sns.countplot(x='Day_of_Week', data=training_data)
#sns.boxplot(x='Day_of_Week', y='Ratings', data=training_data)
average_ratings_per_day = training_data.groupby(by='Day_of_Week')['Ratings'].mean()


## Creating User-Item sparse matrix for Train & Test data:
if not os.path.isfile('./data/training_sparse_data.npz'):
    training_sparse_data = sparse.csr_matrix((training_data.Ratings, (training_data.CustID, training_data.MovieID)))
    sparse.save_npz('./data/training_sparse_data.npz', training_sparse_data)
else:
    training_sparse_data = sparse.load_npz('./data/training_sparse_data.npz')

if not os.path.isfile('./data/testing_sparse_data.npz'):
    testing_sparse_data = sparse.csr_matrix((testing_data.Ratings, (testing_data.CustID, testing_data.MovieID)))
    sparse.save_npz('./data/testing_sparse_data.npz', testing_sparse_data)
else:
    testing_sparse_data = sparse.load_npz('./data/testing_sparse_data.npz')


# Fetching Average Rating at various levels:
def get_avg_ratings(sparse_matrix, if_user):
    ax = 1 if if_user else 0
    sum_of_ratings = sparse_matrix.sum(axis=ax).A1
    num_of_ratings = (sparse_matrix!=0).sum(axis=ax).A1
    rows,cols = sparse_matrix.shape
    avg_ratings = {i: sum_of_ratings[i]/num_of_ratings[i] for i in range(rows if if_user else cols) if num_of_ratings[i]!=0}
    return avg_ratings

# Ratings based on Global Average, per User & per Movie:
global_avg_rating = training_sparse_data.sum()/training_sparse_data.count_nonzero()
user_avg_rating = get_avg_ratings(training_sparse_data, True)
movie_avg_rating = get_avg_ratings(training_sparse_data, False)
print(f'Global Average Rating is {global_avg_rating}.\nAverage rating of user 50 is {user_avg_rating[50]}.\nAverage rating of movie 3000 is {movie_avg_rating[3000]}.')


# Computing User-User Similarity Matrix (Computationally exhaustive):
# Thus with 'top' param value, limiting number of similar users [Default being 100].
def user_user_similarity(sparse_matrix, top=100):
    row_index, col_index = sparse_matrix.nonzero()
    rows = np.unique(row_index)
    similarity_matrix = np.zeros(61700).reshape(617,100)  #617*100 = 61700
    for i in rows[:top]:
        similarity = cosine_similarity(sparse_matrix.getrow(i), sparse_matrix).ravel()
        similar_indices = similarity.argsort()[-top:]
        similarity_matrix[i] = similarity[similar_indices]
    return similarity_matrix

print(f'Similarity Matrix for Top-100 Users is: \n{user_user_similarity(training_sparse_data)}.')


# Computing & Storing Movie-Movie Similarity Matrix:
if os.path.isfile('./data/m_m_similarity.npz'):
    m_m_similarity = sparse.load_npz('./data/m_m_similarity.npz')
    print(f'Dimension of Matrix: {m_m_similarity.shape}')
else:
    m_m_similarity = cosine_similarity(training_sparse_data.T, dense_output=False)
    print(f'Dimension of Matrix: {m_m_similarity.shape}')
    sparse.save_npz('./data/m_m_similarity.npz', m_m_similarity)

# Checking Top-5 most similar & All similar movies to any random movie:
similar_movies = dict()
movie_id = np.unique(m_m_similarity.nonzero())
for i in movie_id:
    similar = np.argsort(-m_m_similarity[i].toarray().ravel())[1:100]
    similar_movies[i] = similar

movie_titles = pd.read_csv('./data/movie_titles.csv',sep=',', header=None, names=['MovieID','Year_of_Release','Movie_Title'], index_col='MovieID', encoding='iso8859_2')

movie_id = 17765      #Godzilla's Revenge
movie_titles.loc[similar_movies[movie_id][:5]]
all_sim_movies = sorted(m_m_similarity[movie_id].toarray().ravel(), reverse=True)[1:]
print(all_sim_movies[:26])    #Top-25 similar movies


## Creating sample Sparse Matrix for Training & Test datasets:
def fetch_sample_sparse_matrix(sparse_matrix, n_users, n_movies):
    users, movies, ratings = sparse.find(sparse_matrix)
    distinct_users = np.unique(users)
    distinct_movies = np.unique(movies)
    np.random.seed(41)
    user_sparse = np.random.choice(distinct_users, n_users, replace=False)
    movie_sparse = np.random.choice(distinct_movies, n_movies, replace=False)
    mask = np.logical_and(np.isin(users, user_sparse), np.isin(movies, movie_sparse))
    sparse_sample = sparse.csr_matrix((ratings[mask], (users[mask], movies[mask])), shape=(max(user_sparse)+1, max(movie_sparse)+1))
    sparse.save_npz(path, sparse_sample)
    print(f'Shape of Sample Sparse Matrix: {sparse_sample.shape}.'
    return sparse_sample

path = './data/training_sample_sparse_data.npz'
if os.path.isfile(path):
    training_sample_sparse = sparse.load_npz(path)
    print(f'Shape of Training Sample Sparse Matrix: {training_sample_sparse.shape}.')
else:
    training_sample_sparse = fetch_sample_sparse_matrix(training_sparse_data, 5000, 500)

path = './data/test_sample_sparse_data.npz'
if os.path.isfile(path):
    test_sample_sparse = sparse.load_npz(path)
    print(f'Shape of Test Sample Sparse Matrix: {test_sample_sparse.shape}.')
else:
    test_sample_sparse = fetch_sample_sparse_matrix(testing_sparse_data, 2000, 200)


## Transforming Sample Sparse data & preparing Train/Test CSV File [Computationally Exhaustive]:
sample_train_users, sample_train_movies, sample_train_ratings = sparse.find(training_sample_sparse)
if not os.path.isfile('./data/train.csv'):
    print(f'Preparing Train CSV File for {len(sample_train_ratings)} rows.')
    with open('./data/train.csv', 'w') as data:
        count = 0
        for user, movie, rating in zip(sample_train_users,sample_train_movies,sample_train_ratings):
            row = []
            row.append(user)
            row.append(movie)
            row.append(training_sample_sparse.sum()/training_sample_sparse.count_nonzero())

            # Ratings given to 'movie' by Top-5 similar users wrt 'user':
            similar_users = cosine_similarity(training_sample_sparse[user], training_sample_sparse).ravel()
            sim_users_indices = np.argsort(-similar_users)[1:]
            sim_users_ratings = training_sample_sparse[sim_users_indices, movie].toarray().ravel()
            top_sim_usr_ratings = list(sim_users_ratings[sim_users_ratings != 0][:5])
            top_sim_usr_ratings.extend([get_avg_ratings(training_sample_sparse, False)[movie]]*(5-len(top_sim_usr_ratings)))
            row.extend(top_sim_usr_ratings)

            # Ratings given by 'user' to Top-5 similar movies wrt 'movie':
            similar_movies = cosine_similarity(training_sample_sparse[:,movie].T, training_sample_sparse.T).ravel()
            sim_movies_indices = np.argsort(-similar_movies)[1:]
            sim_movies_ratings = training_sample_sparse[user, sim_movies_indices].toarray().ravel()
            top_sim_movie_ratings = list(sim_movies_ratings[sim_movies_ratings != 0][:5])
            top_sim_movie_ratings.extend([get_avg_ratings(training_sample_sparse, True)[user]]*(5-len(top_sim_movie_ratings)))
            row.extend(top_sim_movie_ratings)

            # Appending 'user' average, 'movie' average & rating of 'user' per 'movie':
            row.append([get_avg_ratings(training_sample_sparse, True)[user])
            row.append([get_avg_ratings(training_sample_sparse, False)[movie])
            row.append(rating)

            # Transforming rows to add to file:
            data.write(','.join(map(str,row)))
            data.write('\n')

            count += 1

sample_test_users, sample_test_movies, sample_test_ratings = sparse.find(test_sample_sparse)
if not os.path.isfile('./data/test.csv'):
    print(f'Preparing Test CSV File for {len(sample_test_ratings)} rows.')
    with open('./data/test.csv', 'w') as data:
        count = 0
        for user, movie, rating in zip(sample_test_users,sample_test_movies,sample_test_ratings):
            row = []
            row.append(user)
            row.append(movie)
            row.append(test_sample_sparse.sum()/test_sample_sparse.count_nonzero())

            # Ratings given to 'movie' by Top-5 similar users wrt 'user':
            try:
                similar_users = cosine_similarity(test_sample_sparse[user], test_sample_sparse).ravel()
                sim_users_indices = np.argsort(-similar_users)[1:]
                sim_users_ratings = test_sample_sparse[sim_users_indices, movie].toarray().ravel()
                top_sim_usr_ratings = list(sim_users_ratings[sim_users_ratings != 0][:5])
                top_sim_usr_ratings.extend([get_avg_ratings(test_sample_sparse, False)[movie]]*(5-len(top_sim_usr_ratings)))
                row.extend(top_sim_usr_ratings)
            # In case of a new Movie or a new User:
            except(IndexError, KeyError):
                gAvg_test_rating = [test_sample_sparse.sum()/test_sample_sparse.count_nonzero()]*5
                row.extend(gAvg_test_rating)
            except:
                raise

            # Ratings given by 'user' to Top-5 similar movies wrt 'movie':
            try:
                similar_movies = cosine_similarity(test_sample_sparse[:, movie].T, test_sample_sparse.T).ravel()
                sim_movies_indices = np.argsort(-similar_movies)[1:]
                sim_movies_ratings = test_sample_sparse[user, sim_movies_indices].toarray().ravel()
                top_sim_movie_ratings = list(sim_movies_ratings[sim_movies_ratings != 0][:5])
                top_sim_movie_ratings.extend([get_avg_ratings(test_sample_sparse, True)[user]]*(5-len(top_sim_movie_ratings)))
                row.extend(top_sim_movie_ratings)
            # In case of a new Movie or a new User:
            except(IndexError, KeyError):
                gAvg_test_rating = [test_sample_sparse.sum()/train_sample_sparse.count_nonzero()]*5
                row.extend(gAvg_test_rating)
            except:
                raise

            # Appending 'user' average, 'movie' average & rating of 'user' per 'movie':
            try:
                row.append(get_avg_ratings(test_sample_sparse, True)[user])
            except(KeyError):
                gAvg_test_rating = test_sample_sparse.sum()/test_sample_sparse.count_nonzero()
                row.append(gAvg_test_rating)
            except:
                raise

            try:
                row.append(get_avg_ratings(test_sample_sparse, False)[movie])
            except(KeyError):
                gAvg_test_rating = test_sample_sparse.sum()/test_sample_sparse.count_nonzero()
                row.append(gAvg_test_rating)
            except:
                raise

            row.append(rating)

            # Transforming rows to add to file:
            data.write(','.join(map(str, row)))
            data.write('\n')

            count += 1


## Loading our prepared Traning & Testing CSV files for Regression:
train_data = pd.read_csv('./data/train.csv', names=['User_ID','Movie_ID','Global_Average','SUR1','SUR2','SUR3','SUR4','SUR5','SMR1','SMR2','SMR3','SMR4','SMR5','User_Average','Movie_Average','Rating'])
test_data = pd.read_csv('./data/test.csv', names=['User_ID','Movie_ID','Global_Average','SUR1','SUR2','SUR3','SUR4','SUR5','SMR1','SMR2','SMR3','SMR4','SMR5','User_Average','Movie_Average','Rating'])
print(f'Shape of Training DataFrame: {train_data.shape} & Shape of Training DataFrame: {test_data.shape}.\n')
print(train_data.head())


## Transforming Train & Test data as per Surprise requirements:
reader = Reader(rating_scale=(1, 5))
tr_data = Dataset.load_from_df(train_data[['User_ID','Movie_ID','Rating']], reader)
trainset = tr_data.build_full_trainset()
testset = list(zip(test_data['User_ID'].values, test_data['Movie_ID'].values, test_data['Rating'].values))

def get_ratings(predictions):
    actual = np.array([pred.r_ui for pred in predictions])
    predicted = np.array([pred.est for pred in predictions])
    return actual, predicted

def get_error(predictions):
    actual, predicted = get_ratings(predictions)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(abs((actual - predicted)/actual))*100
    return rmse, mape


## Error computation utilities for XGBoost model:
def error_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(abs((y_true - y_pred)/y_true))*100
    return rmse, mape

error_table = pd.DataFrame(columns= ['Model', 'Train RMSE', 'Train MAPE', 'Test RMSE', 'Test MAPE'])
train_evaluation = dict()
test_evaluation = dict()

def make_table(model_name, rmse_train, mape_train, rmse_test, mape_test):
    global error_table
    error_table = error_table.append(pd.DataFrame([[model_name, rmse_train, mape_train, rmse_test, mape_test]], columns= ['Model','Train RMSE','Train MAPE','Test RMSE','Test MAPE']))
    error_table.reset_index(drop=True, inplace=True)

def train_test_xgboost(x_train, x_test, y_train, y_test, model_name):
    train_result = dict()
    test_result = dict()

    clf = xgb.XGBRegressor(n_estimators=100, silent=False, n_jobs=-1)
    clf.fit(x_train, y_train)

    y_pred_train = clf.predict(x_train)
    rmse_train, mape_train = error_metrics(y_train, y_pred_train)
    print(f'RMSE: {rmse_train}')
    print(f'MAPE: {mape_train}')
    print('-' * 50)
    train_result = {'RMSE':rmse_train, 'MAPE':mape_train, 'Prediction':y_pred_train}

    rmse_test, mape_test = error_metrics(y_test, y_pred_test)
    print(f'RMSE: {rmse_test}')
    print(f'MAPE: {mape_test}')
    print('-' * 50)
    test_result = {'RMSE':rmse_test, 'MAPE':mape_test, 'Prediction':y_pred_test}

    make_table(model_name, rmse_train, mape_train, rmse_test, mape_test)
    return train_result, test_result


# Defining Surprise Model:
my_seed = 41
random.seed(my_seed)
np.random.seed(my_seed)

def run_surprise(algo, trainset, testset, model_name):
    train = dict()
    test = dict()
    algo.fit(trainset)

    # Evaluating Train data:
    train_pred = algo.test(trainset.build_testset())
    train_actual, train_predicted = get_ratings(train_pred)
    train_rmse, train_mape = get_error(train_pred)
    print(f'RMSE: {train_rmse}')
    print(f'MAPE: {train_mape}')
    print('-' * 50)
    train = {'RMSE':train_rmse, 'MAPE':train_mape, 'Prediction':train_predicted}

    # Evaluating Test data:
    test_pred = algo.test(testset)
    test_actual, test_predicted = get_ratings(test_pred)
    test_rmse, test_mape = get_error(test_pred)
    print(f'RMSE: {test_rmse}')
    print(f'MAPE: {test_mape}')
    print('-' * 50)
    test = {'RMSE':train_rmse, 'MAPE':train_mape, 'Prediction':train_predicted}

    make_table(model_name, train_rmse, train_mape, test_rmse, test_mape)
    return train, test


## Executing XGBoost_13 Model:
x_train = train_data.drop(['User_ID','Movie_ID','Rating'], axis=1)
x_test = test_data.drop(['User_ID','Movie_ID','Rating'], axis=1)
y_train = train_data['Rating']
y_test = test_data['Rating']

train_result, test_result = train_test_xgboost(x_train, x_test, y_train, y_test, 'XGBoost_13')
train_result = model_train_evaluation['XGBoost_13']
test_result = model_test_evaluation['XGBoost_13']


## Executing Surprise BaselineOnly Model:
bsl_options = {'method':'sgd','learning_rate':0.01, 'n_epochs':50}
algo = BaselineOnly(bsl_options=bsl_options)

train_result, test_result = run_surprise(algo, trainset, testset, 'BaselineOnly')
train_result = model_train_evaluation['BaselineOnly']
test_result = model_test_evaluation['BaselineOnly"]


## Executing XGBoost_13 + Surprise BaselineOnly Model:
train_data['BaselineOnly'] = model_train_evaluation['BaselineOnly']['Prediction']
x_train = train_data.drop(['User_ID','Movie_ID','Rating'], axis=1)
x_test = test_data.drop(['User_ID','Movie_ID','Rating'], axis=1)
y_train = train_data['Rating']
y_test = test_data['Rating']

train_result, test_result = train_test_xgboost(x_train, x_test, y_train, y_test, 'XGB_BSL')
train_result = model_train_evaluation['XGB_BSL']
test_result = model_test_evaluation['XGB_BSL']
