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
