from loadData import getTrainDictionaries
import random
import numpy as np
import math
from scipy.spatial import distance
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle


labelsTrainFile = "Task2_user_reputation_prediction/Reputation_prediction_labels_train.txt"
dataTrainFile = "Task2_user_reputation_prediction/Reputation_prediction_data_train.txt"
cachingFile = 'cachedMatrix.csv'

userToReps, postsMatrix = getTrainDictionaries(labelsTrainFile, dataTrainFile, cachingFile)
postsMatrix = postsMatrix.values

num_values = len(postsMatrix)
num_dim = len(postsMatrix[0])

X = postsMatrix[:, :num_dim - 1]
y = postsMatrix[:, num_dim - 1]

kf = KFold(n_splits=5, shuffle = True)
kf.get_n_splits(postsMatrix)

for train_index, test_index in kf.split(X):

	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	

	knn = KNeighborsClassifier(n_neighbors=5)

	lin_dis = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='lsqr', store_covariance=True, tol=0.0000000001)

	nb = GaussianNB()

	rf = RandomForestClassifier(n_estimators=100, max_depth = 50)

	model = VotingClassifier(estimators=[('knn', knn), ('lin_dis', lin_dis), ('nb', nb), ('rf', rf)], voting='soft')
	model.fit(X_train, y_train)
	pickle.dump(model, open('./model.sav', 'wb'))
