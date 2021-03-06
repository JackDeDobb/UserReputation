from loadData import getTrainDictionaries
import random
import numpy as np
import math
from scipy.spatial import distance
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
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

kf = KFold(n_splits=25, shuffle = True)
kf.get_n_splits(postsMatrix)

total_verified_acc = 0
for train_index, test_index in kf.split(X):

	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]


	knn = KNeighborsClassifier(n_neighbors=85)


	lin_dis = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='svd', store_covariance=True, tol=0.000000000001)


	rf = RandomForestClassifier(n_estimators=150, max_depth = 10)


	model = VotingClassifier(estimators=[('knn', knn), ('lin_dis', lin_dis), ('rf', rf)], voting='soft')
	model.fit(X_train, y_train)
	pred = model.predict(X_test)
	total_verified_acc += accuracy_score(y_test, pred)
	pickle.dump(model, open('./model.sav', 'wb'))
average_verified_acc = total_verified_acc/25
print("Verified accuracy of the model is " + str(average_verified_acc))
