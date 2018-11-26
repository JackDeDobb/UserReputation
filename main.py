import random
import numpy as np
import math
from scipy.spatial import distance
from loadData import getDictionaries
from KNN import *
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsRegressor


# userToReps is a dictionary of users to their reputations
# postsMatrix is 2d array indexed by number for each post
userToReps, postsMatrix = getDictionaries()

# print(postsMatrix.shape)
postsMatrix = (postsMatrix.values)
# print(postsMatrix.shape)

num_values = len(postsMatrix)
num_dim = len(postsMatrix[0])

X = postsMatrix[:, :num_dim - 1]
y = postsMatrix[:, num_dim - 1]


def get_acc(pred, y_test):
    return np.sum(y_test==pred)/len(y_test)*100

kf = KFold(n_splits=5, shuffle = True)
kf.get_n_splits(postsMatrix)

k = 100
	
# for k in k_values:
avg_acc = 0.0
for train_index, test_index in kf.split(X):

	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	# knn = KNN(k)
	# knn.train(X_train, y_train)
	# pred_knn = knn.predict(X_test)	#eventually write this to a file 
	# knn = KNeighborsRegressor(n_neighbors=100)
	# knn.fit(X_train, y_train)
	# knn_pred = knn.predict(X_test)
	# # print(knn.score(X_test, y_test))
	# print(knn_pred)


	reg = LinearRegression()
	reg.fit(X_train, y_train)
	pred_reg = reg.predict(X_test)
	print(pred_reg)
	print(reg.score(X_test, y_test))

	# clf = LinearDiscriminantAnalysis()
	# clf.fit(X_train, y_train)
	# pred_clf = clf.predict(X_test)

	# model = VotingClassifier(estimators=[('knn', knn), ('linreg', reg), ('lindis', clf)], voting='hard')
	# model.fit(X_train, y_train)
	# # model.predict(X_test)
	# # model.score(X_test, y_test)

	# acc = get_acc(pred_knn, y_test)
	# print('The validation of %d accuracy is given by : %f' % (k, acc))
	# avg_acc += acc
# avg_acc = avg_acc/2
# print('The average validation of %d accuracy is given by : %f' % (k, avg_acc))



