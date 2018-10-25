import random
import numpy as np
import math
from scipy.spatial import distance
from loadData import getDictionaries
from KNN import *
from sklearn.model_selection import KFold # import KFold


# userToReps is a dictionary of users to their reputations
# postsMatrix is 2d array indexed by number for each post
userToReps, postsMatrix = getDictionaries()

# print(postsMatrix.shape)
postsMatrix = (postsMatrix.values)#[:10000]
# print(postsMatrix.shape)

num_values = len(postsMatrix)
num_dim = len(postsMatrix[0])

X = postsMatrix[:, :num_dim - 1]
y = postsMatrix[:, num_dim - 1]

# X_train = postsMatrix[: math.floor(num_values*.8), : num_dim - 1]
# y_train = postsMatrix[: math.floor(num_values*.8), num_dim - 1]
# X_test = postsMatrix[math.floor(num_values*.8) : , : num_dim -1]
# y_test = postsMatrix[math.floor(num_values*.8) : , num_dim - 1]

# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

def get_acc(pred, y_test):
    return np.sum(y_test==pred)/len(y_test)*100

kf = KFold(n_splits=20, shuffle = True)
kf.get_n_splits(postsMatrix)

k_values = np.array([1])
	
for k in k_values:
	avg_acc = 0
	for train_index, test_index in kf.split(X):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
	
		knn = KNN(k)
		knn.train(X_train, y_train)

		pred_knn = knn.predict(X_test)
		acc = get_acc(pred_knn, y_test)
		print('The validation of %d accuracy is given by : %f' % (k, acc))
		avg_acc += acc
	avg_acc = avg_acc/20
	print('The average validation of %d accuracy is given by : %f' % (k, avg_acc))



