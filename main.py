import random
import numpy as np
import math
from scipy.spatial import distance
from loadData import getDictionaries
from KNN import *
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle


# userToReps is a dictionary of users to their reputations
# postsMatrix is 2d array indexed by number for each post
userToReps, postsMatrix = getDictionaries()

# print(postsMatrix.shape)
postsMatrix = (postsMatrix.values)
# print(postsMatrix.shape)

num_values = len(postsMatrix)
num_dim = len(postsMatrix[0])

X = postsMatrix[:, :num_dim - 1]#[:5000]
y = postsMatrix[:, num_dim - 1]#[:5000]


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
	# print(pred_knn)
	knn = KNeighborsClassifier(n_neighbors=5)
	# knn.fit(X_train, y_train)
	# knn_pred = knn.predict(X_test)
	# knn_acc = accuracy_score(y_test, knn_pred)
	# print(knn_acc)
	# pickle.dump(knn, open('./saved_models/knn.sav', 'wb'))


	lin_dis = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
              solver='lsqr', store_covariance=True, tol=0.0000000001)
	# lin_dis.fit(X_train, y_train)
	# pred_lin_dis = lin_dis.predict(X_test)
	# print(pred_lin_dis)
	# lin_dis_acc = accuracy_score(y_test, pred_lin_dis)
	# print(lin_dis_acc)
	# pickle.dump(lin_dis, open('./saved_models/lin_dis.sav', 'wb'))



	# svc = SVC(random_state = 0)
	# svc.fit(X_train, y_train)
	# pred_svc = svc.predict(X_test)
	# # print(pred_lin_svc)
	# svc_acc = accuracy_score(y_test, pred_svc)
	# print(svc_acc)


	nb = GaussianNB()
	# nb.fit(X_train, y_train)
	# pred_nb = nb.predict(X_test)
	# nb_acc = accuracy_score(y_test, pred_nb)
	# print(nb_acc)
	# pickle.dump(nb, open('./saved_models/nb.sav', 'wb'))



	rf = RandomForestClassifier(n_estimators=100, max_depth = 50)
	# rf.fit(X_train, y_train)
	# pred_rf = rf.predict(X_test)
	# rf_acc = accuracy_score(y_test, pred_rf)
	# print(rf_acc)
	# pickle.dump(rf, open('./saved_models/rf.sav', 'wb'))

	# num_not_1 = 0
	# for i in pred_rf:
	# 	if i != 1:
	# 		num_not_1 +=1
	# print(num_not_1)

	# model = VotingClassifier(estimators=[('knn', knn), ('lin_dis', lin_dis), ('svc', svc), ('nb', nb), ('rf', rf)], voting='soft')
	model = VotingClassifier(estimators=[('knn', knn), ('lin_dis', lin_dis), ('nb', nb), ('rf', rf)], voting='soft')
	pickle.dump(model, open('./model.sav', 'wb'))

	# model.fit(X_train, y_train)
	# pred = model.predict(X_test)
	# acc = accuracy_score(y_test, pred)
	# print("accuracy_score is " + str(acc))
	# num_not_1 = 0
	# for i in pred_svc:
	# 	if i != 1:
	# 		num_not_1 +=1
	# print(num_not_1)

	# acc = get_acc(pred_knn, y_test)
	# print('The validation of %d accuracy is given by : %f' % (k, acc))
# 	avg_acc += acc
# avg_acc = avg_acc/5
# print(avg_acc)
# print('The average validation of %d accuracy is given by : %f' % (k, avg_acc))



