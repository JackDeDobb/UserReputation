import random
import numpy as np
import math
from scipy.spatial import distance
from loadData import getDictionaries
from KNN import *
#from load_data import get_CIFAR10_data


# userToReps is a dictionary of users to their reputations
# postsMatrix is 2d array indexed by number for each post
userToReps, postsMatrix = getDictionaries()

print(postsMatrix.shape)
postsMatrix = postsMatrix.values

print(len(postsMatrix))

X_train = postsMatrix[: math.floor(95803*.8), :3]
y_train = postsMatrix[: math.floor(95803*.8), 3]
X_val = postsMatrix[math.floor(95803*.8) : , :3]
y_val = postsMatrix[math.floor(95803*.8) : , 3]

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

def get_acc(pred, y_test):
    return np.sum(y_test==pred)/len(y_test)*100

k_values = np.array([1])
for k in k_values:
    knn = KNN(k)
    knn.train(X_train, y_train)

    pred_knn = knn.predict(X_val)
    print('The validation of %d accuracy is given by : %f' % (k, get_acc(pred_knn, y_val)))








# You can change these numbers for experimentation
# For submission we will use the default values 
# TRAIN_IMAGES = 49000
# VAL_IMAGES = 1000
# TEST_IMAGES = 5000


# data = get_CIFAR10_data(TRAIN_IMAGES, VAL_IMAGES, TEST_IMAGES)
# X_train, y_train = data['X_train'], data['y_train']
# X_val, y_val = data['X_val'], data['y_val']
# X_test, y_test = data['X_test'], data['y_test']