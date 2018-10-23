import random
import numpy as np
import math
from scipy.spatial import distance
from loadData import getDictionaries
#from load_data import get_CIFAR10_data


# userToReps is a dictionary of users to their reputations
# postsMatrix is 2d array indexed by number for each post
userToReps, postsMatrix = getDictionaries()

print(userToReps)






# You can change these numbers for experimentation
# For submission we will use the default values 
TRAIN_IMAGES = 49000
VAL_IMAGES = 1000
TEST_IMAGES = 5000


data = get_CIFAR10_data(TRAIN_IMAGES, VAL_IMAGES, TEST_IMAGES)
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']