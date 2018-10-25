import numpy as np
import scipy
from collections import Counter

class KNN():
    def __init__(self, k):
        """
        Initializes the KNN classifier with the k.
        """
        self.k = k
        # self.train_data_ = None
        # self.labels = None
    
    def train(self, X, y):

        self.train_data_ = X
        self.labels = y
    
    def find_dist(self, X_test):
        """
        Returns :
        - dist_ : Distances between each test point and training point
        """
        dist_ = scipy.spatial.distance.cdist(X_test, self.train_data_, 'euclidean')
        return dist_
    
    def predict(self, X_test):
        """
        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        def predict_label(list_of_distances):
            index_sorted = np.argsort(list_of_distances)

            k_labels = self.labels[index_sorted[:self.k]]

            k_labels_counter = Counter(k_labels)
            most_common = k_labels_counter.most_common(1)

            return most_common[0][0]

        distance = self.find_dist(X_test)
        pred = list(map(predict_label, distance))

        return pred

        


