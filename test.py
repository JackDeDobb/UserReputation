import sys
import pickle
from sklearn.ensemble import VotingClassifier
from loadData import getTestDictionary
from sklearn.metrics import accuracy_score
import time


def getTestFile():
    if len(sys.argv) == 2:
        path = sys.argv[1]
        arr = path.split("/")
        directory = '/'.join(arr[:len(arr) - 1])
    else:
        path = "./Task2_user_reputation_prediction/Reputation_prediction_data_train.txt"
        directory = "Task2_user_reputation_prediction"
    return path, directory

def useModel(test_data, directory):
    model = pickle.load(open('model.sav', 'rb'))
    print(test_data.shape)
    pred = model.predict(test_data)
    with open(directory + '/output.txt', 'w') as f:
        for elem in pred:
            f.write(str(elem) + '\n')


path, directory = getTestFile()
print("**********Creating feature vectors now.***********")
start = time.time()
test_data = getTestDictionary(path)
end = time.time()
print("**********Done creating feature vectors.**********")
print("**********Time elapsed to create feature vectors: " + str(end-start) + "**********")
print("**************************************************")
print()
print("**********Making test predictions now.**********")
start = time.time()
useModel(test_data, directory)
end = time.time()
print("**********Done making test predictions**********")
print("**********Time elapsed for predictions: " + str(end-start) + "**********")
