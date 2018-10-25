import os
import json
from sentimentCalculator import textBlockToSentiment
import pandas as pd

labelsTrainFile = "Task2_user_reputation_prediction/Reputation_prediction_labels_train.txt"
dataTrainFile = "Task2_user_reputation_prediction/Reputation_prediction_data_train.txt"
cachingFile = 'cachedMatrix.csv'
contentTypes = {'Q': 1, 'A': 2, 'C': 3, 'E': 4, 'F': 5}


def getDictionaries():
    userToReps = {}
    with open(labelsTrainFile, "r") as labelsTrain:
        line = labelsTrain.readline()
        while (line):
            userId, reputation = line.split('\t')
            userToReps[userId] = int(reputation)
            line = labelsTrain.readline()


    try:
        with open(cachingFile, 'r') as dataTrain:
            return userToReps, pd.read_csv(dataTrain)
    except:
        # print("here")
        postsMatrix = []
        try:
            with open(dataTrainFile, "r") as dataTrain:
                line = dataTrain.readline()
                # print("now here")
                while (line):
                    textualDescription, userId, contentType, timeStamp = line.split('\t')

                    polarity, subjectivity = textBlockToSentiment(textualDescription)

                    postsMatrix.append([polarity, subjectivity, contentTypes[contentType], userToReps[userId] ])

                    line = dataTrain.readline()


            df = pd.DataFrame(postsMatrix)
            print()
            df.to_csv(cachingFile, header=True, index=False)

        except:
            try:
                pass
            except:
                pass



    return userToReps, postsMatrix




# first, second = getDictionaries()
