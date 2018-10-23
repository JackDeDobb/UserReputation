import os
import json
from sentimentCalculator import textBlockToSentiment

labelsTrainFile = "Task2_user_reputation_prediction/Reputation_prediction_labels_train.txt"
dataTrainFile = "Task2_user_reputation_prediction/Reputation_prediction_data_train.txt"
cachingFile = 'cachedMatrix.txt'
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
            return userToReps, json.load(dataTrain)
    except:
        postsMatrix = []
        try:
            with open(dataTrainFile, "r") as dataTrain:
                line = dataTrain.readline()
                while (line):
                    textualDescription, userId, contentType, timeStamp = line.split('\t')

                    polarity, subjectivity = textBlockToSentiment(textualDescription)

                    postsMatrix.append({'textualDescription': textualDescription, 'polarity': polarity, \
                    'subjectivity': subjectivity, 'contentType': contentTypes[contentType], \
                    'timeStamp': float(timeStamp), 'reputation': userToReps[userId]})

                    line = dataTrain.readline()

            
            with open(cachingFile, 'w') as fout:
                json.dump(postsMatrix, fout)
        
        except:
            try:
                os.remove(cachingFile)
            except:
                pass



    return userToReps, postsMatrix




first, second = getDictionaries()