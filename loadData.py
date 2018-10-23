import os
import json
from sentimentCalculator import textBlockToSentiment

labelsTrainFile = "Task2_user_reputation_prediction/Reputation_prediction_labels_train.txt"
dataTrainFile = "Task2_user_reputation_prediction/Reputation_prediction_data_train.txt"
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
        with open('cachedMatrix.txt', 'r') as dataTrain:
            return userToReps, json.load(dataTrain)
    except:
        postsMatrix = []
        try:
            with open(dataTrainFile, "r") as dataTrain:
                line = dataTrain.readline()
                counter = 0
                while (line):
                    textualDescription, userId, contentType, timeStamp = line.split('\t')

                    pos, neutral, neg = textBlockToSentiment(textualDescription), 1, 1

                    postsMatrix.append({'textualDescription': textualDescription, 'sentimentPos': pos, \
                    'sentimentNeutral': neutral, 'sentimentNeg': neg, 'contentType': contentTypes[contentType], \
                    'timeStamp': float(timeStamp), 'reputation': userToReps[userId]})

                    line = dataTrain.readline()
                    print(counter)
                    counter += 1
            
            with open('cachedMatrix.txt', 'w') as fout:
                json.dump(postsMatrix, fout)
        
        except:
            try:
                os.remove('cachedMatrix.txt')
            except:
                pass



    return userToReps, postsMatrix




first, second = getDictionaries()