import os
import json
from sentimentCalculator import textBlockToSentiment
from readabilityCalculator import textBlockToReadability
import pandas as pd


contentTypes = {'Q': 1, 'A': 2, 'C': 3, 'E': 4, 'F': 5}



def getPostsMatrix(dataTrainFile, userToReps, cachingFile):
    postsMatrix = []
    try:
        with open(dataTrainFile, "r") as dataTrain:
            line = dataTrain.readline()
            while (line):
                textualDescription, userId, contentType, timeStamp = line.split('\t')
                polarity, subjectivity = textBlockToSentiment(textualDescription)
                readability, avg_length = textBlockToReadability(textualDescription)
                if cachingFile:
                    postsMatrix.append([polarity, subjectivity, readability, avg_length, contentTypes[contentType], userToReps[userId] ])
                else:
                    postsMatrix.append([polarity, subjectivity, readability, avg_length, contentTypes[contentType]])
                line = dataTrain.readline()

        df = pd.DataFrame(postsMatrix)
        print()
        if cachingFile:
            df.to_csv(cachingFile, header=True, index=False)

    except:
        pass

    return postsMatrix


def getTrainDictionaries(labelsTrainFile, dataTrainFile, cachingFile):
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
        postsMatrix = getPostsMatrix(dataTrainFile, userToReps, cachingFile)
    
    return userToReps, postsMatrix
    
    
def getTestDictionary(dataTestFile):
    postsMatrix = getPostsMatrix(dataTestFile, None, None)
    return postsMatrix

