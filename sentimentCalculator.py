import requests
import json


def textBlockToSentiment(textBlock):
    data = {
    'text': textBlock
    }
    response = requests.post('http://text-processing.com/api/sentiment/', data=data)
    print(response)
    newJson = json.loads(response.text)
    return float(newJson['probability']['pos'])
