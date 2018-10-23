import textblob
from textblob import TextBlob


def textBlockToSentiment(textBlock):
    sentiment = TextBlob(textBlock)
    return float(sentiment.sentiment.polarity), float(sentiment.sentiment.subjectivity)