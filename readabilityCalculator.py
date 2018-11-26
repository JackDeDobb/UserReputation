#install textstat (put into readme)
import textstat


def textBlockToReadability(textBlock):
    readability = textstat.flesch_reading_ease(textBlock)
    words = textBlock.split()
    if len(words) == 0:
        avg_length = 0
    else:
        avg_length = sum(map(len, words))/len(words)
    return readability, avg_length
