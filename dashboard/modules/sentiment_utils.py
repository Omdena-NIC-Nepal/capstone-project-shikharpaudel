# modules/sentiment_utils.py

from textblob import TextBlob
import pandas as pd

def analyze_sentiments(text_list):
    sentiments = []
    for text in text_list:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
        sentiments.append({'Text': text, 'Polarity': polarity, 'Sentiment': sentiment})
    return pd.DataFrame(sentiments)
