import re
import json
import os
import joblib
import string
import nltk
import unidecode
import pandas as pd

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("SnowballStemmer")
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize


def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It's the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION).
    # For multiple models, it points to the folder containing all deployed models (./azureml-models).
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "colombian_tweet_clf.joblib")
    model = joblib.load(model_path)


def run(raw_data):
    data = json.loads(raw_data)["data"]
    data = [str(data)]
    data = pd.DataFrame(data, columns=["text"])
    # Make prediction.
    data["text"] = join_stem_tweet(data["text"])
    y_hat = model.predict_proba(data)
    not_col = y_hat[0]
    col = y_hat[1]
    # You can return any data type as long as it's JSON-serializable.
    return json.dumps(
        {"probability_NOT_colombian": not_col, "probability_colombian": col, "data": "{}".format(data["text"].iloc[0]),}
    )


def clean_tweet(tweet, joined=False):
    """
    Clean tweets from links, emojis and punctuation
    joined = True exports the tweet as 1 string
    joined = False exports as list of tokens
    """
    try:
        sw_sp = set(stopwords.words("spanish"))
        sw_en = set(stopwords.words("english"))
        tweet = re.sub("https\S+", "", tweet)  # remove links
        tokens = word_tokenize(tweet)
        tokens = [unidecode.unidecode(w) for w in tokens]  # convert characters to closest ascii representation
        tokens = [w.lower() if w not in sw_sp else "" for w in tokens]  # all lower case
        tokens = [w if w not in sw_en else "" for w in tokens]  # all lower case
        # tokens = [w.lower() for w in tokens]  # all lower case
        table = str.maketrans("", "", string.punctuation)  # translation mask to remove punctuation
        stripped = [w.translate(table) for w in tokens]  # remove punctuation
        words = [word for word in stripped if word.isalpha()]
    except TypeError:
        words = [""]
    if joined:
        return " ".join(words)
    else:
        return words


def join_stem_tweet(tweets):
    """
    Clean temm and join incoming tweets 
    """
    tkn_tweets = [clean_tweet(tweet) for tweet in tweets]
    txt = " "
    tweets = [txt.join(stem_es(word)) for word in tkn_tweets]
    return tweets


def stem_es(words):
    """
    stem words
    """
    spanish_stemmer = SnowballStemmer("spanish")
    stemmed = [spanish_stemmer.stem(word) for word in words]
    return stemmed
