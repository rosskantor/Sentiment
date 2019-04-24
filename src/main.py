import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
import string
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as sh
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.naive_bayes import MultinomialNB

def vectorize(df):
    """
    :param df: dataframe
    :return: term-document matrix (sparse matrix)
    """
    cv = CountVectorizer(stop_words='english')
    bag_of_words = cv.fit_transform(df.lemmatize)

    return bag_of_words


def r_shuffle(df):

    return sh(df, random_state=24)[:25000]

def lower_case_strip(text):
    """
    :param text: string
    :return: dataframe
    """

    df.Phrase = df.Phrase.str.lower()
    df.Phrase = df.Phrase.str.strip()
    df.Phrase = df.Phrase.str.translate(str.maketrans('', '', string.punctuation))

    return df


def tokenize(df):
    """
    :param df: dataframe
    :return: dataframe
    """
    df['lemmatize'] = df.Phrase.apply(lemmatize_text)
    return df


def lemmatize_text(text):
    """
    :param text: string
    :return: bag of words array
    """
    lemmatizer = WordNetLemmatizer()
    w_tokenizer = WhitespaceTokenizer()
    return ' '.join(map(str, [lemmatizer.lemmatize(word)
                              for word in w_tokenizer.tokenize(text)]))


def r_forest():
    model = RandomForestClassifier()
    param_grid = {'n_estimators': [50, 100]
        , 'min_samples_split': [5, 6]}

    return model, param_grid


def train_test(bag_of_words, y):
    """
    :param bag_of_words: bag of words array
    :return: sklearn.pipeline object
    """

    return train_test_split(bag_of_words, y, test_size=0.33, random_state=42)


def grid_search(model, param_grid):
    """
    :param pipe: sklearn.pipeline object
    :param search_space: list of model dictionaries
    :return: best model object
    """
    return GridSearchCV(model, param_grid, cv=5, verbose=0, n_jobs=-1)


def naive():

    """
    :return: returns a multinomial Naive Bayes classifier
    """
    return MultinomialNB()


def best_model(x, y, input_model):
    """
    :param x: numpy array (bag of words)
    :param y: numpy array (target)
    :param input_model: gridsearch model object
    :return: sklearn model object
    """
    return input_model.fit(x, y)

def scoring_metrics(input_model, x, y):
    """
    :param input_model: best random forest classifier
    :param x_test: sparse matrix
    :return:
    """
    y_pred = input_model.predict(x)
    accuracy = accuracy_score(y_pred, y)
    return accuracy

if __name__ == "__main__":
    df = pd.read_csv('../data/train.tsv', sep='\t')
    df = r_shuffle(df)
    lower_case_strip(df)
    df = tokenize(df)
    bag_of_words = vectorize(df)
    x_train, x_test, y_train, y_test = train_test(bag_of_words, df.Sentiment)
    rf_model, param_grid = r_forest()
    model = grid_search(rf_model, param_grid)
    b_model = best_model(x_train, y_train, model)
    acc = scoring_metrics(b_model, x_test, y_test)
