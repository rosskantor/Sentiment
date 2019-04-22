import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
import string

def vectorize(df):
    """
    :param df: dataframe
    :return:
    """
    cv = CountVectorizer(stop_words='english')
    bag_of_words = cv.fit_transform(df.lemmatize)


def lower_case_strip(text):
    """
    :param text: string
    :return: dataframe
    """

    df.Phrase = df.Phrase.str.lower()
    df.Phrase = df.Phrase.str.translate(str.maketrans('', '', string.punctuation))

    return df


def tokenize(df):
    """
    :param df: dataframe
    :return:
    """
    df['lemmatize'] = df.Phrase.apply(lemmatize_text)
    return df


def lemmatize_text(text):
    """
    :param text: string
    :return:
    """
    lemmatizer = WordNetLemmatizer()
    w_tokenizer = WhitespaceTokenizer()
    return ' '.join(map(str, [lemmatizer.lemmatize(word)
                              for word in w_tokenizer.tokenize(text)]))

if __name__ == "__main__":
    df = pd.read_csv('../data/train.tsv',sep='\t')
    lower_case_strip(df)
    tokenize(df)
    vectorize(df)
