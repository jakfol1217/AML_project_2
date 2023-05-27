import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def get_artificial_data(path = 'data/'):
    train_art = pd.read_csv(path + 'artificial_train.data', delim_whitespace=True, header=None)
    test_art  = pd.read_csv(path + 'artificial_valid.data', delim_whitespace=True, header=None)
    train_labels = pd.read_csv(path + 'artificial_train.labels', header=None)
    return train_art, train_labels, test_art

def get_spam_training_data(file, labels, sep):
    spam_train = []
    spam_train_labels = []
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.replace('"', '')
            label, text = line.split(sep=sep, maxsplit=1)
            if label not in labels:
                continue
            spam_train_labels.append(label)
            spam_train.append(text[:-1])
    cv = CountVectorizer(strip_accents='unicode')
    X = cv.fit_transform(spam_train)
    df = pd.DataFrame(X.toarray(),
                      columns=cv.get_feature_names_out())
    return df, spam_train_labels, cv

def get_spam_test_data(file, cv):
    spam_test = []
    with open(file, encoding='utf8') as f:
        for line in f:
            line = line.replace('"', '')
            spam_test.append(line[:-1])
    spam_test.pop(0)
    X = cv.transform(spam_test)
    df = pd.DataFrame(X.toarray(),
                  columns=cv.get_feature_names_out())
    return df, spam_test