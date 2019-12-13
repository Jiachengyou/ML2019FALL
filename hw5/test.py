#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 

@author:Jiachengyou(贾成铕)
@license: Apache Licence 
@file: test.py 
@time: 2019/12/08
@contact: 1284975112@qq.com
@site:  
@software: PyCharm 
"""


import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Masking
from keras.layers import LSTM
from keras.models import load_model
from keras.layers import BatchNormalization, Activation
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import regularizers
from keras import backend as K
from keras.callbacks import History, ModelCheckpoint, EarlyStopping, Callback
import emoji
import spacy
import sys
import pickle


max_len = 120
vector_dim = 300


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def is_token_allowed(token):
    if (not token or not token.string.strip() or
            token.is_stop or token.is_punct):
        return False
    else:
        return True

def preprocess_token(token):
    return token.lemma_.strip().lower()


def scapy_process(data_path):
    data = pd.read_csv(data_path)['comment'].values
    data = [emoji.demojize(seg) for seg in data]
    data = [seg.replace('@user', ' ') for seg in data]
    nlp = spacy.load('en_core_web_sm')
    list_arr = []
    for index, sentence in enumerate(data):
        doc = nlp(sentence)
        list1 = [preprocess_token(token) for token in doc if is_token_allowed(token)]
        list_arr.append(list1)
    arr = np.array(list_arr)
    return arr





def eval(trainX, model_path):

    # wordvc = gensim.models.Word2Vec.load('./wordEmbed')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    # tokenizer
    # print('Found %s unique tokens.' % len(word_index))
    sequences_test = tokenizer.texts_to_sequences(trainX)
    test_X = pad_sequences(sequences_test, maxlen=max_len)

    def swish(x):
        return (K.sigmoid(x) * x)

    get_custom_objects().update({'swish': Activation(swish)})

    model = load_model(model_path, custom_objects={'f1': f1})
    result = model.predict(test_X)
    result = np.argmax(result, axis=1)
    return result


if __name__ == '__main__':
    test_data_path = sys.argv[1]
    output_path = sys.argv[2]
    testX = scapy_process(test_data_path)
    print("preprocess over!!!")
    # eval
    model_path = "./model/model1.h5"
    # train
    result = eval(testX, model_path)
    print("eval over!!!")
    df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
    df.to_csv(output_path, index=False)
    print("all over!!!")
