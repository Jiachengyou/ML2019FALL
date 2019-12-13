#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 

@author:Jiachengyou(贾成铕)
@license: Apache Licence 
@file: model1.py
@time: 2019/11/26
@contact: 1284975112@qq.com
@site:
@software: PyCharm
"""
import string as st
import pandas as pd
import spacy
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Masking
from keras.layers import LSTM
from keras.models import load_model
from keras.layers import BatchNormalization, Activation
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import Dropout
import re
from keras.preprocessing.text import Tokenizer
import gensim
from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import regularizers
from keras import backend as K
from keras.callbacks import History, ModelCheckpoint, EarlyStopping, Callback
import random
import emoji
import spacy
import os
import pickle
import sys





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

# class Metrics(Callback):
#     def on_train_begin(self, logs={}):
#         self.val_f1s = []
#         self.val_recalls = []
#         self.val_precisions = []
#
#
#     def on_epoch_end(self, epoch, logs={}):
#         val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
#         val_targ = self.model.validation_data[1]
#         _val_f1 = f1_score(val_targ, val_predict)
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
#         self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         print("— val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
#         return

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
    print('data preprocess...')
    for index, sentence in enumerate(data):
        doc = nlp(sentence)
        list1 = [preprocess_token(token) for token in doc if is_token_allowed(token)]
        list_arr.append(list1)
        # if index % 1000 == 0:
            # print(index)
            # print(list1)
    arr = np.array(list_arr)
    print("preprocess over!!!")
    return arr

def word_embeded(data):
    # train
    w2v_model = gensim.models.Word2Vec(data, size=300, window=5, min_count=0, workers=8)
    w2v_model.save('./model/wordEmbed')


def label_loader(datapath):
    data = pd.read_csv(datapath)['label'].values
    return data

def train(trainX, labels):

    wordvc = gensim.models.Word2Vec.load('./model/wordEmbed')

    # tokenizer
    model = Sequential()
    batch_size = 64
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    sequences = tokenizer.texts_to_sequences(trainX)
    word_index = tokenizer.word_index
    # with open('tokenizer.pickle', 'wb') as handle:
    #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('pickle save over!!!')
    oov_count = 0
    embedding_matrix = np.zeros((len(word_index)+1, vector_dim))
    for word, i in word_index.items():
        try:
            embedding_vector = wordvc.wv[word]
            embedding_matrix[i] = embedding_vector
        except:
            oov_count += 1
            print(word)
    print("oov_count:", oov_count)



    def swish(x):
        return (K.sigmoid(x) * x)

    get_custom_objects().update({'swish': Activation(swish)})
    train_X = pad_sequences(sequences, maxlen=max_len)
    train_Y = to_categorical(np.asarray(labels))
    train_x, valid_x, train_y, valid_y = train_test_split(train_X, train_Y,
                                                    test_size=0.07, random_state=501)
    print("train_x shape :", train_x.shape)
    print("word_index:", len(word_index))
    model.add(Embedding(len(word_index)+1, output_dim=vector_dim, weights=[embedding_matrix],
                            input_length=max_len,
                        ))
    # model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(128, activation="tanh", dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, activation="tanh", dropout=0.2, return_sequences=False)))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='swish'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='swish'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    model.fit(train_x, train_y,
              batch_size=batch_size,
              nb_epoch=2,
              validation_data=(valid_x, valid_y),
            )
    # hist.loss_plot('epoch')
    model.save('model2.h5')

    # model = model.load('model1.h5')





if __name__ == '__main__':
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[3]
    # trainX = trainX_process(train_data_path)
    label_path = sys.argv[2]
    # analyze(train_data_path, label_path)
    trainX = scapy_process(train_data_path)
    # testX = scapy_process(test_data_path)
    labels = label_loader(label_path)
    # embedding word
    print("embedding...")
    word_embeded(trainX)
    print("embedding over!!!")



    # train
    print("train...")
    train(trainX, labels)
    print("train_over!!!")

    # model = load_model('model1.h5')







