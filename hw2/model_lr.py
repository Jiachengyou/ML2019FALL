#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 

@author:Jiachengyou(贾成铕)
@license: Apache Licence 
@file: Tacode.py 
@time: 2019/10/18
@contact: 1284975112@qq.com
@site:  
@software: PyCharm 
"""

import numpy as np
import pandas as pd
import csv
import sys

def load_data(X_train_path, X_test_path, Y_train_path):
    x_train = pd.read_csv(X_train_path)
    x_test = pd.read_csv(X_test_path)

    x_train = x_train.values
    x_test = x_test.values

    y_train = pd.read_csv(Y_train_path, header = None)
    y_train = y_train.values
    y_train = y_train.reshape(-1)

    return x_train, y_train, x_test

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)


def normalize(x_train, x_test):
    x_all = np.concatenate((x_train, x_test), axis=0)
    mean = np.mean(x_all, axis=0)
    std = np.std(x_all, axis=0)

    index = [0, 1, 3, 4, 5]
    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    x_all_nor = (x_all - mean_vec) / std_vec

    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]

    return x_train_nor, x_test_nor

def getP(w, b, x):
    z = np.dot(w, x) + b
    return sigmoid(z)

def train(x_train, y_train):
    b = 0.0
    w = np.zeros(x_train.shape[1])
    lr = 0.05
    epoch = 1000
    b_lr = 0
    w_lr = np.ones(x_train.shape[1])

    for e in range(epoch):
        z = np.dot(x_train, w) + b
        pred = sigmoid(z)
        loss = y_train - pred

        b_grad = -1 * np.sum(loss)
        w_grad = -1 * np.dot(loss, x_train)

        b_lr += b_grad ** 2
        w_lr += w_grad ** 2

        b = b - lr / np.sqrt(b_lr) * b_grad
        w = w - lr / np.sqrt(w_lr) * w_grad

        if (e + 1) % 100 == 0:
            loss = -1 * np.mean(y_train * np.log(pred + 1e-100) + (1 - y_train) * np.log(1 - pred + 1e-100))
    return w, b


if __name__ == '__main__':
    X_train_path = sys.argv[3]
    X_test_path = sys.argv[5]
    Y_train_path = sys.argv[4]
    output_f = sys.argv[6]
    x_train, y_train, x_test = load_data(X_train_path, X_test_path, Y_train_path)

    x_train, x_test = normalize(x_train, x_test)

    # w, b = train(x_train, y_train)
    #
    # writer = csv.writer(open("./data/lr_w.csv", 'w+', newline=''))
    # writer.writerows(map(lambda x: [x], w))
    # print(b)
    # acc: 0.8527993611989804
    # get w,b
    r = csv.reader(open('./data/lr_w.csv'))
    w = [float(l[0]) for l in r]
    w = np.array(w)
    b = -0.51701396456

    # validate
    # res = []
    # for i in x_train:
    #     p = getP(w, b, i)
    #     if p > 0.5:
    #         res.append(1)
    #     else:
    #         res.append(0)
    # cnt = 0
    # for i in range(x_train.shape[0]):
    #     if res[i] == y_train[i]:
    #         cnt += 1
    # print("val_data acc :", cnt / x_train.shape[0])

    # test

    res = []
    for i in x_test:
        p = getP(w, b, i)
        if p > 0.5:
            res.append(1)
        else:
            res.append(0)
    # write
    r = csv.reader(open('./data/sample_submission.csv'))
    lines = [l for l in r]
    for i in range(0, len(lines) - 1):
        lines[i + 1][1] = res[i]
    writer = csv.writer(open(output_f, 'w+', newline=''))
    writer.writerows(lines)