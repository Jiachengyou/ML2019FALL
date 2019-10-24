#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""

@author:Jiachengyou(贾成铕)
@license: Apache Licence
@file: model_lr.py
@time: 2019/10/15
@contact: 1284975112@qq.com
@site:
@software: PyCharm
"""


import math
import numpy as np
import pandas as pd
import csv
import sys

#  processData as a numpy array
def processXData(filePath):
    with open(filePath, 'r') as f:
        data = f.readlines()
        data = data[1:]
        data = [s.rstrip() for s in data]
        data = [s.split(',') for s in data]
        data = [[int(i) for i in list] for list in data]
    return np.array(data)
# 1    --14 52 105

def createOnehot(x_data):
    cnt = x_data.shape[0]
    # age
    age_arr = x_data[:, 0]
    age_num = int(np.max(age_arr) / 20) + 1
    age_onehot = np.zeros((cnt, age_num))
    for i in range(cnt):
        pos = x_data[i][0] / 20
        age_onehot[i][int(pos)] = 1
    # fnlwgt
    fnlwgt_arr = x_data[:, 1]
    print(np.max(fnlwgt_arr))
    fnlwgt_num = int(np.max(fnlwgt_arr) / 400000) + 1
    fnlwgt_onehot = np.zeros((cnt, fnlwgt_num))
    for i in range(cnt):
        pos = x_data[i][1] / 400000
        fnlwgt_onehot[i][int(pos)] = 1

    # capital_gain
    capital_gain_arr = x_data[:, 3]
    capital_gain_num = int(np.max(capital_gain_arr) / 20000) + 1
    capital_gain_onehot = np.zeros((cnt, capital_gain_num))
    for i in range(cnt):
        pos = x_data[i][3] / 20000
        capital_gain_onehot[i][int(pos)] = 1

    # capital_loss
    capital_loss_arr = x_data[:, 4]
    capital_loss_num = int(np.max(capital_loss_arr) / 1000) + 1
    capital_loss_onehot = np.zeros((cnt, capital_loss_num))
    for i in range(cnt):
        pos = x_data[i][4] / 1000
        capital_loss_onehot[i][int(pos)] = 1

    # hours_per_week
    hours_per_week_arr = x_data[:, 5]
    hours_per_week_num = int(np.max(hours_per_week_arr) / 20) + 1
    hours_per_week_onehot = np.zeros((cnt, hours_per_week_num))
    for i in range(cnt):
        pos = x_data[i][5] / 20
        hours_per_week_onehot[i][int(pos)] = 1
    x_data = x_data[:, :64]
    # [0, 3, 4, 5]
    x_data = np.delete(x_data, [0, 1, 3, 4, 5], axis=1)
    x_data = np.concatenate((x_data, age_onehot), axis=1)
    x_data = np.concatenate((x_data, capital_gain_onehot), axis=1)
    x_data = np.concatenate((x_data, capital_loss_onehot), axis=1)
    x_data = np.concatenate((x_data, hours_per_week_onehot), axis=1)
    x_data = np.concatenate((x_data, fnlwgt_onehot), axis=1)
    res = x_data
    print(x_data.shape)
    print(res.shape)
    for k in range(x_data.shape[1]):
        if np.max(x_data[:, k]) == 0:
            res = np.delete(x_data, k, axis=1)
    print(res.shape)
    return res



def processXdata2(data, y_data):
    x = []
    y = []
    for i in range(len(data)):
        if 1:
            x.append(data[i])
            y.append(y_data[i])
    x = np.array(x)
    y = np.array(y)
    x = np.delete(x, [1, 14, 52, 105], axis=1)
    print(len(x))
    return x, y

def normalize(x_data):
    mean = np.mean(x_data, axis=0)
    std = np.std(x_data, axis=0)

    index = [0, 1, 3, 4, 5]
    mean_vec = np.zeros(x_data.shape[1])
    std_vec = np.ones(x_data.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    x_all_nor = (x_data - mean_vec) / std_vec
    return x_all_nor


def processYData(filePath):
    with open(filePath, 'r') as f:
        data = f.readlines()
        data = [s.rstrip() for s in data]
        data = [int(i) for i in data]
    return np.array(data)

def createModel(x_data, y_data):
    merge_arr = np.column_stack((x_data, y_data))
    class1 = [i[:-1] for i in merge_arr if i[-1] == 0]
    class2 = [i[:-1] for i in merge_arr if i[-1] == 1]
    class1 = np.array(class1, dtype=np.float64)
    class2 = np.array(class2, dtype=np.float64)
    class1_t = class1.T
    class2_t = class2.T
    u1 = np.sum(class1_t, axis=1) / len(class1)
    u2 = np.sum(class2_t, axis=1) / len(class2)
    var1 = np.zeros((len(u1), len(u1)), dtype=np.float64)
    for i in class1:
        var1 += np.outer(i-u1, (i-u1).T)
    var1 = var1 / len(class1)
    var2 = np.zeros((len(u2), len(u2)), dtype=np.float64)
    for i in class2:
        var2 += np.outer((i-u2), (i-u2).T)
    var2 = var2 / len(class2)
    print(var1)
    common_var = (len(class1)*var1 + len(class2)*var2) / (len(class1)+len(class2))
    print(np.linalg.det(var1), np.linalg.det(var2), np.linalg.det(common_var))
    return u1, u2, common_var, len(class1), len(class2)

def gauss_dis(u, var, x):
    n = len(u)
    front = 1 / np.sqrt(np.power(2*np.pi, n)*abs(np.linalg.det(var)))
    head = -(1/2) * np.dot(np.dot((x-u).T, np.linalg.inv(var)), x-u)
    # print(front)
    # print(head)
    # print(np.exp2(head))
    return front*np.exp(head)

def train(X_train, Y_train):
    # vaild_set_percetange = 0.1
    # X_train, Y_train, X_valid, Y_valid = split_valid_set(X, Y, vaild_set_percetange)

    #Gussian distribution parameters
    train_data_size = X_train.shape[0]
    dimen = X_train.shape[1]

    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((dimen,))
    mu2 = np.zeros((dimen,))
    for i in range(train_data_size):
        if Y_train[i] == 1:     # >50k
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((dimen, dimen))
    sigma2 = np.zeros((dimen, dimen))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [X_train[i] - mu1])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [X_train[i] - mu2])

    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2

    N1 = cnt1
    N2 = cnt2

    return mu1, mu2, shared_sigma, N1, N2

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def getP(u1, u2, var, n1, n2, x):
    p1 = gauss_dis(u1, var, x)
    p2 = gauss_dis(u2, var, x)
    z = np.log((p1*(n1/(n1+n2))) / (p2*(n2/(n1+n2))))
    return sigmoid(z)

if __name__ == "__main__":

    input_f = sys.argv[5]
    output_f = sys.argv[6]
    # filePath_x_train = "./data/X_train"
    # filePath_y_train = "./data/Y_train"
    filePath_x_test = input_f
    # x_data = processXData(filePath_x_train)
    # y_data = processYData(filePath_y_train)
    # x_data = normalize(x_data)
    # u1, u2, var, n1, n2 = train(x_data, y_data)
    # writer = csv.writer(open("./data/gp_u1.csv", 'w+', newline=''))
    # writer.writerows(map(lambda x: [x], u1))
    # writer = csv.writer(open("./data/gp_u2.csv", 'w+', newline=''))
    # writer.writerows(map(lambda x: [x], u2))
    # writer = csv.writer(open("./data/gp_var.csv", 'w+', newline=''))
    # var = [l for l in var]
    # writer.writerows(var)

    # read
    n1 = 7841
    n2 = 24720
    r = csv.reader(open('./data/gp_u1.csv'))
    u1 = [float(l[0]) for l in r]
    u1 = np.array(u1)
    r = csv.reader(open('./data/gp_u2.csv'))
    u2 = [float(l[0]) for l in r]
    u2 = np.array(u2)
    r = csv.reader(open('./data/gp_var.csv'))
    var = [list(l) for l in r]
    var = np.array(var).astype(np.float64)
    # val
    # res = []
    # for i in x_data:
    #     p = getP(u1, u2, var, n1, n2, i)
    #     if p > 0.5:
    #         res.append(1)
    #     else:
    #         res.append(0)
    # cnt = 0
    # for i in range(len(y_data)):
    #     if res[i] == y_data[i]:
    #         cnt += 1
    # print(cnt/len(y_data))

    # test
    x_data = processXData(filePath_x_test)
    x_data = normalize(x_data)
    res = []
    for i in x_data:
        p = getP(u1, u2, var, n1, n2, i)
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