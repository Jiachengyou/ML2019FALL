#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 

@author:Jiachengyou(贾成铕)
@license: Apache Licence 
@file: model_svm.py
@time: 2019/10/18
@contact: 1284975112@qq.com
@site:
@software: PyCharm
"""

#!/usr/bin/env python
# -*- coding:utf-8 _*-
""" 

@author:Jiachengyou(贾成铕)
@license: Apache Licence 
@file: model1.py.py 
@time: 2019/10/14
@contact: 1284975112@qq.com
@site:  
@software: PyCharm 
"""
import math
import numpy as np
import pandas as pd
import csv
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, train_test_split
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


# 1    --14 52 105
def processXdata2(data, y_data):
    x = []
    y = []
    for i in range(len(data)):
        if data[i][14] == 0 and data[i][52] == 0 and data[i][105] == 0:
            x.append(data[i])
            y.append(y_data[i])
    x = np.array(x)
    y = np.array(y)
    x = np.delete(x, [14, 52, 105], axis=1)
    return x, y

def processYData(filePath):
    with open(filePath, 'r') as f:
        data = f.readlines()
        data = [s.rstrip() for s in data]
        data = [int(i) for i in data]
    return np.array(data)


def train(x_train, y_train):

    return 0


if __name__ == "__main__":
    input_f = sys.argv[5]
    output_f = sys.argv[6]
    # filePath_x_train = "./data/X_train"
    # filePath_y_train = "./data/Y_train"
    filePath_x_test = input_f
    # x_data = processXData(filePath_x_train)
    # y_data = processYData(filePath_y_train)
    # x_data = normalize(x_data)
    x_test = processXData(filePath_x_test)
    x_test = normalize(x_test)
    # cross_val_score, train_test_split

    # c = [float(i/10) for i in range(5, 11)]
    # kernel = ['linear', 'rbf']
    # rbf 1.0
    # clf = svm.SVC(C=0.9, kernel='rbf')
    # clf.fit(x_data, y_data)
    # joblib.dump(clf, "train_model_rbf_1.0.m")
    # write
    clf = joblib.load("./data/train_model_rbf_1.0.m")
    res = clf.predict(x_test)
    r = csv.reader(open('./data/sample_submission.csv'))
    lines = [l for l in r]
    for i in range(0, len(lines) - 1):
         lines[i + 1][1] = res[i]
    writer = csv.writer(open(output_f, 'w+', newline=''))
    writer.writerows(lines)