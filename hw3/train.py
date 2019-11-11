#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 

@author:Jiachengyou(贾成铕)
@license: Apache Licence 
@file: pre_train.py 
@time: 2019/11/02
@contact: 1284975112@qq.com
@site:  
@software: PyCharm 
"""

import torch
import torch.utils.data
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
import pandas as pd
import torchvision.models as models
import sys

# dataProcess
class dataProcess(Dataset):

    def __init__(self, data_dir, label):
        self.data_dir = data_dir
        self.label = label

    def __getitem__(self, index):
        pic_path = '{:0>5d}.jpg'.format(self.label[index][0])
        # covert rgb
        img = cv2.imread(os.path.join(self.data_dir, pic_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        return img.float().div(255), self.label[index][1]

    def __len__(self):
        return self.label.shape[0]


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.resnet = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.Dropout(0.4),
            nn.Linear(128, 7)
        )


    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 1 * 1 * 512)
        x = self.fc(x)
        return x

def train(train_loader, test_loader):
    epoch = 175
    # isCuda
    isCuda = torch.cuda.is_available()
    device = torch.device('cuda' if isCuda else 'cpu')
    model = Resnet18().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for i in range(epoch):
        model.train()
        train_loss = 0
        correct = 0
        for batchi_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, label)
            train_loss += F.cross_entropy(output, label).item()
            loss.backward()
            optimizer.step()
        if (i % 1 == 0):
            print(i, ":", train_loss)
    torch.save(model.state_dict(), './model.pkl')
    print("over!")

def test(test_loader, i):
    model = Resnet18()
    model.load_state_dict(torch.load("./resnet18/model{0}.pkl".format(i)))
    # model.load_state_dict(torch.load("./resnet18/model.pkl"))
    isCuda = torch.cuda.is_available()
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (img, index) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            # 我们的网络输出的实际上是个概率分布，去最大概率的哪一项作为预测分类
            _, pred_label = torch.max(out, 1)
            total += index.size(0)
            correct += (index == pred_label).sum().item()
    print("acc:", correct / total)


def eval(res_loader):
    model = Resnet18()
    model.load_state_dict(torch.load('./resnet18/model175.pkl'))
    isCuda = torch.cuda.is_available()
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    res = []
    with torch.no_grad():
        for batch_idx, (img, index) in enumerate(res_loader):
            img = img.to(device)
            out = model(img)
            # 我们的网络输出的实际上是个概率分布，去最大概率的哪一项作为预测分类
            pred_label = torch.max(out, 1)[1].data.numpy().squeeze()
            for item in pred_label:
                res.append(item)
            # total += index.size(0)
            # correct += (index == pred_label).sum().item()
    # print("acc:", correct / total)
    return res


if __name__ == '__main__':

    img_dir = sys.argv[1]
    train_dir = sys.argv[2]
    # train_data
    all_label = pd.read_csv(train_dir)
    train_label = all_label.values[:25000]
    train_dataSet = dataProcess(img_dir, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataSet, batch_size=128)
    # testdata
    test_label = all_label.values[25000:]
    test_dataSet = dataProcess(img_dir, test_label)
    test_loader = torch.utils.data.DataLoader(test_dataSet, batch_size=128)
    #
    #
    # # train
    train(train_loader, test_loader)
    print("over!!")
    # # res
    # model = Resnet18()
    # print(model)
    # # result
    # res_label = pd.read_csv('./data/sample_submission.csv')
    # res_label = res_label.values
    # res_dataSet = dataProcess('./data/test_img', res_label)
    # res_loader = torch.utils.data.DataLoader(res_dataSet, batch_size=256)
    # res = eval(res_loader)
    #
    # # write
    # id = [i for i in range(len(res))]
    # dataframe = pd.DataFrame({'id': id, 'label': res})
    # dataframe.to_csv("res.csv", index=False, sep=',')


