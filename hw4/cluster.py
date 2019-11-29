#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 

@author:Jiachengyou(贾成铕)
@license: Apache Licence 
@file: cluster.py
@time: 2019/11/17
@contact: 1284975112@qq.com
@site:  
@software: PyCharm 
"""


import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn import mixture
from keras.models import load_model
import sys
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Lambda
from keras.layers.core import Flatten
from keras.layers import Reshape
from keras.losses import mean_squared_error
from keras.models import Model
from keras import backend as K


def loader(npy_dir):
    trainX = np.load(npy_dir)
    trainX = np.transpose(trainX, (0, 1, 2, 3)) / 255. * 2 - 1
    return trainX


def sampling(args):

    """
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    z = z_mean + K.exp(0.5 * z_log_var) * epsilon
    return z


def relu(x):
    return K.relu(x)

def train(trainX):

    # encoder
    input_img = Input(shape=(32, 32, 3))
    cov1 = Convolution2D(8, (3, 3), strides=2, padding='same', activation=relu)(input_img)
    z_mean = Convolution2D(16, (3, 3), strides=2, padding='same', activation=relu, name='z-mean')(cov1)
    z_log_var = Convolution2D(16, (3, 3), strides=2, padding='same', activation=relu, name='z_log_var')(cov1)
    z_mean = Flatten()(z_mean)
    z_log_var = Flatten()(z_log_var)
    # noise
    z = Lambda(sampling, output_shape=(8*8*16,), name='z')([z_mean, z_log_var])
    z = Reshape((8, 8, 16))(z)
    encoder = Model(input_img, z, name='encoder')

    # decoder
    encoded_i = Input(shape=(8, 8, 16))
    covT1 = Conv2DTranspose(8, (3, 3), strides=2, padding="same", activation=relu)(encoded_i)
    output = Conv2DTranspose(3, (3, 3), strides=2, padding="same", activation='tanh')(covT1)
    decoder = Model(encoded_i, output, name='decoder')
    outputs = decoder(encoder(input_img))
    print(outputs.shape)
    vae = Model(input=input_img, output=outputs, name='vae')
    print(encoder.summary())
    print(decoder.summary())
    print(vae.summary())

    reconstruction_loss = mean_squared_error(input_img, outputs)
    reconstruction_loss *= 32*32*3
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = (kl_loss + reconstruction_loss) / (32*32*3)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.fit(trainX[:8000],
                    epochs=50,
                    batch_size=32,
                    shuffle=False,
                    )
    # vae.save('./model32/autoencoder.h5')
    # encoder.save('./model32/encoder.h5')
    # decoder.save('./model32/decoder.h5')
    # reconstruction
    # result = vae.predict(trainX[:10])
    # reconstruction(trainX[:10], result[:10])
    del vae
    del encoder
    del decoder


if __name__ == '__main__':
    npy_dir = sys.argv[1]
    output_dir = sys.argv[2]
    trainX = loader(npy_dir)
    encoder = load_model("./data/encoder.h5")
    print("encode...")
    latents = encoder.predict(trainX)
    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0))
    print("encoder over!!!")
    print("PCA...")
    res = PCA(n_components=2).fit_transform(latents)
    print("PCA over!!!")

    print('Computing t-SNE embedding...')
    tsne = TSNE(n_components=2, init='pca', random_state=501)
    res = tsne.fit_transform(res)
    print("t-SNE over!!!")
    print("clusering...")
    result = SpectralClustering(n_clusters=2, gamma=0.01).fit(res).labels_
    # g = mixture.GaussianMixture(n_components=2).fit(res)
    # result = g.predict(res)
    print("cluster over!!!")


    if np.sum(result[:5]) >= 3:
        result = 1 - result

    df = pd.DataFrame({'id': np.arange(0, len(result)), 'label': result})
    df.to_csv(output_dir, index=False)


