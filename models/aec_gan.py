import numpy as np
from sklearn import metrics
import yaml

import sys,os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

from tqdm import tqdm
import os,sys
import datetime
import numpy as np

import scipy
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn import cross_validation, metrics

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dropout, Activation
from keras import layers
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Concatenate
from keras.layers.normalization import BatchNormalization
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.layers import PReLU

import glob
import argparse


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class PlotLoss(callbacks.History):
    def __init__(self,fname):
        self.fname = fname
        self.history_dict={'loss':[],'val_loss':[]}
        # purge history
        pd.DataFrame(self.history_dict).to_csv(self.fname,index=False)
    def on_epoch_end(self,batch, logs):
        self.history_dict['loss'].append(logs['loss'])
        self.history_dict['val_loss'].append(logs['val_loss'])
        # store history
        pd.DataFrame(self.history_dict).to_csv(self.fname,index=False)

def block(m,node_num=[32,32,16],dropout_rate=0.0,mode=-1,cons=False):
    merge_list = []
    for n in node_num:
        m = Dense(n)(m)
        m = BatchNormalization(axis=mode)(m)
        m = PReLU()(m)
        m = Dropout(dropout_rate)(m)
        merge_list.append(m)
    if len(merge_list) > 1:
        m = Concatenate(axis=-1)([merge_list[0],merge_list[-1]])
        if cons is True:
            m = Dense(node_num[-1])(m)
            m = BatchNormalization(axis=mode)(m)
            m = PReLU()(m)
            m = Dropout(dropout_rate)(m)

    elif len(merge_list) == 1:
        pass
    return m




def get_aec(
    input_shape=50,
    mode = -1,
    dropout_rate=0.0):
    
    f_in = Input(shape=(input_shape,))
    e = Dense(1024)(f_in)
    e = BatchNormalization(axis=mode)(e)
    e = Activation('relu')(e)
    e = Dropout(dropout_rate)(e)
    e = Dense(128)(e)
    e = BatchNormalization(axis=mode)(e)
    e = PReLU()(e)
    e = Dropout(dropout_rate)(e)
    e = Dense(64)(e)
    e = BatchNormalization(axis=mode)(e)
    e = PReLU()(e)
    e = Dropout(dropout_rate)(e)
    e = Dense(8)(e)
    e = BatchNormalization(axis=mode)(e)
    e = PReLU()(e)
    e = Dropout(dropout_rate)(e)
    d = Dense(64)(e)
    d = BatchNormalization(axis=mode)(d)
    d = PReLU()(d)
    d = Dropout(dropout_rate)(d)    
    d = Dense(128)(d)
    d = BatchNormalization(axis=mode)(d)
    d = PReLU()(d)
    d = Dropout(dropout_rate)(d)
    d = Dense(1024)(d)
    d = BatchNormalization(axis=mode)(d)
    d = PReLU()(d)
    d = Dropout(dropout_rate)(d)
    d = Dense(input_shape)(d)
    f_out = Activation('sigmoid')(d)
    
    encoder = Model(inputs=f_in, outputs=e)
    decoder = Model(inputs=f_in, outputs=f_out)
    return encoder, decoder

def get_res(
    input_shape=8,
    dropout_rate=0.0,):
        
    f_in = Input(shape=(input_shape,))
    m=block(f_in,node_num=[32,32,8],dropout_rate=dropout_rate)
    m=block(m,node_num=[32,32,8],dropout_rate=dropout_rate)
    m=block(m,node_num=[16],dropout_rate=dropout_rate)
    m=block(m,node_num=[16],dropout_rate=dropout_rate)
    m = Dense(1)(m)
    f_out = Activation('sigmoid')(m)
    model = Model(inputs=f_in, outputs=f_out)
    return model

def get_discriminator(
    input_shape=50,
    dropout_rate=0.0,):
        
    f_in = Input(shape=(input_shape,))
    m=block(f_in,node_num=[64,64,32],dropout_rate=dropout_rate)
    m=block(m,node_num=[32,32,8],dropout_rate=dropout_rate)
    m=block(m,node_num=[32,32,8],dropout_rate=dropout_rate)
    m=block(m,node_num=[16],dropout_rate=dropout_rate)
    m=block(m,node_num=[16],dropout_rate=dropout_rate)
    m = Dense(1)(m)
    f_out = Activation('sigmoid')(m)
    model = Model(inputs=f_in, outputs=f_out)
    return model

def get_gan(generator, discriminator,input_shape=50,):
    I = Input(shape=(input_shape,))
    G = generator(I)
    D = discriminator(G)
    D.trainable = False
    model = Model(input=I,output=D)
    return model

def chunkify(x, y, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(y), n):
        yield x[i:i + n,:], y[i:i + n]

class AecAdvModel(object):
    def __init__(self,):
        self.is_trained = False
        
        self.gan_weight_path = lambda x: os.path.join(THIS_DIR,'aec_gan_file','gan{:03d}.hdf5'.format(x)) 
        self.discr_weight_path = lambda x: os.path.join(THIS_DIR,'aec_gan_file','discr{:03d}.hdf5'.format(x))
        self.decoder_weight_path = lambda x: os.path.join(THIS_DIR,'aec_gan_file','decoder{:03d}.hdf5'.format(x))
        self.history_path = os.path.join(THIS_DIR,'aec_gan_file','history.yml')
        self.encoder, self.decoder = get_aec()
        self.res = get_res()
        self.discr = get_discriminator()
        self.gan = get_gan(self.decoder,self.discr)

    def load(self):
        return
        res_results=pd.read_csv(self.res_csv)
        res_epoch=np.argmin(res_results['val_loss'])+1
        res_weights = glob.glob(os.path.join(THIS_DIR,r'aec_file/weightsRES.'+'{:03d}'.format(res_epoch)+'*'))
        self.res_weight_file = res_weights[0]
        self.res.load_weights(self.res_weight_file)

        self.is_trained = True

    def fit(self,X_train=None,y_train=None,X_validation=None,y_validation=None):
        if X_validation is None or y_validation is None:
            raise IOError()
        info_list = []
        with open(self.history_path, "w") as f:
            f.write(yaml.dump(info_list))

        loss = 'mse' #<-- mse likely sucks
        enc_opt = Adam(lr=0.0005)
        self.encoder.compile(loss=loss, optimizer=enc_opt)
        dec_opt = Adam(lr=0.0005)
        self.decoder.compile(loss=loss, optimizer=dec_opt)

        stacked_opt = Adam(lr=0.0005)
        self.gan.compile(loss='binary_crossentropy', optimizer=stacked_opt)
        
        self.discr.trainable = True
        descr_opt = Adam(lr=0.0001)
        self.discr.compile(loss='binary_crossentropy', optimizer=descr_opt)

        epoch_num = 200
        batch_size = 64

        for epoch in range(epoch_num):
            d_loss_list = []
            g_loss_list = []
            s_loss_list = []

            for bX_train,by_train in chunkify(X_train,y_train,batch_size):


                bdecOut = self.decoder.predict(bX_train, verbose=0)

                X = np.concatenate((bdecOut,bX_train),axis=0).astype('float')
                
                b_size = bX_train.shape[0]
                y = [1] * b_size + [0] * b_size
                y +=0.5*(np.random.random(2*b_size)-0.5)
                
                one_y = np.array([1]  *b_size)

                # train only discriminator with pos and neg.
                for _X,_y in [(X[:b_size,:],y[:b_size]),(X[b_size:,:],y[b_size:])]:
                    d_loss = self.discr.train_on_batch(_X,_y)
                    d_loss_list.append(d_loss)

                self.discr.trainable = False

                g_loss = self.decoder.train_on_batch(bX_train,bX_train)
                s_loss = self.gan.train_on_batch(bX_train,one_y)

                self.discr.trainable = True

                g_loss_list.append(g_loss)
                s_loss_list.append(s_loss)
                
            decoder_val_loss = self.decoder.evaluate(X_validation, X_validation, batch_size=batch_size)
            print(epoch,decoder_val_loss,g_loss,s_loss)
            info_list.append({
                'epoch':epoch,
                'd_loss':float(np.mean(d_loss_list)),
                'g_loss':float(np.mean(g_loss_list)),
                's_loss':float(np.mean(s_loss_list)),
                'decoder_val_loss': float(decoder_val_loss),
            })
            self.decoder.save_weights(self.decoder_weight_path(epoch), True)
            self.gan.save_weights(self.gan_weight_path(epoch), True)
            self.discr.save_weights(self.discr_weight_path(epoch), True)
            with open(self.history_path, "w") as f:
                f.write(yaml.dump(info_list))

        self.is_trained = True

    def predict(self,X,y_true=None):
        if self.is_trained is False:
            self.load()

        eX = self.encoder.predict(X)
        y_pred = self.res.predict(eX)
        print(y_pred.shape)

        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true,y_pred)
            print('logloss %r' % logloss)

        return y_pred, logloss
