import numpy as np
from sklearn import metrics

import sys,os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


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

def block(m,node_num=[32,32,16],dropout_rate=0.2,mode=-1,cons=False):
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
    dropout_rate=0.3):
    
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
    dropout_rate=0.3,):
        
    f_in = Input(shape=(input_shape,))
    m=block(f_in,node_num=[32,32,8],dropout_rate=dropout_rate)
    m=block(m,node_num=[32,32,8],dropout_rate=dropout_rate)
    m=block(m,node_num=[16],dropout_rate=dropout_rate)
    m=block(m,node_num=[16],dropout_rate=dropout_rate)
    m = Dense(1)(m)
    f_out = Activation('sigmoid')(m)
    model = Model(inputs=f_in, outputs=f_out)
    return model



class AecModel(object):
    def __init__(self,):
        self.is_trained = False
        
        self.aec_csv = os.path.join(THIS_DIR,r'aec_file/aec.csv')
        self.res_csv = os.path.join(THIS_DIR,r'aec_file/res.csv')

        self.encoder, self.decoder = get_aec()
        self.res = get_res()
        


    def _load_aec(self):
        aec_results=pd.read_csv(self.aec_csv)
        aec_epoch=np.argmin(aec_results['val_loss'])+1
        aec_weights = glob.glob(os.path.join(THIS_DIR,r'aec_file/weightsAEC.'+'{:03d}'.format(aec_epoch)+'*'))
        self.aec_weight_file = aec_weights[0]
        self.decoder.load_weights(self.aec_weight_file)
        self.encoder.set_weights(self.decoder.get_weights()[:-1])

    def _load_res(self):

        res_results=pd.read_csv(self.res_csv)
        res_epoch=np.argmin(res_results['val_loss'])+1
        res_weights = glob.glob(os.path.join(THIS_DIR,r'aec_file/weightsRES.'+'{:03d}'.format(res_epoch)+'*'))
        self.res_weight_file = res_weights[0]
        self.res.load_weights(self.res_weight_file)

    def load(self):
        self._load_aec()
        self._load_res()
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,X_validation=None,y_validation=None,X_test=None):
        if X_validation is None or y_validation is None:
            raise IOError()

        lr=0.0001,
        #opt = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        opt = Adam(lr=lr) #<faster than SGD
        loss = 'mse' #<-- lets do gan next MSE sucks.
        self.encoder.compile(loss=loss, optimizer=opt)
        self.decoder.compile(loss=loss, optimizer=opt)

        
        aec_callbacks = [
            PlotLoss(self.aec_csv),
            callbacks.EarlyStopping(monitor='val_loss', patience=5),
            callbacks.ModelCheckpoint(os.path.join(
                THIS_DIR,r'aec_file/weightsAEC.{epoch:03d}.hdf5')),
        ]
        
        decoder_X_train = X_train
        if X_test is not None:
            decoder_X_train = np.concatenate([X_train,X_test],axis=0).astype('float')
            
        nb_epoch= 200
        batch_size =64
        history = self.decoder.fit(decoder_X_train,decoder_X_train,shuffle=True,
            batch_size=batch_size,epochs=nb_epoch,
            verbose=1,callbacks=aec_callbacks,validation_data=(X_validation,X_validation),)

        self._load_aec()

        lr=0.0001
        opt = Adam(lr=lr) #< faster than SGD
        self.res.compile(loss='binary_crossentropy', optimizer=opt)

        res_callbacks = [
            PlotLoss(self.res_csv),
            callbacks.EarlyStopping(monitor='val_loss', patience=5),
            callbacks.ModelCheckpoint(os.path.join(
                THIS_DIR,r'aec_file/weightsRES.{epoch:03d}.hdf5')),
        ]

        nb_epoch= 200
        batch_size = 64

        proba = self.encoder.predict(X_train)
        probaVal = self.encoder.predict(X_validation)
        history = self.res.fit(proba, y_train,shuffle=True,
            batch_size=batch_size, epochs=nb_epoch,
            verbose=1, validation_data=(probaVal,y_validation), callbacks=res_callbacks)

        self.load()

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

