import numpy as np
from sklearn import metrics
import yaml

import sys,os

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

def get_datadiscr(
    input_shape=50,
    mode = -1,
    dropout_rate=0.3):
    f_in = Input(shape=(input_shape,))
    e = Dense(28)(f_in)
    e = BatchNormalization(axis=mode)(e)
    e = Activation('relu')(e)
    e = Dropout(dropout_rate)(e)
    e = Dense(28)(e)
    e = BatchNormalization(axis=mode)(e)
    e = PReLU()(e)
    e = Dropout(dropout_rate)(e)
    e = Dense(1)(e)
    f_out = Activation('sigmoid')(e)
    model = Model(inputs=f_in, outputs=f_out)
    return model

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

def get_stack(encoder,resnet,input_shape=50):
    I = Input(shape=(input_shape,))
    E = encoder(I)
    R = resnet(E)
    model = Model(inputs=I,outputs=R)
    return model

    
FDNAME = 'aec_gan_stack_file'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class AecAdvStackModel(object):
    def __init__(self,this_dir=THIS_DIR,fdname=FDNAME):
        self.is_trained = False
        self.this_dir = this_dir
        self.fdname = fdname
        self.stack_csv = os.path.join(self.this_dir,self.fdname,'stack.csv')
        
        # copy file sfrom aec_adv_file
        self.history_path = os.path.join(self.this_dir,self.fdname,'history.yml')
        self.decoder_weight_path = lambda x: os.path.join(self.this_dir,self.fdname,'decoder{:03d}.hdf5'.format(x))
            
        self.encoder, self.decoder = get_aec()        
        self._load_aec()
        self.res = get_res()
        self.stack = get_stack(self.encoder,self.res)
        # things may get dicy from here, perhaps load-aec from gan trained encoder is already pretty overfitting.
        #self._load_stack(pre=True)# load with weights trained with Adam
            
    def _load_aec(self):
        with open(self.history_path, "r") as f:
            history = yaml.load(f.read())
        epoch = history[np.argmin([x['decoder_val_loss'] for x in history])]['epoch']
        self.decoder.load_weights(self.decoder_weight_path(epoch))
        self.encoder.set_weights(self.decoder.get_weights()[:-1])

    def _load_stack(self,pre=False):
        if pre is True:
            adam_folder = os.path.join(self.this_dir,self.fdname,'adam')
            res_results=pd.read_csv(os.path.join(adam_folder,'stack.csv'))
            res_epoch=np.argmin(res_results['val_loss'])+1
            res_weights = glob.glob(os.path.join(adam_folder,'weightsSTACK.'+'{:03d}'.format(res_epoch)+'*'))
            self.stack.load_weights(res_weights[0])            
            return
        res_results=pd.read_csv(self.stack_csv)
        res_epoch=np.argmin(res_results['val_loss'])+1
        res_weights = glob.glob(os.path.join(self.this_dir,self.fdname,'weightsSTACK.'+'{:03d}'.format(res_epoch)+'*'))
        self.stack_weight_file = res_weights[0]
        self.stack.load_weights(self.stack_weight_file)

    def load(self):
        self._load_stack()
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,X_validation=None,y_validation=None,sample_weight=None,**kwargs):
        if X_validation is None or y_validation is None:
            raise IOError()
            
        stacked_opt = Adam(lr=0.0001) # used first to train with 7 epochs
        #stacked_opt = SGD(lr=0.01)
        self.stack.compile(loss='binary_crossentropy', optimizer=stacked_opt)
        
        nb_epoch= 200
        batch_size = 64

        res_callbacks = [
            PlotLoss(self.stack_csv),
            callbacks.EarlyStopping(monitor='val_loss', patience=10),
            callbacks.ModelCheckpoint(os.path.join(
                self.this_dir,self.fdname,'weightsSTACK.{epoch:03d}.hdf5')),
        ]
        history = self.stack.fit(X_train, y_train,shuffle=True,sample_weight=sample_weight,
            batch_size=batch_size, epochs=nb_epoch,
            verbose=1, validation_data=(X_validation,y_validation), callbacks=res_callbacks)

        self.load()

    def predict(self,X,y_true=None):
        if self.is_trained is False:
            self.load()

        y_pred = self.stack.predict(X)

        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true,y_pred)

        return y_pred, logloss
