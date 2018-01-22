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
from keras import optimizers
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

import tensorflow as tf
sess = tf.InteractiveSession()

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


def unit0(input,num,
          drop=0.3,axis=-1,
          kernel_regularizer=regularizers.l2(10e-8),
          activity_regularizer=regularizers.l1(10e-8),):
    e = Dense(num,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
        )(input) #l1 l2?
    e = BatchNormalization(axis=axis)(e)
    e = PReLU()(e)
    e = Dropout(drop)(e)
    return e

def unit1(input,num,
          drop=0.3,
          axis=-1,
          activation='sigmoid',
          kernel_regularizer=regularizers.l2(10e-8),
          activity_regularizer=regularizers.l1(10e-8),):
    e = Dense(num,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
        )(input)
    e = BatchNormalization(axis=axis)(e)
    e = Activation(activation)(e)
    return e

def get_upstreams():
    
    input_shape=50
    mode = -1
    dropout_rate=0.3
    in_dropout_rate=0.1
    
    f_in = Input(shape=(input_shape,))
    
    # specific encoder
    e = unit0(f_in,1024,axis=mode,drop=dropout_rate)
    e = unit0(e,128,axis=mode,drop=dropout_rate)
    e = unit0(e,64,axis=mode,drop=dropout_rate)
    e = unit1(e,8,axis=mode,drop=dropout_rate,activation='relu')

    #unspecific encoder
    z = unit0(f_in,4,axis=mode,drop=dropout_rate)
    z = unit1(z,4,axis=mode,drop=dropout_rate,activation='relu')
        
    se = Model(inputs=f_in, outputs=e)    
    ze = Model(inputs=f_in, outputs=z)
    return se,ze

def get_downstreams(SE,ZE):
    
    input_shape=50
    mode = -1
    dropout_rate=0.3
    
    I = Input(shape=(input_shape,))
    se = SE(I)
    ze = ZE(I)
    
    m = Concatenate(axis=-1)([se,ze])
    
    # specific discriminator    
    d = unit0(m,64,axis=mode,drop=dropout_rate)
    d = unit0(d,128,axis=mode,drop=dropout_rate)
    d = unit0(d,1024,axis=mode,drop=dropout_rate)
    d = unit1(d,50,axis=mode,drop=dropout_rate,activation='sigmoid')
    SD = Model(inputs=I,outputs=d)
    
    # unspecific classifier
    c = unit0(ze,4,axis=mode,drop=dropout_rate)
    c = unit1(c,1,axis=mode,drop=dropout_rate,activation='sigmoid')
    ZC = Model(inputs=I, outputs=c)

    return SD,ZC

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


def chunkify(x,y, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, x.shape[0], n):
        yield x[i:i + n,:],y[i:i + n],

FDNAME = 'disentangle_file'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(THIS_DIR,FDNAME)) is False:
    os.makedirs(os.path.join(THIS_DIR,FDNAME))

class DisentangleModel(object):
    def __init__(self,this_dir=THIS_DIR,fdname=FDNAME):
        self.is_trained = False
        self.this_dir = this_dir
        self.fdname = fdname
                
        self.se_weight_path = lambda x: os.path.join(self.this_dir,self.fdname,'se{:03d}.hdf5'.format(x)) 
        self.ze_weight_path = lambda x: os.path.join(self.this_dir,self.fdname,'ze{:03d}.hdf5'.format(x))
        self.sd_weight_path = lambda x: os.path.join(self.this_dir,self.fdname,'sd{:03d}.hdf5'.format(x))
        self.zc_weight_path = lambda x: os.path.join(self.this_dir,self.fdname,'zc{:03d}.hdf5'.format(x))
        self.history_path = os.path.join(self.this_dir,self.fdname,'history.yml')

        self.SE, self.ZE = get_upstreams()
        self.SD, self.ZC = get_downstreams(self.SE,self.ZE)

    def _load(self):
        with open(self.history_path, "r") as f:
            history = yaml.load(f.read())
        epoch = history[np.argmin([x['val_loss'] for x in history])]['epoch']
        
        self.SE.load_weights(self.se_weight_path(epoch))
        self.ZE.load_weights(self.ze_weight_path(epoch))
        self.SD.load_weights(self.sd_weight_path(epoch))
        self.ZC.load_weights(self.zc_weight_path(epoch))

    def load(self):
        self._load()
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,X_validation=None,y_validation=None,X_test=None,**kwargs):
        if X_validation is None or y_validation is None:
            raise IOError()
        '''
        # at lr of 0.0001 for sd and 0.001 for zc
        # mse for sd was 0.726,0.0.0419 at epochs 0 and 4
        # logloss for zc was 0.722,0.0.693 at epochs 0 and 4        
        '''
        
        
        #sd_opt = Adam(lr=0.000001)
        sd_opt = optimizers.Nadam(lr=0.00001)
        self.SD.compile(loss='mse',optimizer=sd_opt)
        
        #zc_opt = SGD(lr=0.001) # after about 10 epoch swich to lr of 0.0001 below
        zc_opt = optimizers.Nadam(lr=0.0001)
        self.ZC.compile(loss='binary_crossentropy',optimizer=zc_opt)

        self.SE.trainable = True
        
        info_list = []
        
        old_info_list = []
        if os.path.exists(self.history_path):
            with open(self.history_path,"r") as f:
                old_info_list = yaml.load(f.read())

        reduce_lr = callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.2,
                        patience=5, mode='min')
        reduce_lr.on_train_begin()                
        reduce_lr.model = self.ZC
        
        early_stop = callbacks.EarlyStopping(
                monitor='val_loss', mode='min',
                        min_delta=0, patience=10)
        early_stop.on_train_begin()                
        early_stop.model = self.ZC

        
        epoch_num = 2000
        batch_size = 64
        
        for epoch in range(epoch_num):
            
            train_inds = np.random.permutation(len(y_train))
            X_train = X_train[train_inds,:]
            y_train = y_train[train_inds]
            
            sd_loss_list = []
            zc_loss_list = []
            
            wlist=[
                self.se_weight_path(epoch),
                self.ze_weight_path(epoch),
                self.sd_weight_path(epoch),
                self.zc_weight_path(epoch),
            ]
            if all([os.path.exists(x) for x in wlist]) and len(old_info_list)>0:
                print(epoch)
                info_list.append(old_info_list[epoch])
                self.SE.load_weights(self.se_weight_path(epoch))
                self.ZE.load_weights(self.ze_weight_path(epoch))
                self.SD.load_weights(self.sd_weight_path(epoch))
                self.ZC.load_weights(self.zc_weight_path(epoch))
                print('skipping epoch',len(info_list))
                continue

            for bX_train,by_train in chunkify(X_train,y_train,batch_size):
                           
                sd_loss = self.SD.train_on_batch(bX_train,bX_train)
                sd_loss_list.append(sd_loss)
                
                self.SE.trainable = False
                for _ in range(3):
                    zc_loss = self.ZC.train_on_batch(bX_train,by_train)
                    zc_loss_list.append(zc_loss)
                self.SE.trainable = True
                 
                
            predZ = self.ZC.predict(X_validation).squeeze()
            z_val_loss = metrics.log_loss(y_validation,predZ)
            predX = self.SD.predict(X_validation).squeeze()
            s_val_loss = metrics.mean_squared_error(X_validation,predX)

            info = {
                'epoch':epoch,
                'sd_loss':float(np.mean(sd_loss_list)),
                'zc_loss':float(np.mean(zc_loss_list)),
                'val_loss': float(z_val_loss),
                'sd_val_loss': float(s_val_loss),
            }
            info_list.append(info)
            print(info)
            self.SE.save_weights(self.se_weight_path(epoch),True)
            self.ZE.save_weights(self.ze_weight_path(epoch),True)
            self.SD.save_weights(self.sd_weight_path(epoch),True)
            self.ZC.save_weights(self.zc_weight_path(epoch),True)
            
            with open(self.history_path, "w") as f:
                f.write(yaml.dump(info_list))
            
            reduce_lr.on_epoch_end(epoch,logs=info)
            early_stop.on_epoch_end(epoch,logs=info)
            
            print('!!lr',reduce_lr.model.optimizer.lr.eval(),reduce_lr.wait)
            print('!!es',early_stop.stopped_epoch,early_stop.wait)
            
            if early_stop.stopped_epoch!=0 and early_stop.stopped_epoch <= epoch:
                break
                
        self.load()

    def predict(self,X,y_true=None):
        if self.is_trained is False:
            self.load()

        y_pred = self.ZC.predict(X)[:,-1]
        y_pred = np.expand_dims(y_pred,axis=-1)
        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true,y_pred)

        return y_pred, logloss
