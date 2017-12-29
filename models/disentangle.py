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

def get_upstreams():
    
    input_shape=50
    mode = -1
    dropout_rate=0.3
    in_dropout_rate=0.1
    
    f_in = Input(shape=(input_shape,))
    
    # specific encoder
    e = Dense(32)(f_in)
    e = BatchNormalization(axis=mode)(e)
    e = LeakyReLU()(e)
    e = Dropout(dropout_rate)(e)
    e = Dense(16)(e)
    e = BatchNormalization(axis=mode)(e)
    e = LeakyReLU()(e)
    e = Dropout(dropout_rate)(e)
    e = Dense(16)(e)
    e = BatchNormalization(axis=mode)(e)
    e = LeakyReLU()(e)
    e = Dropout(dropout_rate)(e)
    e = Dense(8)(e) 
    e = BatchNormalization(axis=mode)(e)
    e = Activation('tanh')(e) #?

    #unspecific encoder
    z = Dense(128)(f_in)
    z = BatchNormalization(axis=mode)(z)
    z = LeakyReLU()(z)
    z = Dropout(dropout_rate)(z)
    z = Dense(64)(z)
    z = BatchNormalization(axis=mode)(z)
    z = LeakyReLU()(z)
    z = Dropout(dropout_rate)(z)
    z = Dense(64)(z)
    z = BatchNormalization(axis=mode)(z)
    z = LeakyReLU()(z)
    z = Dropout(dropout_rate)(z)
    z = Dense(32)(z)
    z = BatchNormalization(axis=mode)(z)
    z = Activation('tanh')(z) #?
     
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
    d = Dense(42)(m)
    d = BatchNormalization(axis=mode)(d)
    d = LeakyReLU()(d)
    d = Dropout(dropout_rate)(d)
    d = Dense(64)(d)
    d = BatchNormalization(axis=mode)(d)
    d = LeakyReLU()(d)
    d = Dropout(dropout_rate)(d)
    d = Dense(64)(d)
    d = BatchNormalization(axis=mode)(d)
    d = LeakyReLU()(d)
    d = Dropout(dropout_rate)(d)    
    d = Dense(128)(d) 
    d = BatchNormalization(axis=mode)(d)
    d = LeakyReLU()(d)
    d = Dropout(dropout_rate)(d)
    d = Dense(50)(d)
    d = Activation('linear')(d)
    SD = Model(inputs=I,outputs=d)
    
    # unspecific classifier
    c = Dense(32)(ze) #follow the paper first, next try `m`?
    c = BatchNormalization(axis=mode)(c)
    c = LeakyReLU()(c)
    c = Dropout(dropout_rate)(c)
    c = Dense(16)(c)
    c = BatchNormalization(axis=mode)(c)
    c = LeakyReLU()(c)
    c = Dropout(dropout_rate)(c)
    c = Dense(1)(c) # use softmax with dense of 2?
    c = Activation('sigmoid')(c)
    ZC = Model(inputs=I,outputs=c)
    
    return SD,ZC

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
        sd_opt = Adam(lr=0.0000001)
        self.SD.compile(loss='mse',optimizer=sd_opt)
        
        zc_opt = SGD(lr=0.001)
        self.ZC.compile(loss='binary_crossentropy',optimizer=zc_opt)
        
        self.SE.trainable = True
        
        info_list = []
        
        old_info_list = []
        if os.path.exists(self.history_path):
            with open(self.history_path,"r") as f:
                old_info_list = yaml.load(f.read())

        epoch_num = 200
        batch_size = 64
        
        for epoch in range(epoch_num):
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
               
                #predX = SD.predict()
                #predZ = ZC.predict()
            
                sd_loss = self.SD.train_on_batch(bX_train,bX_train)
                sd_loss_list.append(sd_loss)
                
                self.SE.trainable = False
                for _ in range(3):
                    zc_loss = self.ZC.train_on_batch(bX_train,by_train)
                    zc_loss_list.append(zc_loss)
                self.SE.trainable = True
                
                # train the SDD with real and fake dataset
                
                # train the stacked-SDD
                
                # train the ZCD with real and fake dataset 50+label real, 50+pred fake
                
                # train the scaked-ZCD
                
                
            pred = self.ZC.predict(X_validation).squeeze()
            val_loss = metrics.log_loss(y_validation,pred)
            info = {
                'epoch':epoch,
                'sd_loss':float(np.mean(sd_loss_list)),
                'zc_loss':float(np.mean(zc_loss_list)),
                'val_loss': float(val_loss),
            }
            info_list.append(info)
            print(info)
            self.SE.save_weights(self.se_weight_path(epoch),True)
            self.ZE.save_weights(self.ze_weight_path(epoch),True)
            self.SD.save_weights(self.sd_weight_path(epoch),True)
            self.ZC.save_weights(self.zc_weight_path(epoch),True)
            
            with open(self.history_path, "w") as f:
                f.write(yaml.dump(info_list))
        
        self.load()

    def predict(self,X,y_true=None):
        if self.is_trained is False:
            self.load()

        y_pred = self.ZC.predict(X)[:,-1]

        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true,y_pred)

        return y_pred, logloss
