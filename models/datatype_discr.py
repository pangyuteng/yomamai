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


def get_ddiscr(
    input_shape=50,
    mode = -1,
    dropout_rate=0.4):
    f_in = Input(shape=(input_shape,))
    
    e = Dense(28)(f_in)
    e = BatchNormalization(axis=mode)(e)
    e = PReLU()(e)
    e = Dropout(dropout_rate)(e)
    e = Dense(28)(e)
    e = BatchNormalization(axis=mode)(e)
    e = PReLU()(e)
    e = Dropout(dropout_rate)(e)
    e = Dense(1)(e)
    f_out = Activation('sigmoid')(e)
    model = Model(inputs=f_in, outputs=f_out)
    return model

    
FDNAME = 'ddiscr_file'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class DDiscrModel(object):
    def __init__(self,this_dir=THIS_DIR,fdname=FDNAME):
        self.is_trained = False
        self.this_dir = this_dir
        self.fdname = fdname
        
        self.ddiscr_csv = os.path.join(self.this_dir,self.fdname,'ddiscr.csv')
        
        self.ddiscr = get_ddiscr()
        
    def _load_ddiscr(self):
        res_results=pd.read_csv(self.ddiscr_csv)
        res_epoch=np.argmin(res_results['val_loss'])+1
        res_weights = glob.glob(os.path.join(self.this_dir,self.fdname,'weightsDDISCR.'+'{:03d}'.format(res_epoch)+'*'))
        self.ddiscr_weight_file = res_weights[0]
        self.ddiscr.load_weights(self.ddiscr_weight_file)

    def load(self):
        self._load_ddiscr()
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,X_validation=None,y_validation=None,X_test=None,**kwargs):
        
        if X_test is None or X_validation is None:
            raise IOError()
            
        _X = np.concatenate([X_train,X_validation,X_test],axis=0)
        _y = np.concatenate([[0]*X_train.shape[0],[0]*X_validation.shape[0],[1]*X_test.shape[0]],axis=0)

        lr=0.001
        opt = Adam(lr=lr)
        self.ddiscr.compile(loss='binary_crossentropy', optimizer=opt)
        
        skf = cross_validation.StratifiedKFold(_y,n_folds=2,shuffle=True,random_state=69)
        for train_ind,val_ind in skf:
            np.random.shuffle(train_ind)
            np.random.shuffle(val_ind)
            X_train = _X[train_ind,:]
            y_train = _y[train_ind]
            X_val = _X[val_ind,:]
            y_val = _y[val_ind]

            nb_epoch= 20
            batch_size = 64

            mcallbacks = [
                PlotLoss(self.ddiscr_csv),
                callbacks.EarlyStopping(monitor='val_loss', patience=5),
                callbacks.ModelCheckpoint(os.path.join(
                    self.this_dir,self.fdname,'weightsDDISCR.{epoch:03d}.hdf5')),
            ]
            history = self.ddiscr.fit(X_train, y_train,shuffle=True,
                batch_size=batch_size, epochs=nb_epoch,
                verbose=1, validation_data=(X_validation,y_validation), callbacks=mcallbacks)
            break
        self.load()

    def predict(self,X,y_true=None):
        if self.is_trained is False:
            self.load()

        y_pred = self.ddiscr.predict(X).squeeze()

        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true,y_pred)

        return y_pred, logloss
