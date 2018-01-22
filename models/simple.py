import traceback
import numpy as np
from sklearn import metrics

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
from keras.optimizers import SGD, Adam, RMSprop, Adadelta
from keras import optimizers
from keras.utils import np_utils

from keras.models import Model
from keras.layers import Input, Dense, maximum
from keras.layers import Concatenate
from keras.layers.normalization import BatchNormalization
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.layers import PReLU

from .tsne import Parametric_tSNE

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

def unit0(input,
          num,
          drop=0.3,
          axis=-1,
          kernel_regularizer=regularizers.l2(10e-8),
          activity_regularizer=regularizers.l1(10e-8),):
    e = Dense(num,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
        )(input)#l1 l2?
    e = BatchNormalization(axis=axis)(e)
    e = PReLU()(e)
    e = Dropout(drop)(e)
    return e

def unit1(input,
          num,
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

def get_simple_nn():
    m_in = Input(shape=(50,))
    mode = -1
    dropout_rate = 0.3
    m = unit0(m_in,32,axis=mode,drop=dropout_rate)
    m = unit0(m,32,axis=mode,drop=dropout_rate)
    m_out = unit1(m,2,axis=mode,drop=dropout_rate,activation='softmax')
    model = Model(inputs=m_in, outputs=m_out)
    return model


np.random.seed(69)


FDNAME = 'simple_file'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Simple(object):
    def __init__(self,this_dir=THIS_DIR,fdname=FDNAME):
        self.is_trained = False
        self.fdname = fdname
        self.this_dir = this_dir
        self.tsne_weight_file = os.path.join(THIS_DIR,self.fdname,'tsne_{o:03d}_{p:03d}.hdf5')
        
        high_dims=50
        self.tsne_list = [(3,5),(3,15),(3,30),(5,5),(5,15),(5,30)]
        self.tsne = {}
        for key in self.tsne_list:
            num_outputs,perplexity = key
            self.tsne[key] = Parametric_tSNE(high_dims, num_outputs, perplexity, dropout=0.3, all_layers=None)

        
        self.snn_weight_file = os.path.join(THIS_DIR,self.fdname,'snn_{epoch:03d}.hdf5')
        self.snn_csv = os.path.join(THIS_DIR,self.fdname,'snn.csv')
        self.snn = get_simple_nn()
        
    def _load_tsne(self):
        for key in self.tsne_list:
            o,p = key
            self.tsne[key].restore_model(self.tsne_weight_file.format(o=o,p=p))
        
    def _load_snn(self):
        _results=pd.read_csv(self.snn_csv)
        _epoch=np.argmin(_results['val_loss'])
        self.snn.load_weights(self.snn_weight_file.format(epoch=_epoch))
        
    def load(self):
        #self._load_tsne()
        self._load_snn()
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,
            X_validation=None,y_validation=None,X_test=None,**kwargs):
        if X_validation is None or y_validation is None:
            raise IOError()
            
        if os.path.exists(os.path.dirname(self.snn_csv)) is False:
            os.makedirs(os.path.dirname(self.snn_csv))
        
        y_train = np_utils.to_categorical(y_train)
        y_validation = np_utils.to_categorical(y_validation)
        
        train_inds = np.random.permutation(len(y_train))
        X_train = X_train[train_inds,:]
        y_train = y_train[train_inds,:]
        
        snn_callbacks = [
            PlotLoss(self.snn_csv),
            callbacks.ReduceLROnPlateau(monitor='val_loss',
                    factor=0.2,patience=5, mode='min'),
            callbacks.EarlyStopping(monitor='val_loss', patience=10),
            callbacks.ModelCheckpoint(self.snn_weight_file),
        ]
        
        batch_size=64
        #opt = optimizers.SGD(lr=0.001, clipnorm=0.9)
        opt = optimizers.Nadam(lr=0.0001)
        self.snn.compile(loss='binary_crossentropy',optimizer=opt)
        self.snn.fit(X_train,y_train,
                     batch_size=batch_size,epochs=200,
                     validation_data=(X_validation,y_validation),
                     callbacks=snn_callbacks,
                    )
        
        self.load()

    def predict(self,X,y_true=None):
        if self.is_trained is False:
            self.load()
        
        y_pred = self.snn.predict(X)[:,1]

        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true,y_pred)
            print('logloss %r' % logloss)

        return y_pred, logloss

