
'''
Krauss, Christopher, Xuan Anh Do, and Nicolas Huck. "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500." European Journal of Operational Research 259.2 (2017): 689-702.
https://www.econstor.eu/bitstream/10419/130166/1/856307327.pdf
'''
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

def get_simple_nn():
    m_in = Input(shape=(52,))
    drop_out=0.3
    m = Dense(52,)(m_in)
    m = BatchNormalization(axis=-1)(m)
    m = Activation('relu')(m)
    m = Dropout(drop_out)(m)
    m = Dense(52,)(m)
    m = BatchNormalization(axis=-1)(m)
    m = Activation('relu')(m)
    m = Dropout(drop_out)(m)
    m = Dense(2,)(m)
    m_out = Activation('softmax')(m)
    model = Model(inputs=m_in, outputs=m_out)
    return model


np.random.seed(69)


FDNAME = 'tsne_simple_file'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TsneSimple(object):
    def __init__(self,this_dir=THIS_DIR,fdname=FDNAME):
        self.is_trained = False
        self.fdname = fdname
        self.this_dir = this_dir
        self.tsne_weight_file = os.path.join(THIS_DIR,self.fdname,'tsne_{o:03d}_{p:03d}.hdf5')
        
        high_dims=50
        self.tsne_list = [(2,5),(2,10),(2,15),(2,30),(3,15),]
        self.tsne = {}
        for key in self.tsne_list:
            o,p = key
            self.tsne[key] = Parametric_tSNE(high_dims, o, p, all_layers=None)

        
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
        self._load_tsne()
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
        
        batch_size = 64
        for key in self.tsne_list:
            o,p = key
            self.tsne[key].fit(X_train,verbose=1,epochs=2,)
            self.tsne[key].save_model(self.tsne_weight_file.format(o=o,p=p))
        
        x_list = [X_train]
        for key in self.tsne_list:
            x_list.append(self.tsne[key].transform(X_train))
        X_train = np.concatenate(x_list,axis=-1)

        
        x_list = [X_validation]
        for key in self.tsne_list:
            x_list.append(self.tsne[key].transform(X_validation))
        X_validation = np.concatenate(x_list,axis=-1)
        
        snn_callbacks = [
            PlotLoss(self.snn_csv),
            callbacks.EarlyStopping(monitor='val_loss', patience=5),
            callbacks.ModelCheckpoint(self.snn_weight_file),
        ]
        
        batch_size=64
        opt = optimizers.SGD(lr=0.001, clipnorm=0.9)
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
            
        x_list = [X]
        for key in self.tsne_list:
            x_list.append(self.tsne[key].transform(X))
        X = np.concatenate(x_list,axis=-1)
        
        y_pred = self.snn.predict(X)[:,1]

        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true,y_pred)
            print('logloss %r' % logloss)

        return y_pred, logloss

