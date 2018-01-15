
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
from keras.utils import np_utils

from keras.models import Model
from keras.layers import Input, Dense, maximum
from keras.layers import Concatenate
from keras.layers.normalization import BatchNormalization
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.layers import PReLU

from .ptsne import Parametric_tSNE

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

def get_dnn0():
    input_shape=50
    m = Model(inputs=f_in, outputs=f_out)
    return m


np.random.seed(69)


FDNAME = 'tsne_simple_file'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TsneSimple(object):
    def __init__(self,this_dir=THIS_DIR,fdname=FDNAME):
        self.is_trained = False
        self.fdname = fdname
        self.this_dir = this_dir
        self.tsne_weight_file = os.path.join(THIS_DIR,self.fdname,'tsne_{epoch:03d}.hdf5')
        self.tsne_csv = os.path.join(THIS_DIR,self.fdname,'tsne.csv')
        
        num_outputs=2
        high_dims=50
        perplexity=30
        self.tsne = Parametric_tSNE(high_dims, num_outputs, perplexity, all_layers=None)
        
    def _load_tsne(self):
        _results=pd.read_csv(self.dnnloss_csv)
        _epoch=np.argmin(_results['val_loss'])+1
        self.dnn.load_weights(self.dnn_weight_file.format(epoch=_epoch))

    def load(self):
        self._load_tsne()
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,
            X_validation=None,y_validation=None,X_test=None,**kwargs):
        if X_validation is None or y_validation is None:
            raise IOError()
            
        if os.path.exists(os.path.dirname(self.tsne_csv)) is False:
            os.makedirs(os.path.dirname(self.tsne_csv))
        
        train_inds = np.random.permutation(len(y_train))
        X_train = X_train[train_inds,:]
        y_train = y_train[train_inds]
        
                    
        tsne_callbacks = [
            PlotLoss(self.tsne_csv),
            callbacks.EarlyStopping(monitor='val_loss', patience=5),
            callbacks.ModelCheckpoint(self.tsne_weight_file),
        ]
        batch_size = 64,
        self.tsne.fit(X_train,X_train,
              verbose=1,epochs=100,
              shuffle=True,batch_size=batch_size,
              callbacks=tsne_callbacks,
              validation_data=(X_validation,y_validation))
        
        
#        self.dnn.fit(X_train, y_train
#             batch_size=batch_size, epochs=nb_epoch,
#             verbose=1,callbacks=_callbacks,
#             ,
        
#         y_train = np_utils.to_categorical(y_train)
#         y_validation = np_utils.to_categorical(y_validation)
#         lr=0.001,
#         opt = Adadelta(lr=lr)
#         self.dnn.compile(loss='categorical_crossentropy', optimizer=opt)

#         _callbacks = [
#             PlotLoss(self.dnnloss_csv),
#             callbacks.EarlyStopping(monitor='val_loss', patience=20),
#             callbacks.ModelCheckpoint(self.dnn_weight_file),
#         ]

#         nb_epoch= 2000
#         batch_size = 128

#         history = self.dnn.fit(X_train, y_train,shuffle=True,
#             batch_size=batch_size, epochs=nb_epoch,
#             verbose=1,callbacks=_callbacks,
#             validation_data=(X_validation,y_validation),)   

        self.load()

    def predict(self,X,y_true=None):
        if self.is_trained is False:
            self.load()

        y_pred = self.dnn.predict(X)[:,1]
        print(y_pred.shape)

        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true,y_pred)
            print('logloss %r' % logloss)

        return y_pred, logloss

