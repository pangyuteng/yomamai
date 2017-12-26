
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


# https://groups.google.com/forum/#!topic/keras-users/rQ2fjaNbX5w
def maxoutdense0(m,num,dropout_rate=0.5,n_pieces=2,mode=-1,input_dropout_rate=None,**dense_args):
    if input_dropout_rate is not None:
        d0 = Dropout(input_dropout_rate)(m)
        d0 = Dense(num, **dense_args)(d0)        
        d1 = Dropout(input_dropout_rate)(m)
        d1 = Dense(num, **dense_args)(d1)
        m = maximum([d0,d1])
    else:
        m = maximum([Dense(num, **dense_args)(m) for _ in range(n_pieces)])
    m = BatchNormalization(axis=mode)(m)
    m = Dropout(dropout_rate)(m)
    return m

# https://groups.google.com/forum/#!topic/keras-users/rQ2fjaNbX5w
def mydense0(m,num,dropout_rate=0.5,n_pieces=2,mode=-1,input_dropout_rate=None,**dense_args):
    if input_dropout_rate is not None:
        d0 = Dropout(input_dropout_rate)(m)
        d0 = Dense(num, **dense_args)(d0)
        d1 = Dropout(input_dropout_rate)(m)
        d1 = Dense(num, **dense_args)(d1)
    else:
        d0 = Dense(num, **dense_args)(m)
        d1 = Dense(num, **dense_args)(m)
    m = maximum([d0,d1])
    m = BatchNormalization(axis=mode)(m)
    m = PReLU()(m) # add in another activation layer, why the hell not.
    m = Dropout(dropout_rate)(m)
    return m

def get_dnn0(input_shape=50,):

    kwargs = dict(
        dropout_rate=0.5,
        n_pieces=2,
        mode=-1,
    )
    
    f_in = Input(shape=(input_shape,))
    input_dropout_rate=0.1
    m = maxoutdense0(f_in,input_shape,
            input_dropout_rate=input_dropout_rate,
            activity_regularizer=regularizers.l1(0.00001),
            **kwargs,
       )
    m = maxoutdense0(m,16,**kwargs)
    m = maxoutdense0(m,8,**kwargs)
    m = Dense(2)(m)
    f_out = Activation('softmax')(m)
    m = Model(inputs=f_in, outputs=f_out)
    return m


def get_dnn1(input_shape=50,):

    kwargs = dict(
        dropout_rate=0.5,
        n_pieces=2,
        mode=-1,
    )
    
    f_in = Input(shape=(input_shape,))
    input_dropout_rate=0.1
    m = maxoutdense0(f_in,input_shape,
            input_dropout_rate=input_dropout_rate,
            activity_regularizer=regularizers.l1(0.00001),
            **kwargs,
       )
    m = maxoutdense0(m,16,**kwargs)
    m = maxoutdense0(m,16,**kwargs)
    m = maxoutdense0(m,8,**kwargs)
    m = maxoutdense0(m,8,**kwargs)
    m = Dense(2)(m)
    f_out = Activation('softmax')(m)
    m = Model(inputs=f_in, outputs=f_out)
    return m

def get_dnn2(input_shape=50,):

    kwargs = dict(
        dropout_rate=0.5,
        n_pieces=2,
        mode=-1,
    )
    
    f_in = Input(shape=(input_shape,))
    input_dropout_rate=0.1
    m = mydense0(f_in,input_shape,
            input_dropout_rate=input_dropout_rate,
            activity_regularizer=regularizers.l1(0.00001),
            **kwargs,
       )
    m = mydense0(m,16,**kwargs)
    m = mydense0(m,8,**kwargs)
    m = Dense(2)(m)
    f_out = Activation('softmax')(m)
    m = Model(inputs=f_in, outputs=f_out)
    return m

FDNAME = 'krauss_file'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_LIST = [
    (get_dnn0,os.path.join(THIS_DIR,FDNAME,'dnnloss0.csv')),
    (get_dnn1,os.path.join(THIS_DIR,FDNAME,'dnnloss1.csv')),
    (get_dnn2,os.path.join(THIS_DIR,FDNAME,'dnnloss2.csv')),
]

class KraussModel(object):
    def __init__(self,this_dir=THIS_DIR,fdname=FDNAME):
        self.is_trained = False
        self.fdname = fdname
        self.this_dir = this_dir
        self.dnn_weight_file = os.path.join(THIS_DIR,self.fdname,'weightsDNN_{epoch:03d}.hdf5')
        get_dnn ,self.dnnloss_csv = MODEL_LIST[2]
        self.dnn = get_dnn()
    def _load_dnn(self):
        _results=pd.read_csv(self.dnnloss_csv)
        _epoch=np.argmin(_results['val_loss'])+1
        self.dnn.load_weights(self.dnn_weight_file.format(epoch=_epoch))

    def load(self):
        self._load_dnn()
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,
            X_validation=None,y_validation=None,X_test=None,**kwargs):
        if X_validation is None or y_validation is None:
            raise IOError()
        y_train = np_utils.to_categorical(y_train)
        y_validation = np_utils.to_categorical(y_validation)
        lr=0.001,
        opt = Adadelta(lr=lr)
        self.dnn.compile(loss='categorical_crossentropy', optimizer=opt)

        _callbacks = [
            PlotLoss(self.dnnloss_csv),
            callbacks.EarlyStopping(monitor='val_loss', patience=20),
            callbacks.ModelCheckpoint(self.dnn_weight_file),
        ]

        nb_epoch= 2000
        batch_size = 128

        history = self.dnn.fit(X_train, y_train,shuffle=True,
            batch_size=batch_size, epochs=nb_epoch,
            verbose=1,callbacks=_callbacks,
            validation_data=(X_validation,y_validation),)
                               

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

