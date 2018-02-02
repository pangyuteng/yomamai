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

def get_res(
    input_shape=50,
    dropout_rate=0.3,):
        
    f_in = Input(shape=(input_shape,))
    m=block(f_in,node_num=[32,32,8],dropout_rate=dropout_rate)
    #m=block(m,node_num=[32,32,8],dropout_rate=dropout_rate)
    m=block(m,node_num=[16],dropout_rate=dropout_rate)
    m=block(m,node_num=[16],dropout_rate=dropout_rate)
    m = Dense(1)(m)
    f_out = Activation('sigmoid')(m)
    model = Model(inputs=f_in, outputs=f_out)
    return model



np.random.seed(69)


FDNAME = 'res_file'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Res(object):
    def __init__(self,this_dir=THIS_DIR,fdname=FDNAME):
        self.is_trained = False
        self.fdname = fdname
        self.this_dir = this_dir
        
        self.snn_weight_file = os.path.join(THIS_DIR,self.fdname,'nn_{epoch:03d}.hdf5')
        self.snn_csv = os.path.join(THIS_DIR,self.fdname,'nn.csv')
        self.snn = get_res()
        
    def _load_snn(self):
        _results=pd.read_csv(self.snn_csv)
        _epoch=np.argmin(_results['val_loss'])
        self.snn.load_weights(self.snn_weight_file.format(epoch=_epoch))
        
    def load(self):
        self._load_snn()
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,
            X_validation=None,y_validation=None,X_test=None,**kwargs):
        if X_validation is None or y_validation is None:
            raise IOError()
            
        if os.path.exists(os.path.dirname(self.snn_csv)) is False:
            os.makedirs(os.path.dirname(self.snn_csv))
        
        #y_train = np_utils.to_categorical(y_train)
        #y_validation = np_utils.to_categorical(y_validation)
        
        train_inds = np.random.permutation(len(y_train))
        X_train = X_train[train_inds,:]
        y_train = y_train[train_inds]
        
        snn_callbacks = [
            PlotLoss(self.snn_csv),
            callbacks.ReduceLROnPlateau(monitor='val_loss',
                    factor=0.2,patience=5, mode='min'),
            callbacks.EarlyStopping(monitor='val_loss', patience=3),
            callbacks.ModelCheckpoint(self.snn_weight_file),
        ]
        
        batch_size=64
        
        #opt = optimizers.Nadam(lr=0.0001)
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
        
        y_pred = self.snn.predict(X).squeeze()

        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true,y_pred)
            print('logloss %r' % logloss)

        return y_pred, logloss

