import numpy as np
from sklearn import metrics
import yaml
import copy
import warnings, traceback

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
from sklearn import model_selection

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dropout, Activation
from keras import layers
from keras.optimizers import SGD, Adam, RMSprop
from keras import optimizers
from keras.utils import np_utils


from keras import initializers, regularizers, activations, constraints
from keras.engine import Layer, InputSpec
from keras.models import Model, model_from_json
from keras.layers import Input, Embedding, TimeDistributed, Dense, Dropout, Reshape, Concatenate, LSTM, Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K

from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Concatenate
from keras.layers.normalization import BatchNormalization
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.layers import PReLU
from . import opt

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

#https://gist.github.com/codekansas/3d314f6ea1fcdb1d588379ceda3efc94
#https://github.com/keras-team/keras/blob/master/keras/legacy/layers.py
class Highway(Layer):
    """Densely connected highway network.
    Highway layers are a natural extension of LSTMs to feedforward networks.
    # Arguments
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # References
        - [Highway Networks](http://arxiv.org/abs/1505.00387v2)
    """

    def __init__(self,
                 init='glorot_uniform',
                 activation=None,
                 weights=None,
                 W_regularizer=None,
                 b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 input_dim=None,
                 **kwargs):
        warnings.warn('The `Highway` layer is deprecated '
                      'and will be removed after 06/2017.')
        if 'transform_bias' in kwargs:
            kwargs.pop('transform_bias')
            warnings.warn('`transform_bias` argument is deprecated and '
                          'has been removed.')
        self.init = initializers.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(),
                                    shape=(None, input_dim))

        self.W = self.add_weight((input_dim, input_dim),
                                 initializer=self.init,
                                 name='W',
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.W_carry = self.add_weight((input_dim, input_dim),
                                       initializer=self.init,
                                       name='W_carry')
        if self.bias:
            self.b = self.add_weight((input_dim,),
                                     initializer='zero',
                                     name='b',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.b_carry = self.add_weight((input_dim,),
                                           initializer='one',
                                           name='b_carry')
        else:
            self.b_carry = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, x):
        y = K.dot(x, self.W_carry)
        if self.bias:
            y += self.b_carry
        transform_weight = activations.sigmoid(y)
        y = K.dot(x, self.W)
        if self.bias:
            y += self.b
        act = self.activation(y)
        act *= transform_weight
        output = act + (1 - transform_weight) * x
        return output

    def get_config(self):
        config = {'init': initializers.serialize(self.init),
                  'activation': activations.serialize(self.activation),
                  'W_regularizer': regularizers.serialize(self.W_regularizer),
                  'b_regularizer': regularizers.serialize(self.b_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'W_constraint': constraints.serialize(self.W_constraint),
                  'b_constraint': constraints.serialize(self.b_constraint),
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(Highway, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def unit0(input,num,
          drop=0.3,axis=-1,
          kernel_regularizer=None, #regularizers.l2(10e-5),
          activity_regularizer=None): #regularizers.l1(10e-5),):
    e = Dense(num,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
        )(input) #l1 l2?
    e = BatchNormalization(axis=axis)(e)
    e = PReLU()(e) # ?LeakyReLU
    e = Dropout(drop)(e)
    return e

def unit1(input,num,
          drop=0.3,
          axis=-1,
          activation='sigmoid',
          kernel_regularizer=None, #regularizers.l2(10e-5),
          activity_regularizer=None): #regularizers.l1(10e-5),):
    e = Dense(num,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer,
        )(input)
    e = BatchNormalization(axis=axis)(e)
    e = Activation(activation)(e)
    return e

def get_highway():
    
    input_shape=50
    mode = -1
    dropout_rate=0.75
    
    f_in = Input(shape=(input_shape,))
    
    act = 'relu'
    z = unit0(f_in,16,axis=mode,drop=dropout_rate)
    z=Highway(activation=act)(z)
    z=Dropout(dropout_rate)(z)
    z = unit0(z,8,axis=mode,drop=dropout_rate)
    z=Highway(activation=act)(z)
    z=Dropout(dropout_rate)(z)
    z = unit1(z,1,axis=mode,drop=dropout_rate,activation='relu')
    
    ZE = Model(inputs=f_in, outputs=z)
    return ZE


np.random.seed(69)


FDNAME = 'highwaykfold_file'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class HighwayModel(object):
    def __init__(self,this_dir=THIS_DIR,fdname=FDNAME):
        self.is_trained = False
        self.fdname = fdname
        self.this_dir = this_dir
        
        self.w_file = os.path.join(this_dir,fdname,'weights.npy')
        self.fold_num = 10
        self._init_nn()
        
    def _init_nn(self):
        self.nn={}
        for n in range(self.fold_num):
            nn_weight_file = os.path.join(THIS_DIR,self.fdname,str(n),'nn_{epoch:03d}.hdf5')
            nn_csv = os.path.join(THIS_DIR,self.fdname,str(n),'nn.csv')
            self.nn[n]={
                'nn_weight_file':nn_weight_file,
                'nn_csv':nn_csv,
                'nn':get_highway(),
            }
        
    def _load_nn(self):
        for n in range(self.fold_num):
            nn_weight_file = self.nn[n]['nn_weight_file']
            nn_csv = self.nn[n]['nn_csv']
            _results=pd.read_csv(nn_csv)
            _epoch=np.argmin(_results['val_loss'])+1
            self.nn[n]['nn'].load_weights(nn_weight_file.format(epoch=_epoch))
        
    def load(self):
        self._load_nn()
        self.w=np.load(self.w_file)
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,
            X_validation=None,y_validation=None,
            eras_train=None,eras_validation=None,**kwargs):
        
        if X_validation is None or y_validation is None:
            raise IOError()
            
        if eras_train is None or eras_validation is None:
            raise IOError()
        
        n=0
        kf = model_selection.StratifiedKFold(n_splits=self.fold_num,random_state=69,shuffle=True)
        for train_index, test_index in kf.split(X_train,eras_train):
            _X_train = X_train[train_index,:]
            _y_train = y_train[train_index]
            self._fit(n,_X_train,_y_train,X_validation,y_validation)
            n+=1
            
        
        # optimize and save weights
        pred_list = []
        for n in range(self.fold_num):
            y_pred=self.nn[n]['nn'].predict(X_validation)
            pred_list.append(y_pred.squeeze())
    
        self.w = opt.opt_weights(pred_list,y_validation)
        np.save(self.w_file, self.w) 
        
        self.load()
        
    def _fit(self,_n,X_train,y_train,X_validation,y_validation):
            
        nn_csv = self.nn[_n]['nn_csv']
        nn_weight_file = self.nn[_n]['nn_weight_file']
        
        if os.path.exists(os.path.dirname(nn_csv)) is False:
            os.makedirs(os.path.dirname(nn_csv))
                
        train_inds = np.random.permutation(len(y_train))
        X_train = X_train[train_inds,:]
        y_train = y_train[train_inds]
        
        snn_callbacks = [
            PlotLoss(nn_csv),
            callbacks.ReduceLROnPlateau(monitor='val_loss',
                    factor=0.2,patience=5, mode='min'),
            callbacks.EarlyStopping(monitor='val_loss', patience=10),
            callbacks.ModelCheckpoint(nn_weight_file),
        ]
        
        batch_size=64
        
        opt = optimizers.Nadam(lr=0.0001)
        self.nn[_n]['nn'].compile(loss='binary_crossentropy',optimizer=opt)
        self.nn[_n]['nn'].fit(X_train,y_train,
                     batch_size=batch_size,epochs=200,
                     validation_data=(X_validation,y_validation),
                     callbacks=snn_callbacks,
                    )
        

    def predict(self,X,y_true=None):
        if self.is_trained is False:
            self.load()

        pred_list = []
        for n in range(self.fold_num):
            y_pred=self.nn[n]['nn'].predict(X)
            pred_list.append(y_pred.squeeze())
        
        pred_arr = np.array(pred_list)
        y_pred = np.dot(self.w,pred_arr)
        print(self.w,pred_arr.shape,y_pred.shape)
        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true,y_pred)
            print('logloss %r' % logloss)

        return y_pred, logloss


