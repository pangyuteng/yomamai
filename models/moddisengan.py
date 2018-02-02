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

def get_upstreams():
    
    input_shape=8
    mode = -1
    dropout_rate=0.2
    in_dropout_rate=0.1
    
    f_in = Input(shape=(input_shape,))
    
    # specific encoder
    e = unit0(f_in,8,axis=mode,drop=dropout_rate)
    e = unit0(e,8,axis=mode,drop=dropout_rate)
    e = unit1(e,4,axis=mode,drop=dropout_rate,activation='relu')

    #unspecific encoder
    z = unit0(f_in,8,axis=mode,drop=dropout_rate)
    z = unit1(z,8,axis=mode,drop=dropout_rate,activation='relu')
    # act = 'relu'
    # z = unit0(f_in,8,axis=mode,drop=dropout_rate)
    # z=Highway(activation=act)(z)
    # z=Dropout(dropout_rate)(z)
    # z = unit0(z,8,axis=mode,drop=dropout_rate)
    # z=Highway(activation=act)(z)
    # z=Dropout(dropout_rate)(z)
    # z = unit1(z,8,axis=mode,drop=dropout_rate,activation='relu')
    
    se = Model(inputs=f_in, outputs=e)    
    ze = Model(inputs=f_in, outputs=z)
    return se,ze

def get_downstreams(SE,ZE):
    
    input_shape=8
    mode = -1
    dropout_rate=0.2
    
    I = Input(shape=(input_shape,))
    se = SE(I)
    ze = ZE(I)
    
    m = Concatenate(axis=-1)([se,ze])
    
    # specific discriminator    
    d = unit0(m,8,axis=mode,drop=dropout_rate)
    d = unit0(d,8,axis=mode,drop=dropout_rate)
    d = unit1(d,8,axis=mode,drop=dropout_rate,activation='sigmoid')
    SD = Model(inputs=I,outputs=d)
    
    # unspecific classifier
    c = unit0(ze,8,axis=mode,drop=dropout_rate)
    c = unit1(c,1,axis=mode,drop=dropout_rate,activation='sigmoid')
    # act='relu'
    # c = unit0(ze,8,axis=mode,drop=dropout_rate)
    # c=Highway(activation=act)(c)
    # c=Dropout(dropout_rate)(c)
    # c = unit0(ze,8,axis=mode,drop=dropout_rate)
    # c=Highway(activation=act)(c)
    # c=Dropout(dropout_rate)(c)
    # c = unit1(c,1,axis=mode,drop=dropout_rate,activation='sigmoid')
    ZC = Model(inputs=I, outputs=c)

    return SD,ZC


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

def chunkify(x,y, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, x.shape[0], n):
        yield x[i:i + n,:],y[i:i + n],

FDNAME = 'moddisentangle_file'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(THIS_DIR,FDNAME)) is False:
    os.makedirs(os.path.join(THIS_DIR,FDNAME))

class DisentangleModel(object):
    def __init__(self,this_dir=THIS_DIR,fdname=FDNAME):
        self.is_trained = False
        self.this_dir = this_dir
        self.fdname = fdname
        
        self.aec_csv = os.path.join(THIS_DIR,fdname,'aec_file','aec.csv')
        self.aec_dir = os.path.dirname(self.aec_csv)
        self.encoder, self.decoder = get_aec()
        
        self.w_file = os.path.join(this_dir,fdname,'weights.npy')
        self.fold_num = 5
        self._init_nn()
        
    def _init_nn(self):
        self.nn={}
        for n in range(self.fold_num):
            cfn = str(n)
            se_weight_path = os.path.join(self.this_dir,self.fdname,cfn,'se{:03d}.hdf5').format
            ze_weight_path = os.path.join(self.this_dir,self.fdname,cfn,'ze{:03d}.hdf5').format
            sd_weight_path = os.path.join(self.this_dir,self.fdname,cfn,'sd{:03d}.hdf5').format
            zc_weight_path = os.path.join(self.this_dir,self.fdname,cfn,'zc{:03d}.hdf5').format
            history_path = os.path.join(self.this_dir,self.fdname,cfn,'history.yml')

            if os.path.exists(os.path.dirname(history_path)) is False:
                os.makedirs(os.path.dirname(history_path))
                
            SE, ZE = get_upstreams()
            SD, ZC = get_downstreams(SE,ZE)
            
            self.nn[n]={}
            self.nn[n].update(dict(
                se_weight_path=se_weight_path,
                ze_weight_path=ze_weight_path,
                sd_weight_path=sd_weight_path,
                zc_weight_path=zc_weight_path,
                history_path=history_path,
                SE=SE,ZE=ZE,SD=SD,ZC=ZC,
            ))
            
    def _load_aec(self):
        aec_results=pd.read_csv(self.aec_csv)
        aec_epoch=np.argmin(aec_results['val_loss'])+1
        
        aec_weights = glob.glob(os.path.join(self.aec_dir,'weightsAEC.'+'{:03d}'.format(aec_epoch)+'*'))
        self.aec_weight_file = aec_weights[0]
        self.decoder.load_weights(self.aec_weight_file)
        self.encoder.set_weights(self.decoder.get_weights()[:-1])
        
    def _load_nn(self):
        
        for n in range(self.fold_num):
            
            with open(self.nn[n]['history_path'], "r") as f:
                history = yaml.load(f.read())
            epoch = history[np.argmin([x['val_loss'] for x in history])]['epoch']
            
            self.nn[n]['SE'].load_weights(self.nn[n]['se_weight_path'](epoch))
            self.nn[n]['ZE'].load_weights(self.nn[n]['ze_weight_path'](epoch))
            self.nn[n]['SD'].load_weights(self.nn[n]['sd_weight_path'](epoch))
            self.nn[n]['ZC'].load_weights(self.nn[n]['zc_weight_path'](epoch))
        
    def load(self):
        self._load_aec()
        self._load_nn()
        self.w=np.load(self.w_file)
        self.is_trained = True


    def _fit_aec(self,X_train=None,y_train=None,X_validation=None,y_validation=None,X_test=None,**kwargs):
        if X_validation is None or y_validation is None:
            raise IOError()

        lr=0.0001,
        #opt = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        opt = Adam(lr=lr) #<faster than SGD
        loss = 'mse' #<-- lets do gan next MSE sucks.
        #self.encoder.compile(loss=loss, optimizer=opt)
        self.decoder.compile(loss=loss, optimizer=opt)
        
        aec_callbacks = [
            PlotLoss(self.aec_csv),
            callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.2,
                        patience=8, mode='min'),
            callbacks.EarlyStopping(monitor='val_loss', patience=5),
            callbacks.ModelCheckpoint(os.path.join(
                self.aec_dir,'weightsAEC.{epoch:03d}.hdf5')),
        ]
        
        decoder_X_train = X_train
        if X_test is not None:
            decoder_X_train = np.concatenate([X_train,X_test],axis=0).astype('float')
            
        nb_epoch= 200
        batch_size =64
        history = self.decoder.fit(decoder_X_train,decoder_X_train,shuffle=True,
            batch_size=batch_size,epochs=nb_epoch,
            verbose=1,callbacks=aec_callbacks,validation_data=(X_validation,X_validation),)

        self._load_aec()
        
    def fit(self,X_train=None,y_train=None,
            X_validation=None,y_validation=None,
            eras_train=None,eras_validation=None,**kwargs):
        
        if X_validation is None or y_validation is None:
            raise IOError()
        if eras_train is None or eras_validation is None:
            raise IOError()
        
        self._load_aec()
        print('aec loaded')
        #self._fit_aec()
        
        current_fold_num=0

        kf = model_selection.StratifiedKFold(n_splits=self.fold_num,random_state=69,shuffle=True)
        for train_index, test_index in kf.split(X_train,eras_train):
            
            bX_train = X_train[train_index,:]
            by_train = y_train[train_index]
            
            ebX_train = self.encoder.predict(bX_train)
            eX_val = self.encoder.predict(X_validation)
            
            self._fit(current_fold_num,X_train=ebX_train,y_train=by_train,
                      X_validation=eX_val,y_validation=y_validation)
            current_fold_num+=1
        
        # optimize and save weights
        pred_list = []
        for n in range(self.fold_num):
            y_pred=self.nn[n]['ZC'].predict(X_validation)
            pred_list.append(y_pred.squeeze())
    
        self.w = opt.opt_weights(pred_list,y_validation)
        np.save(self.w_file, self.w) 
        
        self.load()
            
            
    
    def _fit(self,current_fold_num,
             X_train=None,y_train=None,
             X_validation=None,y_validation=None,
             X_test=None,**kwargs):
        
        if X_validation is None or y_validation is None:
            raise IOError()
        
        sd_opt = optimizers.Nadam(lr=0.0001)
        self.nn[current_fold_num]['SD'].compile(loss='mse',optimizer=sd_opt)
        
        #zc_opt = optimizers.Nadam(lr=0.00001)
        zc_opt = SGD(lr=0.0001)
        self.nn[current_fold_num]['ZC'].compile(loss='binary_crossentropy',optimizer=zc_opt)
        
        self.nn[current_fold_num]['SE'].trainable = True
        
        info_list = []
        
        old_info_list = []
        if os.path.exists(self.nn[current_fold_num]['history_path']):
            with open(self.nn[current_fold_num]['history_path'],"r") as f:
                old_info_list = yaml.load(f.read())
                
        reduce_lr_sd = callbacks.ReduceLROnPlateau(
                        monitor='sd_val_loss', factor=0.2,
                        patience=5, mode='min')
        reduce_lr_sd.on_train_begin()                
        reduce_lr_sd.model = self.nn[current_fold_num]['SD']
        
        reduce_lr = callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.2,
                        patience=5, mode='min')
        reduce_lr.on_train_begin()                
        reduce_lr.model = self.nn[current_fold_num]['ZC']
        
        early_stop = callbacks.EarlyStopping(
                monitor='val_loss', mode='min',
                        min_delta=0, patience=10)
        early_stop.on_train_begin()                
        early_stop.model = self.nn[current_fold_num]['ZC']
        
        epoch_num = 2000
        batch_size = 32
        
        for epoch in range(epoch_num):
            
            # disable shuffle, too noisy for disentangle.
            train_inds = np.random.permutation(len(y_train))
            X_train = X_train[train_inds,:]
            y_train = y_train[train_inds]
            
            sd_loss_list = []
            zc_loss_list = []
            
            wlist=[
                self.nn[current_fold_num]['se_weight_path'](epoch),
                self.nn[current_fold_num]['ze_weight_path'](epoch),
                self.nn[current_fold_num]['sd_weight_path'](epoch),
                self.nn[current_fold_num]['zc_weight_path'](epoch),
            ]
            if all([os.path.exists(x) for x in wlist]) and len(old_info_list)>0:
                print(epoch)
                info_list.append(old_info_list[epoch])
                self.nn[current_fold_num]['SE'].load_weights(self.nn[current_fold_num]['se_weight_path'](epoch))
                self.nn[current_fold_num]['ZE'].load_weights(self.nn[current_fold_num]['ze_weight_path'](epoch))
                self.nn[current_fold_num]['SD'].load_weights(self.nn[current_fold_num]['sd_weight_path'](epoch))
                self.nn[current_fold_num]['ZC'].load_weights(self.nn[current_fold_num]['zc_weight_path'](epoch))
                print('skipping epoch',len(info_list))
                continue

            for _bX_train,_by_train in chunkify(X_train,y_train,batch_size):
                               
                # # train by class 
                # for myclass in [0,1]:                    
                #     inds = np.where(_by_train==myclass)                    
                #     if len(inds[0]) <2:
                #         continue                        
                #     bX_train = _bX_train[inds,:].squeeze()
                #     by_train = _by_train[inds].squeeze()
                
                bX_train = _bX_train
                by_train = _by_train
                
                sd_loss = self.nn[current_fold_num]['SD'].train_on_batch(bX_train,bX_train)
                sd_loss_list.append(sd_loss)

                self.nn[current_fold_num]['ZC'].trainable = False
                notmyclass = 1 - by_train
                _zc_loss = self.nn[current_fold_num]['ZC'].train_on_batch(bX_train,notmyclass)
                self.nn[current_fold_num]['ZC'].trainable = True

                self.nn[current_fold_num]['SE'].trainable = False
                for _ in range(3):
                    zc_loss = self.nn[current_fold_num]['ZC'].train_on_batch(bX_train,by_train)
                    zc_loss_list.append(zc_loss)
                self.nn[current_fold_num]['SE'].trainable = True
                 
                
            predZ = self.nn[current_fold_num]['ZC'].predict(X_validation).squeeze()
            z_val_loss = metrics.log_loss(y_validation,predZ)
            predX = self.nn[current_fold_num]['SD'].predict(X_validation).squeeze()
            s_val_loss = metrics.mean_squared_error(X_validation,predX)

            info = {
                'epoch':epoch,
                'sd_loss':float(np.mean(sd_loss_list)),
                'zc_loss':float(np.mean(zc_loss_list)),
                'val_loss': float(z_val_loss),
                'sd_val_loss': float(s_val_loss),
            }
            info_list.append(copy.deepcopy(info))
            print(info)
            
            self.nn[current_fold_num]['SE'].save_weights(self.nn[current_fold_num]['se_weight_path'](epoch))
            self.nn[current_fold_num]['ZE'].save_weights(self.nn[current_fold_num]['ze_weight_path'](epoch))
            self.nn[current_fold_num]['SD'].save_weights(self.nn[current_fold_num]['sd_weight_path'](epoch))
            self.nn[current_fold_num]['ZC'].save_weights(self.nn[current_fold_num]['zc_weight_path'](epoch))
            
            with open(self.nn[current_fold_num]['history_path'], "w") as f:
                f.write(yaml.dump(info_list))
           
            reduce_lr_sd.on_epoch_end(epoch,logs=info)
            reduce_lr.on_epoch_end(epoch,logs=info)
            early_stop.on_epoch_end(epoch,logs=info)
            
            print('!!lr',reduce_lr.model.optimizer.lr.eval(),reduce_lr.wait)
            print('!!es',early_stop.stopped_epoch,early_stop.wait)
            
            if early_stop.stopped_epoch!=0 and early_stop.stopped_epoch <= epoch:
                break
                

    def predict(self,X,y_true=None):
        if self.is_trained is False:
            self.load()

        eX = self.encoder.predict(X)
        pred_list = []
        for n in range(self.fold_num):
            y_pred=self.nn[n]['ZC'].predict(eX)
            pred_list.append(y_pred.squeeze())
        
        pred_arr = np.array(pred_list)
        y_pred = np.dot(self.w,pred_arr)
        print(self.w,pred_arr.shape,y_pred.shape)
        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true,y_pred)
            print('logloss %r' % logloss)

        return y_pred, logloss
    