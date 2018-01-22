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

def block(m,node_num=[32,32,16],dropout_rate=0.0,mode=-1,cons=False):
    merge_list = []
    for n in node_num:
        m = Dense(n)(m)
        m = BatchNormalization(axis=mode)(m)
        m = LeakyReLU()(m)
        m = Dropout(dropout_rate)(m)
        merge_list.append(m)
    if len(merge_list) > 1:
        m = Concatenate(axis=-1)([merge_list[0],merge_list[-1]])
        if cons is True:
            m = Dense(node_num[-1])(m)
            m = BatchNormalization(axis=mode)(m)
            m = LeakyReLU()(m)
            m = Dropout(dropout_rate)(m)

    elif len(merge_list) == 1:
        pass
    return m


def get_aec(
    input_shape=50,
    mode = -1,
    dropout_rate=0.3):
    f_in = Input(shape=(input_shape,))
    e = Dense(32)(f_in)
    e = BatchNormalization(axis=mode)(e)
    e = LeakyReLU()(e)
    e = Dropout(dropout_rate)(e)
    e = Dense(8)(e)
    e = BatchNormalization(axis=mode)(e)
    e = LeakyReLU()(e)
    e = Dropout(dropout_rate)(e)
    d = Dense(32)(e)
    d = BatchNormalization(axis=mode)(d)
    d = LeakyReLU()(d)
    d = Dropout(dropout_rate)(d)    
    d = Dense(input_shape+1)(d)
    f_out = Activation('sigmoid')(d)
    # via dimension reduction followed by reconstruction
    # the additional dimension is added output shape to account for the final classification
    # the reconstruction and predicted classification is fed to discriminator 
    # to differentiate if data (feature0,feature1...,target) is real or not
    encoder = Model(inputs=f_in, outputs=e)
    decoder = Model(inputs=f_in, outputs=f_out)
    return encoder, decoder


def get_discriminator(
    input_shape=51,
    dropout_rate=0.3,):
    mode = -1
    f_in = Input(shape=(input_shape,))
    
    e = Dense(51)(f_in)
    e = BatchNormalization(axis=mode)(e)
    e = LeakyReLU()(e)
    e = Dropout(dropout_rate)(e)
    e = Dense(32)(e)
    e = BatchNormalization(axis=mode)(e)
    e = LeakyReLU()(e)
    e = Dropout(dropout_rate)(e)
    e = Dense(1)(e)
    f_out = Activation('sigmoid')(e)
    model = Model(inputs=f_in, outputs=f_out)
    return model

def get_gan(generator, discriminator,input_shape=50,):
    I = Input(shape=(input_shape,))
    G = generator(I)
    D = discriminator(G)
    D.trainable = False
    model = Model(inputs=I,outputs=D)
    return model

def chunkify(x,y, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, x.shape[0], n):
        yield x[i:i + n,:],y[i:i + n,:],

FDNAME = 'ganmore_file'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(THIS_DIR,FDNAME)) is False:
    os.makedirs(os.path.join(THIS_DIR,FDNAME))

class GanMoreModel(object):
    def __init__(self,this_dir=THIS_DIR,fdname=FDNAME):
        self.is_trained = False
        self.this_dir = this_dir
        self.fdname = fdname
        self.gan_weight_path = lambda x: os.path.join(self.this_dir,self.fdname,'gan{:03d}.hdf5'.format(x)) 
        self.discr_weight_path = lambda x: os.path.join(self.this_dir,self.fdname,'discr{:03d}.hdf5'.format(x))
        self.decoder_weight_path = lambda x: os.path.join(self.this_dir,self.fdname,'decoder{:03d}.hdf5'.format(x))
        self.history_path = os.path.join(self.this_dir,self.fdname,'history.yml')
        self.encoder, self.decoder = get_aec()
        self.discr = get_discriminator()
        self.gan = get_gan(self.decoder,self.discr)

    def _load(self):
        with open(self.history_path, "r") as f:
            history = yaml.load(f.read())
        epoch = history[np.argmin([x['val_loss'] for x in history])]['epoch']
        
        self.gan.load_weights(self.gan_weight_path(epoch))
        self.discr.load_weights(self.discr_weight_path(epoch))
        self.decoder.load_weights(self.decoder_weight_path(epoch))
        self.encoder.set_weights(self.decoder.get_weights()[:-1])

    def load(self):
        self._load()
        
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,X_validation=None,y_validation=None,X_test=None,**kwargs):
        if X_validation is None or y_validation is None:
            raise IOError()
            
        decoder_X_train = X_train
        y_train = np.expand_dims(y_train,axis=-1)
        decoder_y_train = np.concatenate([X_train,y_train],axis=1).astype('float')

        loss = 'mse' #<-- mse likely sucks
        
        dec_opt = optimizers.Nadam(lr=0.00001)
        self.decoder.compile(loss=loss, optimizer=dec_opt)

        stacked_opt = optimizers.Nadam(lr=0.000001)
        self.gan.compile(loss='binary_crossentropy', optimizer=stacked_opt)
        
        discr_opt = optimizers.Nadam(lr=0.00001)
        self.discr.compile(loss='binary_crossentropy', optimizer=discr_opt)
        self.discr.trainable = True
        
        
        reduce_lr = callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.2,
                        patience=5, mode='min')
        reduce_lr.on_train_begin()                
        reduce_lr.model = self.decoder
        
        early_stop = callbacks.EarlyStopping(
                monitor='val_loss', mode='min',
                        min_delta=0, patience=10)
        early_stop.on_train_begin()                
        early_stop.model = self.decoder

        
        info_list = []
        old_info_list = []
        #if os.path.exists(self.history_path):
        #    with open(self.history_path,"r") as f:
        #        old_info_list = yaml.load(f.read())
                
        epoch_num = 20
        batch_size = 64

        for epoch in range(epoch_num):
            d_loss_list = []
            g_loss_list = []
            s_loss_list = []
            dec_p = self.decoder_weight_path(epoch)
            dis_p = self.discr_weight_path(epoch)
            gan_p = self.gan_weight_path(epoch)
            if len(old_info_list) > 0 and os.path.exists(dec_p) and os.path.exists(dis_p) and os.path.exists(gan_p):
                info_list.append(old_info_list[epoch])
                self.decoder.load_weights(dec_p)
                self.discr.load_weights(dis_p)
                self.gan.load_weights(gan_p)
                print('skipping epoch',len(info_list))
                continue
            #pbar = tqdm(total=int(X_train.shape[0]/float(batch_size)))
            #c=0
            for bX_train,by_train in chunkify(decoder_X_train,decoder_y_train,batch_size):
                #pbar.update(c)
                #c+=1
                bdecOut = self.decoder.predict(bX_train, verbose=0)

                added_to_y = by_train+np.expand_dims(0.2*(np.random.random(by_train.shape[0])-0.5),axis=-1)
                # ^^ spice up the y to be 0+/-0.25 or 1+/-0.25
                X = np.concatenate((added_to_y,bdecOut),axis=0).astype('float')
                
                b_size = bX_train.shape[0]
                y = [1] * b_size + [0] * b_size
                y +=0.5*(np.random.random(2*b_size)-0.5)
                
                one_y = np.array([1]  *b_size)

                # train only discriminator with pos and neg.
                for _X,_y in [(X[:b_size,:],y[:b_size]),(X[b_size:,:],y[b_size:])]:
                    d_loss = self.discr.train_on_batch(_X,_y)
                    d_loss_list.append(d_loss)

                self.discr.trainable = False

                g_loss = self.decoder.train_on_batch(bX_train,by_train)
                s_loss = self.gan.train_on_batch(by_train[:,:-1],one_y)
                self.discr.trainable = True

                g_loss_list.append(g_loss)
                s_loss_list.append(s_loss)
                
                
            pred = self.decoder.predict(X_validation)
            pred = pred[:,-1].squeeze()
            val_loss = metrics.log_loss(y_validation,pred)
            info = {
                'epoch':epoch,
                'd_loss':float(np.mean(d_loss_list)),
                'g_loss':float(np.mean(g_loss_list)),
                's_loss':float(np.mean(s_loss_list)),
                'val_loss': float(val_loss),
            }
            info_list.append(info)
            print(info)
            self.decoder.save_weights(self.decoder_weight_path(epoch), True)
            self.discr.save_weights(self.discr_weight_path(epoch), True)
            self.gan.save_weights(self.gan_weight_path(epoch),True)
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

        y_pred = self.decoder.predict(X)[:,-1]
        y_pred = np.expand_dims(y_pred,axis=-1)
        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true,y_pred)

        return y_pred, logloss
