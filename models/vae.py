'''
https://github.com/soumith/ganhacks
https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py
https://arxiv.org/abs/1709.00199
'''
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

from keras import backend as K
import tensorflow as tf
# https://github.com/keras-team/keras/issues/2310 
K.set_learning_phase(False)
import keras


from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dropout, Activation
from keras import layers
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from keras.layers import Input, Dense, Lambda, Layer, Add, Multiply
from keras.layers import Concatenate
from keras.layers.normalization import BatchNormalization
from keras import callbacks
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.layers import PReLU

from sklearn.decomposition import FastICA
import glob
import argparse
from time import time
from keras.callbacks import TensorBoard



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


# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py
# https://jaan.io/what-is-variational-autoencoder-vae-tutorial/
# http://louistiao.me/posts/implementing-variational-autoencoders-in-keras-beyond-the-quickstart-tutorial/

def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """
    lh = K.tf.distributions.Bernoulli(probs=y_pred)
    return - K.sum(lh.log_prob(y_true), axis=-1)

class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):

        mu, log_var = inputs

        kl_batch = - .5 * K.sum(1 + log_var -
                                K.square(mu) -
                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

def get_vae(
    original_dim = 50,
    intermediate_dim = 32,
    latent_dim = 8,
    epsilon_std = 1.0
    ):
    
    # input layer
    x = Input(shape=(original_dim,))
    
    # hidden layer
    h0 = Dense(intermediate_dim, activation='relu')(x)

    # output layer for mean and log variance
    z_mu0 = Dense(latent_dim)(h0)
    z_log_var0 = Dense(latent_dim)(h0)

    # normalize log variance to std dev
    z_sigma0 = Lambda(lambda t: K.exp(.5*t))(z_log_var0)

    z_mu0, z_log_var0 = KLDivergenceLayer()([z_mu0, z_log_var0])

    eps0 = Input(tensor=K.random_normal(stddev=epsilon_std,
                            shape=(K.shape(x)[0],latent_dim)))
    z_eps0 = Multiply()([z_sigma0, eps0])
    z0 = Add()([z_mu0, z_eps0])


    # hidden layer
    h1 = Dense(intermediate_dim, activation='relu')(x)
    z_mu1 = Dense(latent_dim)(h1)
    z_log_var1 = Dense(latent_dim)(h1)
    z_sigma1 = Lambda(lambda t: K.exp(.5*t))(z_log_var1)
    z_mu1, z_log_var1 = KLDivergenceLayer()([z_mu1, z_log_var1])
    eps1 = Input(tensor=K.random_normal(shape=(K.shape(x)[0],
                                          latent_dim)))
    z_eps1 = Multiply()([z_sigma1, eps1])
    z1 = Add()([z_mu1, z_eps1])
    sd_seq = Sequential([
        Dense(intermediate_dim, input_dim=latent_dim,
              activation='relu'),
        Dense(original_dim, activation='sigmoid')
    ])
    
    m_z = Concatenate(axis=0)([z0,z1]) #???
    x_pred = sd_seq(z0)
    SD = Model(inputs=[x, eps0], outputs=x_pred)
    SE = Model(inputs=x,outputs=z_mu0)

    zd_seq = Sequential([
        Dense(16, input_dim=latent_dim,
              activation='relu'),
        Dense(2, activation='sigmoid')
    ])
    z_pred = zd_seq(z1)
    ZC = Model(inputs=[x, eps1], outputs=z_pred)
    ZE = Model(inputs=x,outputs=z_mu1)

    m = Concatenate(axis=-1)([x_pred,z_pred])
    return SE,SD,ZE,ZC,AD,GA

def get_gan():
    I = Input(shape=(50,))
    AD = get_adv()(m)
    GA = Model(inputs=x,outputs=AD) 

    GA = Model(inputs=I,outputs=AD) 

def get_adv():
    mode = -1
    dropout_rate = 0.3
    I = Input(shape=(52,))
    d = Dense(52)(I)
    d = BatchNormalization(axis=mode)(d)
    d = LeakyReLU()(d)
    d = Dropout(dropout_rate)(d)
    d = Dense(32)(d)
    d = BatchNormalization(axis=mode)(d)
    d = LeakyReLU()(d)
    d = Dropout(dropout_rate)(d)
    d = Dense(32)(d)
    d = BatchNormalization(axis=mode)(d)
    d = LeakyReLU()(d)
    d = Dropout(dropout_rate)(d)    
    d = Dense(16)(d)
    d = BatchNormalization(axis=mode)(d)
    d = LeakyReLU()(d)
    d = Dropout(dropout_rate)(d)
    d = Dense(16)(d)
    d = BatchNormalization(axis=mode)(d)
    d = LeakyReLU()(d)
    d = Dropout(dropout_rate)(d)
    d = Dense(1)(d)
    d = Activation('sigmoid')(d)

    AD = Model(inputs=I,outputs=d)
    return AD

def chunkify(x,y, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, x.shape[0], n):
        yield x[i:i + n,:],y[i:i + n],

FDNAME = 'vae_file'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(os.path.join(THIS_DIR,FDNAME)) is False:
    os.makedirs(os.path.join(THIS_DIR,FDNAME))

class VaeModel(object):
    def __init__(self,this_dir=THIS_DIR,fdname=FDNAME):
        self.is_trained = False
        self.tensorboard = TensorBoard(
            log_dir=os.path.join(THIS_DIR,FDNAME,"log","{}".format(time())),
            histogram_freq=1,write_grads=True)

        self.this_dir = this_dir
        self.fdname = fdname

        self.se_weight_path = lambda x: os.path.join(self.this_dir,self.fdname,'se{:03d}.hdf5'.format(x)) 
        self.ze_weight_path = lambda x: os.path.join(self.this_dir,self.fdname,'ze{:03d}.hdf5'.format(x))
        self.sd_weight_path = lambda x: os.path.join(self.this_dir,self.fdname,'sd{:03d}.hdf5'.format(x))
        self.zc_weight_path = lambda x: os.path.join(self.this_dir,self.fdname,'zc{:03d}.hdf5'.format(x))
        self.ad_weight_path = lambda x: os.path.join(self.this_dir,self.fdname,'ad{:03d}.hdf5'.format(x))
        self.history_path = os.path.join(self.this_dir,self.fdname,'history.yml')

        self.SE, self.SD, self.ZE, self.ZC, self.AD, self.GA = get_vae()
        
    def _load(self):
        with open(self.history_path, "r") as f:
            history = yaml.load(f.read())
        epoch = history[np.argmin([x['val_loss'] for x in history])]['epoch']
        
        self.SE.load_weights(self.se_weight_path(epoch))
        self.ZE.load_weights(self.ze_weight_path(epoch))
        self.SD.load_weights(self.sd_weight_path(epoch))
        self.ZC.load_weights(self.zc_weight_path(epoch))
        self.AD.load_weights(self.ad_weight_path(epoch))

    def load(self):
        self._load()
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,X_validation=None,y_validation=None,X_test=None,**kwargs):
        if X_validation is None or y_validation is None:
            raise IOError()
                   
        y_train = np_utils.to_categorical(y_train)
        if y_validation is not None:
            y_validation = np_utils.to_categorical(y_validation)
            
        '''
        # at lr of 0.0001 for sd and 0.001 for zc
        # mse for sd was 0.726,0.0.0419 at epochs 0 and 4
        # logloss for zc was 0.722,0.0.693 at epochs 0 and 4        
        '''              
        
        #sd_opt = Adam(lr=0.0000001)                                                 
        self.SD.compile(optimizer='rmsprop', loss=nll)
        
        #zc_opt = SGD(lr=0.001)
        self.ZC.compile(optimizer='rmsprop', loss=nll)

        ad_opt = Adam(lr=0.0000002)
        self.AD.compile(loss='binary_crossentropy',optimizer=ad_opt)
        
        ga_opt = Adam(lr=0.0000001)
        self.GA.compile(loss='binary_crossentropy',optimizer=ga_opt)
        
        self.SE.trainable = True
        self.AD.trainable = True
        
        info_list = []
        
        old_info_list = []
        if os.path.exists(self.history_path):
            with open(self.history_path,"r") as f:
                old_info_list = yaml.load(f.read())

        epoch_num = 2000
        batch_size = 64
        
        for epoch in range(epoch_num):
            
            # random shuffle per epoch/ too noisy
            train_inds = np.random.permutation(X_train.shape[0])
            X_train = X_train[train_inds,:]
            y_train = y_train[train_inds,:]

            sd_loss_list = []
            zc_loss_list = []
            ad_loss_list = []
            ga_loss_list = []
            
            wlist=[
                self.se_weight_path(epoch),
                self.ze_weight_path(epoch),
                self.sd_weight_path(epoch),
                self.zc_weight_path(epoch),
                self.ad_weight_path(epoch),
            ]
            if all([os.path.exists(x) for x in wlist]) and len(old_info_list)>epoch:
                print(epoch)
                info_list.append(old_info_list[epoch])
                self.SE.load_weights(self.se_weight_path(epoch))
                self.ZE.load_weights(self.ze_weight_path(epoch))
                self.SD.load_weights(self.sd_weight_path(epoch))
                self.ZC.load_weights(self.zc_weight_path(epoch))
                self.AD.load_weights(self.ad_weight_path(epoch))
                print('skipping epoch',epoch)
                continue
            
            for bX_train,by_train in chunkify(X_train,y_train,batch_size):
               
                realBatch = np.concatenate([bX_train,by_train],axis=-1)
            
                predX = self.SD.predict(bX_train)
                predZ = self.ZC.predict(bX_train)
                
                generatedBatch = np.concatenate([predX,predZ],axis=-1)
                
                X = np.concatenate((realBatch,generatedBatch),axis=0).astype('float')
                
                b_size = realBatch.shape[0]
                y = [1] * b_size + [0] * b_size
                y +=0.5*(np.random.random(2*b_size)-0.5)
                one_y = np.array([1]  *b_size)

                # train only discriminator with pos and neg.
                for _X,_y in [(X[:b_size,:],y[:b_size]),(X[b_size:,:],y[b_size:])]:
                    ad_loss = self.AD.train_on_batch(_X,_y)
                    ad_loss_list.append(ad_loss)
                
                sd_loss = self.SD.train_on_batch(bX_train,bX_train)
                sd_loss_list.append(sd_loss)
                
                self.AD.trainable=False
                self.SE.trainable = False
                
                for _ in range(3):
                    zc_loss = self.ZC.train_on_batch(bX_train,by_train)
                    zc_loss_list.append(zc_loss)
                    
                self.AD.trainable=False
                self.SE.trainable = True
                ga_loss = self.GA.train_on_batch(predX,one_y)
                ga_loss_list.append(ga_loss)
                
                self.AD.trainable=True
            
            #zc_loss = self.ZC.fit(
            #    X_validation,y_validation,
            #    verbose=0, callbacks=[self.tensorboard],
            #    validation_data=(X_validation,y_validation))
            
            pred = self.ZC.predict(X_validation).squeeze()
            val_loss = metrics.log_loss(y_validation,pred)
            info = {
                'epoch':epoch,
                'sd_loss':float(np.mean(sd_loss_list)),
                'zc_loss':float(np.mean(zc_loss_list)),
                'ad_loss':float(np.mean(ad_loss_list)),
                'ga_loss':float(np.mean(ga_loss_list)),
                'val_loss': float(val_loss),
            }
            info_list.append(info)
            print(info)
            self.SE.save_weights(self.se_weight_path(epoch),True)
            self.ZE.save_weights(self.ze_weight_path(epoch),True)
            self.SD.save_weights(self.sd_weight_path(epoch),True)
            self.ZC.save_weights(self.zc_weight_path(epoch),True)
            self.AD.save_weights(self.ad_weight_path(epoch),True)
            with open(self.history_path, "w") as f:
                f.write(yaml.dump(info_list))
        
        self.load()

    def predict(self,X,y_true=None,icaxlist=None):
        if self.is_trained is False:
            self.load()

        y_pred = self.ZC.predict(X)[:,-1]

        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true,y_pred)

        return y_pred, logloss
