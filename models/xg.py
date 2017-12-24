from hpsklearn import HyperoptEstimator, any_classifier
from sklearn.datasets import fetch_mldata
from hyperopt import tpe
import numpy as np
from sklearn import metrics

from hpsklearn import HyperoptEstimator, svc
from sklearn import svm

import xgboost as xgb

import sys,os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class XgModel(object):
    def __init__(self,):
        self.params = dict(

             n_estimators=182,
             subsample=0.0,
             gamma=0.75,
             min_child_weight=5,
             eta=0.025,             
             colsample_bytree=0.75,
             num_class=2,
             objective='multi:softprob',
             max_depth=6,
             scale_pos_weight=1,
             seed=27,
             nthread=6,
         )
        self.num_round =5
        self.is_trained = False
        self.model = xgb.Booster(self.params,)#{'nthread':6,}) #
        self.xgb_model_fname = os.path.join(THIS_DIR,'xg_file','xg.model')

    def load(self):
        self.model.load_model(self.xgb_model_fname)
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,**kwargs):
        if X_train is None:
            raise IOError()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(self.params, dtrain, self.num_round)
        self.model.save_model(self.xgb_model_fname)
        self.load()

    def predict(self,X,y_true=None):
        if self.is_trained is False:
            self.load()
        dX = xgb.DMatrix(X)
        y_pred = self.model.predict(dX)[:,1]
        y_pred = np.expand_dims(y_pred,axis=1)
        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true, y_pred)
            print('logloss %r' % logloss)

        return y_pred, logloss

