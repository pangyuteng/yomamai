from hpsklearn import HyperoptEstimator, any_classifier
from sklearn.datasets import fetch_mldata
from hyperopt import tpe
import numpy as np
from sklearn import metrics

from hpsklearn import HyperoptEstimator, svc
from sklearn import svm
from tpot import TPOTClassifier

import xgboost as xgb

import sys,os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class SkTPot(object):
    def __init__(self):
        self.is_trained = False
        #self.params.update(dict(seed=69))
        #self.num_round =5
        #self.model = xgb.Booster(self.params,)#{'nthread':6,}) #
        #self.xgb_model_fname = os.path.join(THIS_DIR,'xg_file','xg.model')
        
    def load(self):
        #self.model.load_model(self.xgb_model_fname)
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,X_validation=None,y_validation=None,**kwargs):
        if X_train is None:
            raise IOError()
        tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
        tpot.fit(X_train[:1000,:], y_train[:1000])
        print(tpot.score(X_validation, y_validation))
        tpot.export(os.path.join(THIS_DIR,'tpot_pipeline.py'))
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

