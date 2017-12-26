from hpsklearn import HyperoptEstimator, any_classifier
from sklearn.datasets import fetch_mldata
from hyperopt import tpe
import numpy as np
from sklearn import metrics

from hpsklearn import HyperoptEstimator, svc
from sklearn import svm

import xgboost as xgb

from . import aec_gan

import sys,os
FDNAME = 'aec_gan_xg_file'
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class AecGanXgModel(object):
    def __init__(self,this_dir=THIS_DIR,fdname=FDNAME):
        self.this_dir = this_dir
        self.fdname = fdname
        self.params = {'max_depth': 6, 'colsample_bytree': 0.5, 'subsample': 0.9500000000000001, 'objective': 'multi:softprob', 'eta': 0.1, 'nthread': 2, 'eval_metric': 'mlogloss', 'silent': 1, 'num_class': 2, 'n_estimators': 171.0, 'gamma': 0.55, 'min_child_weight': 3.0}
        self.params.update(dict(seed=69))
        self.num_round =5
        self.is_trained = False
        self.model = xgb.Booster(self.params,)#{'nthread':6,}) #
        self.xgb_model_fname = os.path.join(self.this_dir,self.fdname,'xg.model')

    def _load_aec(self):
        self.aa_inst = aec_gan.AecAdvModel(this_dir=self.this_dir,fdname=self.fdname)
        self.aa_inst._load_aec()

    def load(self):
        self._load_aec()
        self.model.load_model(self.xgb_model_fname)
        self.is_trained = True

    def fit(self,X_train=None,y_train=None,**kwargs):
        if X_train is None:
            raise IOError()
        self._load_aec()
        eX = self.aa_inst.encoder.predict(X_train)
        print(eX.shape,'!!!')
        dtrain = xgb.DMatrix(eX, label=y_train)
        self.model = xgb.train(self.params, dtrain, self.num_round)
        self.model.save_model(self.xgb_model_fname)
        self.load()

    def predict(self,X,y_true=None):
        if self.is_trained is False:
            self.load()
        eX = self.aa_inst.encoder.predict(X)
        dX = xgb.DMatrix(eX)
        y_pred = self.model.predict(dX)[:,1]
        y_pred = np.expand_dims(y_pred,axis=1)
        logloss = None
        if y_true is not None:
            logloss = metrics.log_loss(y_true, y_pred)
            print('logloss %r' % logloss)

        return y_pred, logloss

