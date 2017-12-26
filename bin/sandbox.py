
import sys,os
import socket 

if socket.gethostname() == 'gtx':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from cfg import cfg
from numerapi.numerapi import NumerAPI
import models

from data_utils import get_data_era_balanced,data_files


import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn import preprocessing

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll.base import scope
from hyperopt.pyll.stochastic import sample

import xgboost as xgb

def optimize(trials):
    space = {
             'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
             'eta' : hp.quniform('eta', 0.025, 0.5, 0.025),
             'max_depth' : sample(scope.int(hp.quniform('max_depth', 1, 13, 1))),
             'min_child_weight' : hp.quniform('min_child_weight', 1, 6, 1),
             'subsample' : hp.quniform('subsample', 0.5, 1, 0.05),
             'gamma' : hp.quniform('gamma', 0.5, 1, 0.05),
             'colsample_bytree' : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
             'num_class' : 2,
             'eval_metric': 'mlogloss',
             'objective': 'multi:softprob',
             'nthread' : 2,
             'silent' : 1
             }
    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=25)

    print(best)
    
def score(params):
    print ("Training with params : ")
    print(params)
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    model = xgb.train(params, dtrain, num_round)
    predictions = model.predict(dvalid)[:,1]
    score = log_loss(y_test, predictions)
    print("\tScore {0}\n\n".format(score))
    return {'loss': score, 'status': STATUS_OK}

if __name__ == '__main__':
    
    _X_train,y_train,_X_test,y_test = get_data_era_balanced(data_files[-1]['trainpath'])
    adv=models.aec_gan.AecAdvModel()
    adv.load()
    X_train = adv.decoder.predict(_X_train)
    X_test = adv.decoder.predict(_X_test)
    trials = Trials()

    optimize(trials)