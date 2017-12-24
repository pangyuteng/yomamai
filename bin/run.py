import traceback
import sys,os
import socket

if socket.gethostname() == 'gtx':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
np.random.seed(69)

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import pandas as pd

from cfg import cfg
from numerapi.numerapi import NumerAPI
import models
from data_utils import get_data_era_balanced,data_files,get_data, write_to_csv
import opt

model_list = [
    ('aecgan',models.aec_gan.AecAdvModel,dict(istrain=False)),
    ('aec',models.aec.AecModel,dict(istrain=False)),
    ('xg',models.xg.XgModel,dict(istrain=False)),
    ('aecganxg',models.aec_gan_xg.AecGanXgModel,dict(istrain=False)),# depends on model from AecAdvModel
]


def main():

    # api instance
    api_inst=NumerAPI(**cfg['api'])
    try:
        sub_status = api_inst.submission_status()
    except:
        traceback.print_exc()
        sub_status = None
    
    # download if necessary
    api_inst.download_current_dataset(dest_path=cfg['downpath'],unzip=True)
    print(sub_status)
    print(data_files[-1])

    shrink_train = None
    shrink_test = None
    print('====================')
    # get data 
    X_train,y_train,X_val,y_val = get_data_era_balanced(data_files[-1]['trainpath'])
    X_test,y_test,_,_=get_data(data_files[-1]['testpath'])

    # for testing
    if isinstance(shrink_train,int):
        X_train=X_train[:-1:shrink_train,:]
        y_train=y_train[:-1:shrink_train]
        X_val=X_val[:-1:shrink_train,:]
        y_val=y_val[:-1:shrink_train]

    ##pretrain_params = dict()

    # fit models
    for name,clsf,params in model_list:

        if params['istrain'] is False:
            continue

        inst = clsf()
        ##if hasattr(inst,'pretrain'):
        ##    inst.pretrain(**pretrain_params)
        inst.fit(X_train,y_train,X_validation=X_val,y_validation=y_val,X_test=X_test)

    # predict train
    y_pred_list = []
    for name,clsf,params in model_list:
        inst = clsf()
        y_pred,logloss = inst.predict(X_train,y_true=y_train)
        print('logloss',name,logloss)
        y_pred_list.append(y_pred)
        print(y_pred.shape)

    # optimize with train set
    opt_weights = opt.opt_weights(y_pred_list,y_train)
    np.save('opt_weights.pkl',opt_weights)
    
    # prep for testing
    del X_train,y_train,X_val,y_val    
    X_test,y_test,ids,_eras=get_data(data_files[-1]['testpath'])
    print(data_files[-1])

    # for testing
    if isinstance(shrink_test,int):
        X_test = X_test[:-1:shrink_test,:]
        y_test = y_test[:-1:shrink_test]
        ids = ids[:-1:shrink_test]
        _eras = _eras[:-1:shrink_test]

    # predict test
    y_pred_list = []
    for name,clsf,params in model_list:
        inst = clsf()
        y_pred,_ = inst.predict(X_test)
        print(name,)
        y_pred_list.append(y_pred)

    # optimize prediction
    opt_pred = opt.opt_pred(y_pred_list,opt_weights)
    print(opt_pred.shape)

    # write prediction
    write_to_csv(ids,opt_pred,"predictions.csv")

    # upload
    api_inst.upload_predictions("predictions.csv")
    try:
        sub_status = api_inst.submission_status()
    except:
        traceback.print_exc()
        sub_status = None


if __name__ == '__main__':
    main()