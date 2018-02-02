import traceback
import sys,os
import socket

if socket.gethostname() == 'gtx':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
np.random.seed(69)

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import pandas as pd

from cfg import cfg
from numerapi.numerapi import NumerAPI
import models
from data_utils import get_data_era_balanced,data_files,get_data, write_to_csv
from models import opt

model_list = [
#    ('aec',models.aec.AecModel,dict(istrain=True)),
#    ('aecgan',models.aec_gan.AecAdvModel,dict(istrain=True)),
    #('xg',models.xg.XgModel,dict(istrain=False)),
    #('aecganxg',models.aec_gan_xg.AecGanXgModel,dict(istrain=True)),# depends on model from AecAdvModel
    #worse than random# ('aecgs',models.aec_gan_stack.AecAdvStackModel,dict(istrain=True)),
    #inprogress#just bad# still testing#('krauss',models.krauss.KraussModel,dict(istrain=True)),
    #('ganmore',models.ganmore.GanMoreModel,dict(istrain=True)),
    #('sktpot',models.sk_tpot.SkTPot,dict(istrain=True)),
#    ('disentangle',models.disentangle.DisentangleModel,dict(istrain=True)),
    #('disentangle_kfold',models.disentangle_kfold.DisentangleKfoldModel,dict(istrain=False)),
    #('disentanglegan',models.disentanglegan.DisentangleGanModel,dict(istrain=True)),
    #('mdg',models.moddisengan.DisentangleModel,dict(istrain=True)),
    #('mdg2',models.moddisengan2.DisentangleModel,dict(istrain=True)),
    #('mdg3',models.moddisengan3.DisentangleModel,dict(istrain=True)),#no disentangle
    #('vae',models.vae.VaeModel,dict(istrain=True)),
    #('tsnesimple',models.tsne_simple.TsneSimple,dict(istrain=False)),
    #('tsnesimplekfold',models.tsne_simplekfold.TsneSimpleKfold,dict(istrain=False)),
    #('simple',models.simple.Simple,dict(istrain=False)),
    #('simplekfold',models.simplekfold.SimpleKfold,dict(istrain=False)),
    #('res',models.res.Res,dict(istrain=True)),
    #('highway',models.highway.HighwayModel,dict(istrain=True)),
    #('highway2',models.highway2.HighwayModel,dict(istrain=True)),
    ('highway2k',models.highway_kfold.HighwayModel,dict(istrain=True)),
    
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
    tr_eras_dict = {}
    X_train_tr,y_train_tr,X_val_tr,y_val_tr = get_data_era_balanced(data_files[-1]['trainpath'],eras_dict=tr_eras_dict)
    te_eras_dict = {}
    X_train_te,y_train_te,X_val_te,y_val_te = get_data_era_balanced(data_files[-1]['testpath'],eras_dict=te_eras_dict)
    X_train = np.concatenate([X_train_tr,X_train_te],axis=0)
    y_train = np.concatenate([y_train_tr,y_train_te],axis=0)
    eras_train = np.concatenate([tr_eras_dict['train'],te_eras_dict['train']],axis=0)
    X_val = np.concatenate([X_val_tr,X_val_te],axis=0)
    y_val = np.concatenate([y_val_tr,y_val_te],axis=0)
    eras_val = np.concatenate([tr_eras_dict['validation'],te_eras_dict['validation']],axis=0)
    
    X_test,y_test,_,_,_=get_data(data_files[-1]['testpath'])

    # for testing
    if isinstance(shrink_train,int):
        X_train=X_train[:-1:shrink_train,:]
        y_train=y_train[:-1:shrink_train]
        X_val=X_val[:-1:shrink_train,:]
        y_val=y_val[:-1:shrink_train]

    # determine sample weights
    #ddiscr=models.datatype_discr.DDiscrModel()
    #ddiscr.fit(X_train=X_train,y_train=y_train,X_validation=X_val,y_validation=y_val,X_test=X_test)
    #ddiscr.load()
    #train_sample_weight, _ = ddiscr.predict(X_train)
    train_sample_weight = None
    
    # fit models
    for name,clsf,params in model_list:

        if params['istrain'] is False:
            continue

        inst = clsf()
        ##if hasattr(inst,'pretrain'):
        ##    inst.pretrain(**pretrain_params)
        inst.fit(X_train,y_train,X_validation=X_val,y_validation=y_val,
                 X_test=X_test,
                 eras_train=eras_train,
                 eras_validation=eras_val,
                 sample_weight=train_sample_weight)

    # predict train
    y_pred_list = []
    for name,clsf,params in model_list:
        inst = clsf()
        # optimize with validation set
        y_pred,logloss = inst.predict(X_val,y_true=y_val)
        print('logloss',name,logloss)
        y_pred_list.append(y_pred.squeeze())
        print(y_pred.shape)

    # optimize with train set
    opt_weights = opt.opt_weights(y_pred_list,y_val)
    np.save('opt_weights.pkl',opt_weights)
    
    for n,c in enumerate(model_list):
        print('weight {}:{:1.4f}'.format(c[0],opt_weights[n]))
    
    # prep for testing
    del X_train,y_train,X_val,y_val  # save some memory
    X_test,y_test,ids,_eras,_datatypes=get_data(data_files[-1]['testpath'])
    print(data_files[-1])

    # for testing
    if isinstance(shrink_test,int):
        X_test = X_test[:-1:shrink_test,:]
        y_test = y_test[:-1:shrink_test]
        ids = ids[:-1:shrink_test]
        _eras = _eras[:-1:shrink_test]

    # predict test
    val_inds = np.array(np.where(_datatypes=='validation')).squeeze()
    y_pred_list = []
    for name,clsf,params in model_list:
        inst = clsf()
        y_pred,_ = inst.predict(X_test.values)
        y_pred_list.append(y_pred.squeeze())
        print('logloss',name,opt.log_loss_func([1.],[y_pred[val_inds]],y_test[val_inds]))

    # optimize prediction
    opt_pred = opt.opt_pred(y_pred_list,opt_weights)
    print('final logloss',opt.log_loss_func([1.],[opt_pred[val_inds]],y_test[val_inds]))
    
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
