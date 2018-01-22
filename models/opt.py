import traceback
import pandas as pd
from scipy.optimize import minimize
from sklearn import metrics
import numpy as np


def opt_pred(pred_list,weights):
    print('opt_pred')
    
    pred_arr = np.array(pred_list).squeeze()
    if len(pred_arr.shape) ==1:
        return pred_arr

    print(pred_arr.shape,weights.shape)

    # calculate pred
    y_pred = np.dot(weights,pred_arr)
    return y_pred

def variance(x,*args):
    
    p = np.squeeze(np.asarray(args))
    Acov = np.cov(p.T)
    
    return np.dot(x,np.dot(Acov,x))

def jac_variance(x,*args):
    
    p = np.squeeze(np.asarray(args))
    Acov = np.cov(p.T)
        
    return 2*np.dot(Acov,x)

def opt_func(weights,pred_list,y_test):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, pred in zip(weights, pred_list):
        final_prediction += weight*pred
    
    logloss = metrics.log_loss(y_test, final_prediction)
    logloss = np.nan_to_num(logloss)
    mystd = np.nanstd(y_test.squeeze()-final_prediction.squeeze())
    
    return logloss/mystd



def log_loss_func(weights,pred_list,y_test):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, pred in zip(weights, pred_list):
        final_prediction += weight*pred
    
    logloss = metrics.log_loss(y_test, final_prediction)
    logloss = np.nan_to_num(logloss)
    
    return logloss


# https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
# https://www.quantopian.com/posts/minimum-variance-w-slash-constraint
def opt_weights(pred_list,y_test):

    #the algorithms need a starting value, right not we chose 0.5 for all weights
    #its better to choose many random starting points and run minimize a few times
    #starting_values = [1.0/len(pred_list)]*len(pred_list)
    pred_list = np.array(pred_list)
    y_test = np.array(y_test)
    ret = pred_list - y_test
    ret = ret.T
    candidates = []
    for n in range(10):
        starting_values = np.random.rand(ret.shape[1])
        bnds = tuple([(0.0,1.0)]*ret.shape[1])
        ret_mean = ret.mean(axis=0)
        ret_std = ret.std(axis=0)
        ret_norm = ret_mean/ret_std
        cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x)-1.0}, # weight sums to 1
                #{'type': 'eq', 'fun': lambda x:  np.dot(x,ret_mean)-0.5}, # minimize the error
               )        
    
        try:
            # minimize variance
            res = minimize(variance, starting_values, args=ret,jac=jac_variance, 
                                     method='SLSQP',constraints=cons,bounds=bnds)
        except:
            traceback.print_exc()
            continue
        candidates.append({'score':res['fun'],'weights':res['x']})

        print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
        print('Best Weights: {weights}'.format(weights=res['x']))

    candidates = sorted(candidates,key=lambda x: x['score'],reverse=False)
    print('-----------------------------------')
    print('candidate weight info',candidates[0])
    print('-----------------------------------')
    # return weight with minimum log loss.
    return candidates[0]['weights']
    