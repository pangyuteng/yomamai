import traceback
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import numpy as np


def opt_pred(pred_list,weights):
    print('opt_pred')
    pred_arr = np.array(pred_list).squeeze()
    print(pred_arr.shape,weights.shape)
    pred = pred_arr.T*weights

    if len(pred.shape) ==1:
        return pred
        
    pred = np.sum(pred,axis=1)
    return pred

def log_loss_func(weights,pred_list,y_test):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = 0
    for weight, pred in zip(weights, pred_list):
        final_prediction += weight*pred
    
    lolo = log_loss(y_test, final_prediction)

    return np.nan_to_num(lolo)

def opt_weights(pred_list,y_test):

    #the algorithms need a starting value, right not we chose 0.5 for all weights
    #its better to choose many random starting points and run minimize a few times
    #starting_values = [1.0/len(pred_list)]*len(pred_list)

    candidates = []
    for n in range(10):
        starting_values = np.random.rand(len(pred_list))

        #adding constraints  and a different solver as suggested by user 16universe
        #https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        #our weights are bound between 0 and 1
        bounds = [(0.05,0.6)]*len(pred_list)
        try:
            res = minimize(log_loss_func, starting_values, args=(pred_list,y_test),
                method='SLSQP',bounds=bounds, constraints=cons)
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
    