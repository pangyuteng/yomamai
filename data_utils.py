import sys,os
import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics
from keras.utils import np_utils

from cfg import downpath

data_folders = sorted([os.path.join(downpath,x) for x in os.listdir(downpath) \
        if x.startswith('numerai_dataset') and not x.endswith('.zip')])

data_files = [{
    'trainpath':os.path.join(x,'numerai_training_data.csv'),
    'testpath':os.path.join(x,'numerai_tournament_data.csv'),
} for x in data_folders]

def to_int_func(x):
    try:
        return int(x[0].lstrip('era'))
    except:
        return None

def get_data(data_file_path):
    df = pd.read_csv(data_file_path)
    
    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(df) if "feature" in f]
    X = df[features]
    Y = df["target"]
    ids = df["id"]
    eras = df["era"]        
    datatypes = df["data_type"]

    return X, Y, ids, eras, datatypes

    
def get_data_era_balanced(data_file_path,random_state=1,test_size = 0.1):
    
    X,Y,ids,eras,datatypes = get_data(data_file_path)
    print(type(X),type(Y),type(ids),type(eras))

    X_train, X_test, Y_train, Y_test = ([],[],[],[])
    for era in list(np.unique(eras)):
        if era == 'eraX':
            continue
        inds = np.where(eras==era)[0]
        sX_train, sX_test, sY_train, sY_test = cross_validation.train_test_split(
            np.take(X,inds,axis=0),np.take(Y,inds,axis=0),
            test_size=test_size,
            random_state=random_state)

        X_train.append(sX_train)
        X_test.append(sX_test)
        Y_train.append(sY_train)
        Y_test.append(sY_test)

    X_train = np.concatenate(X_train,axis=0).astype('float')
    X_test = np.concatenate(X_test,axis=0).astype('float')
    Y_train = np.concatenate(Y_train,axis=0).astype('float')
    Y_test = np.concatenate(Y_test,axis=0).astype('float')
    
    print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
    return X_train,Y_train,X_test,Y_test
    #return X_train[:1000,:],X_test[:100,:],Y_train[:1000,:],Y_test[:100,:]
    
def write_to_csv(ids,opt_pred,fname):
    # output optimized prediction of test set
    results_df = pd.DataFrame(data={'probability':opt_pred})
    joined = pd.DataFrame(ids).join(results_df)
    # Save the predictions out to a CSV file
    joined.to_csv(fname, index=False)


if __name__ == '__main__':
   TR_X_train,TR_Y_train,TTR_X_test,R_Y_test = get_data_era_balanced(data_files[-1]['trainpath'])
   TE_X_train,TE_Y_train,TE_X_test,TE_Y_test = get_data_era_balanced(data_files[-1]['testpath'])
   