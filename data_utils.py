import os,sys
import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics
from keras.utils import np_utils

data_id = 'numerai_dataset_20171216'#numerai_dataset_20171216
root = r'/home/yoyoteng/scisoft/data/numerai'
train_file_path = os.path.join(root,data_id,'numerai_training_data.csv')
test_file_path = os.path.join(root,data_id,'numerai_tournament_data.csv')


def to_int_func(x):
    return int(x[0].lstrip('era'))

def get_data(data_file_path,skip_y=False):
    df = pd.read_csv(data_file_path)
    
    X = np.array(df.drop(['id','era','data_type','target'], 1)).astype('float')

    if skip_y is True:
        return X,None,None
    
    _Y = np.array(df['target']).astype('float')
    Y = np_utils.to_categorical(_Y)

    era = np.expand_dims(np.array(df['era']),1)
    e = np.apply_along_axis(to_int_func, 1,era)
    
    return X, Y, e

    
def get_data_era_balanced(data_file_path,random_state=1,test_size = 0.1):
    
    X_train, X_test, Y_train, Y_test = ([],[],[],[])
    
    X,Y,e = get_data(data_file_path)

    for era in list(np.unique(e)):
        print(era,X.shape,Y.shape,e.shape)
        inds = np.where(e==era)[0]
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
    
    print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
    return X_train,X_test,Y_train,Y_test
    #return X_train[:1000,:],X_test[:100,:],Y_train[:1000,:],Y_test[:100,:]
    
if __name__ == '__main__':
   get_data_era_balanced(train_file_path)
