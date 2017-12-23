import sys,os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from cfg import cfg
from numerapi.numerapi import NumerAPI

import data_utils

def main():

	api_inst=NumerAPI(**cfg['api'])
	# download if necessary
	#X_train,y_train,e_train,X_test,y_test,e_test

	# fit models

	# predict train and test set 

	# optimize with train set

	# putput optimized prediction of test set
	

if __name__ == '__main__':
	main()