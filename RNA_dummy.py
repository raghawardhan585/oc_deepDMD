##
import RNAseq_data_preprocessing_functions as rnaf
import ocdeepdmd_simulation_examples_helper_functions as oc
import pickle
import random
import numpy as np
import pandas as pd
import os
import shutil
import random
import matplotlib.pyplot as plt
import copy
import itertools
import seaborn as sb

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
from sklearn.metrics import make_scorer,r2_score

SYSTEM_NO = 104
NO_OF_FOLDS = 12
MAX_ITERS = 100000
data_directory = 'koopman_data/'
data_suffix = 'System_'+str(SYSTEM_NO)+'_ocDeepDMDdata.pickle'
data_file = data_directory + data_suffix

with open(data_file, 'rb') as handle:
    p = pickle.load(handle)

q = copy.deepcopy(p)
for items in q:
    q[items] = q[items][0:72]

kf = KFold(n_splits=NO_OF_FOLDS, shuffle=False, random_state=None)

dict_results = {}
for alpha in np.arange(0.,0.11,0.01):
    iter = -1
    dict_results[alpha] = {}
    for train_index,valid_index in kf.split(p['Xp']):
        iter = iter + 1
        Xp_train = p['Xp'][train_index]
        Xf_train = p['Xf'][train_index]
        Xp_valid = p['Xp'][valid_index]
        Xf_valid = p['Xf'][valid_index]
        if alpha ==0:
            model_1 = LinearRegression(fit_intercept=False)
        else:
            model_1 = Lasso(alpha = alpha, fit_intercept=False, max_iter= MAX_ITERS)
        model_1.fit(Xp_train,Xf_train)
        Xfhat_train = model_1.predict(Xp_train)
        Xfhat_valid = model_1.predict(Xp_valid)
        dict_results[alpha][iter] = r2_score(np.concatenate([Xf_train,Xf_valid],axis=0), np.concatenate([Xfhat_train,Xfhat_valid],axis=0))
        # print(r2_score(Xf_train,Xfhat_train))
        # print(r2_score(Xf_valid, Xfhat_valid))
        # print(r2_score(np.concatenate([Xf_train,Xf_valid],axis=0), np.concatenate([Xfhat_train,Xfhat_valid],axis=0) ))
    print('Alpha = ',alpha, ' r2 = ', [dict_results[alpha][i] for i in dict_results[alpha].keys()])
