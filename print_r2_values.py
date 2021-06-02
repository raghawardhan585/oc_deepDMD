import pickle
import pandas as pd
import os
import numpy as np
pd.reset_option('display.float_format')

file_path = '_current_run_saved_files'
dict_run = {}
for folder in os.listdir(file_path):
    try:
        with open(file_path + '/' + folder + '/' + 'dict_hyperparameters.pickle','rb') as handle:
            dict_hp = pickle.load(handle)
        try:
            dict_run[folder[-2:]] = {'run_no': int(folder[-2:]), 'r2_train': dict_hp['r2 train'],
                                     'r2_valid': dict_hp['r2 valid'],
                                     'difference': dict_hp['r2 train'] - dict_hp['r2 valid'],
                                     'lambda': np.float(dict_hp['regularization factor'])}
        except:
            dict_run[folder[-1]] = {'run_no': int(folder[-1]), 'r2_train': dict_hp['r2 train'],
                                     'r2_valid': dict_hp['r2 valid'],
                                     'difference': dict_hp['r2 train'] - dict_hp['r2 valid'],
                                     'lambda': np.float(dict_hp['regularization factor'])}
    except:
        print('Folder name: ', folder, ' is not a run')
print('=====================================================================')
print('Error Stats')
print('=====================================================================')
df_result = pd.DataFrame(dict_run).T.loc[:,['run_no', 'lambda', 'r2_train', 'r2_valid', 'difference']].sort_values(by=['run_no'])
# df_result.loc[:,'lambda'] ="{:e}".format(df_result.loc[:,'lambda'])
df_result.loc[:,'lambda'] = df_result.loc[:,'lambda'] * 1e6
df_result = df_result.rename(columns={'lambda':'lambda[x1e-6]'})
print(df_result)
print('=====================================================================')