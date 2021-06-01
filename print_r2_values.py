import pickle
import pandas as pd
import os

file_path = '_current_run_saved_files'
dict_run = {}
for folder in os.listdir(file_path):
    try:
        with open(file_path + '/' + folder + '/' + 'dict_hyperparameters.pickle','rb') as handle:
            dict_hp = pickle.load(handle)
        dict_run[folder[-1]] = {'run_no': folder[-1], 'r2_train': dict_hp['r2 train'], 'r2_valid': dict_hp['r2 valid'], 'difference': dict_hp['r2 train'] - dict_hp['r2 valid'], 'lambda':dict_hp['regularization factor']}
    except:
        print('Folder name: ', folder, ' is not a run')
print('=====================================================================')
print('Error Stats')
print('=====================================================================')
print(pd.DataFrame(dict_run).T.loc[:,['run_no', 'lambda', 'r2_train', 'r2_valid', 'difference']].sort_values(by=['run_no']))
print('=====================================================================')