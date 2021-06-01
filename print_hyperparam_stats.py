import pickle
import pandas as pd
import os

file_path = '_current_run_saved_files'
# file_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_400/Sequential'
dict_run = {}
for folder in os.listdir(file_path):
    try:
        with open(file_path + '/' + folder + '/' + 'dict_hyperparameters.pickle','rb') as handle:
            dict_hp = pickle.load(handle)
        try:
            run_no = int(folder[-2:])
        except:
            run_no = int(folder[-1])
        dict_run[run_no] = {'run_no': run_no, 'r2_train': dict_hp['r2 train'],
                                 'r2_valid': dict_hp['r2 valid'],
                                 'lambda': dict_hp['regularization factor'], 'x_obs': dict_hp['x_obs'], 'n_l & n_n': [dict_hp['x_layers'],dict_hp['x_nodes']] }
        # dict_run[run_no] = {'run_no': run_no, 'lambda': dict_hp['regularization factor'], 'x_obs': dict_hp['x_obs'], 'n_l & n_n': [dict_hp['x_layers'],dict_hp['x_nodes']] }
    except:
        print('Folder name: ', folder, ' is not a run')
print('=====================================================================')
print('Error Stats')
print('=====================================================================')
print(pd.DataFrame(dict_run).T.loc[:,['run_no', 'x_obs', 'r2_train', 'r2_valid', 'lambda', 'n_l & n_n']].sort_values(by=['x_obs']))
# print(pd.DataFrame(dict_run).T.loc[:,['run_no', 'x_obs', 'lambda', 'n_l & n_n']].sort_values(by=['run_no']))
print('=====================================================================')