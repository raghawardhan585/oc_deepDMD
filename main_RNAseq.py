##
import RNAseq_data_preprocessing_functions as rnaf
import ocdeepdmd_simulation_examples_helper_functions as oc
import pickle
import random
import numpy as np
import os
import shutil

##
# To get the RNAseq and OD data to RAW format of X and Y data
rnaf.organize_RNAseq_OD_to_RAWDATA()

## Open the RAW datafile

with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle', 'rb') as handle:
    dict_DATA = pickle.load(handle)
dict_DATA_filt0 = rnaf.filter_gene_by_coefficient_of_variation(dict_DATA)
dict_DATA_filt1 = rnaf.filter_gene_by_coefficient_of_variation(dict_DATA_filt0, MEAN_TPM_THRESHOLD = 100)
dict_DATA_filt2 = rnaf.filter_gene_by_coefficient_of_variation(dict_DATA_filt1, CV_THRESHOLD = 0.25, ALL_CONDITIONS= ['MX'])


## Sorting the MAX dataset to deepDMD format
# dict_MAX = dict_DATA_filt2['MX']
dict_MAX = dict_DATA['MX']

ls_all_indices = list(dict_MAX.keys())
random.shuffle(ls_all_indices)
ls_all_train_indices = ls_all_indices
# ls_test_indices = ls_indices[0:5]

n_states = dict_MAX[ls_all_train_indices[0]]['df_X_TPM'].shape[0]
n_outputs = dict_MAX[ls_all_train_indices[0]]['Y'].shape[0]
dict_DMD1 = {'Xp' : np.empty(shape=(0,n_states)), 'Xf': np.empty(shape=(0,n_states)),'Yp' : np.empty(shape=(0,n_outputs)), 'Yf' : np.empty(shape=(0,n_outputs))}
for i in ls_all_train_indices:
    dict_DMD1['Xp'] = np.concatenate([dict_DMD1['Xp'], np.array(dict_MAX[i]['df_X_TPM'].iloc[:,0:-1]).T],axis=0)
    dict_DMD1['Xf'] = np.concatenate([dict_DMD1['Xf'], np.array(dict_MAX[i]['df_X_TPM'].iloc[:, 1:]).T], axis=0)
    dict_DMD1['Yp'] = np.concatenate([dict_DMD1['Yp'], np.array(dict_MAX[i]['Y'].iloc[:, 0:-1]).T], axis=0)
    dict_DMD1['Yf'] = np.concatenate([dict_DMD1['Yf'], np.array(dict_MAX[i]['Y'].iloc[:, 1:]).T], axis=0)

SYSTEM_NO = 101
storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
if os.path.exists(storage_folder):
    get_input = input('Do you wanna delete the existing system[y/n]? ')
    if get_input == 'y':
        shutil.rmtree(storage_folder)
        os.mkdir(storage_folder)
    else:
        quit(0)
else:
    os.mkdir(storage_folder)

storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
_, dict_Scaler, _ = oc.scale_train_data(dict_DMD1, 'standard')
with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle', 'wb') as handle:
    pickle.dump(dict_Scaler, handle)
dict_DATA_OUT = oc.scale_data_using_existing_scaler_folder(dict_DMD1, SYSTEM_NO)
with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle', 'wb') as handle:
    pickle.dump(dict_DATA_OUT, handle)
with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_Data.pickle', 'wb') as handle:
    pickle.dump(dict_MAX, handle)
with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle', 'wb') as handle:
    pickle.dump(ls_all_indices, handle)  # Only training and validation indices are stored
# Store the data in Koopman
with open('/Users/shara/Desktop/oc_deepDMD/koopman_data/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle','wb') as handle:
    pickle.dump(dict_DATA_OUT, handle)

## DMD Stats

dict_MAX = dict_DATA['MX']

