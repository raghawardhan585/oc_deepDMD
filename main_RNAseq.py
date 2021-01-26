##
import RNAseq_data_preprocessing_functions as rnaf
import ocdeepdmd_simulation_examples_helper_functions as oc
import pickle
import random
import numpy as np
import os
import shutil
import random
import matplotlib.pyplot as plt
import copy

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


##
n_genes = dict_MAX[0]['df_X_TPM'].shape[0]
for l in range(20):
    igene = random.randint(0,n_genes)
    plt.figure()
    for i in range(16):
        plt.plot(dict_MAX[i]['df_X_TPM'].iloc[igene,:])
    plt.title('Gene ' + str(igene))
    plt.show()

## DMD Stats

data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_101/System_101_ocDeepDMDdata.pickle'
with open(data_path,'rb') as handle:
    p = pickle.load(handle)

TRAIN_PERCENT = 70
n_tot = p['Xp'].shape[0]
n_train = np.int(np.floor(n_tot*TRAIN_PERCENT/100))
n_valid = n_tot - n_train
Xp_train = p['Xp'][0:n_train].T
Xf_train = p['Xf'][0:n_train].T
Xp_valid = p['Xp'][n_train:].T
Xf_valid = p['Xf'][n_train:].T


## Optimal Number of Principal Components
U,S,Vh = np.linalg.svd(Xp_train)
# plt.stem((1-np.cumsum(S**2)/np.sum(S**2))*100)
# plt.show()
V = Vh.T.conj()
Uh = U.T.conj()
A_hat = np.zeros(shape=U.shape)
ls_error_train = []
ls_error_valid = []
for i in range(len(S)):
    A_hat = A_hat + (1 / S[i]) * np.matmul(np.matmul(Xf_train, V[:, i:i + 1]), Uh[i:i + 1, :])
    ls_error_train.append(np.mean(np.square((Xf_train - np.matmul(A_hat, Xp_train)))))
    if Xp_valid.shape[1] != 0:
        ls_error_valid.append(np.mean(np.square((Xf_valid - np.matmul(A_hat, Xp_valid)))))
if Xp_valid.shape[1] == 0:
    ls_error = np.array(ls_error_train)
else:
    ls_error = np.array(ls_error_train) + np.array(ls_error_valid)
# nPC_opt = np.where(ls_error == np.min(ls_error))[0][0] + 1
nPC_opt = np.where(ls_error_valid == np.min(ls_error_valid))[0][0] + 1
print('Optimal Principal components : ', nPC_opt)
A_hat_opt = np.zeros(shape=U.shape)
for i in range(nPC_opt):
    A_hat_opt = A_hat_opt + (1 / S[i]) * np.matmul(np.matmul(Xf_train, V[:, i:i + 1]), Uh[i:i + 1, :])

##
# plt.plot(ls_error_train[55:])
# plt.plot(ls_error_valid[55:])
plt.plot(np.array(ls_error_train[55:]) + np.array(ls_error_valid[55:]))
plt.show()

## Performance

indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_101/System_101_OrderedIndices.pickle'
with open(indices_path,'rb') as handle:
    ls_indices = pickle.load(handle)
raw_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_101/System_101_Data.pickle'
with open(raw_data_path,'rb') as handle:
    dict_RAWDATA = pickle.load(handle)

# dict_predictions = {}
dict_r2 = {}
for curve in ls_indices:
    dict_r2[curve] ={}
    # Getting the scaled data
    X = oc.scale_data_using_existing_scaler_folder({'X' : np.array(dict_RAWDATA[curve]['df_X_TPM']).T}, SYSTEM_NUMBER=101)['X'].T
    # 1 - step
    Xhat = copy.deepcopy(X[:, 0:1])
    for i in range(len(X[0]) - 1):
        Xhat = np.concatenate([Xhat, np.matmul(A_hat_opt, X[:, i:(i+1)])], axis=1)
    SSE = np.sum((Xhat[:,1:] - X[:,1:]) ** 2)
    SST = np.sum(X[:,1:] ** 2)
    dict_r2[curve]['1-step'] = np.max([0, 100 * (1 - SSE / SST)])
    # n - step
    Xhatn = copy.deepcopy(X[:,0:1])
    for i in range(len(X[0])-1):
        Xhatn = np.concatenate([Xhatn,np.matmul(A_hat_opt,Xhatn[:,-1:])],axis=1)
    SSEn = np.sum((Xhatn[:,1:] - X[:,1:])**2)
    SSTn = np.sum(X[:,1:] ** 2)
    dict_r2[curve]['n-step'] = np.max([0,100*(1-SSEn/SSTn)])
    print('Curve : ' + str(curve), ' r2 accuracy [1-step] : ',dict_r2[curve]['1-step'] )
    print('Curve : ' + str(curve), ' r2 accuracy [n-step] : ', dict_r2[curve]['n-step'])
    print(' ')



