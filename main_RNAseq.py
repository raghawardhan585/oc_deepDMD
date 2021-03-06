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

##
# To get the RNAseq and OD data to RAW format of X and Y data
rnaf.organize_RNAseq_OD_to_RAWDATA(diff_Y = True)
# rnaf.organize_RNAseq_OD_to_RAWDATA(diff_Y = False)

## Open the RAW datafile

with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle', 'rb') as handle:
    dict_DATA_ORIGINAL = pickle.load(handle)
# dict_DATA = rnaf.filter_gene_by_coefficient_of_variation(dict_DATA, MEAN_TPM_THRESHOLD = 1, ALL_CONDITIONS= ['MX'])
dict_DATA_max_denoised = copy.deepcopy(dict_DATA_ORIGINAL)
# dict_DATA_max_denoised['MX'] = rnaf.denoise_using_PCA(dict_DATA_max_denoised['MX'], PCA_THRESHOLD = 99, NORMALIZE=True, PLOT_SCREE=False)



# dict_DATA_filt1 = rnaf.filter_gene_by_coefficient_of_variation(dict_DATA_filt0, MEAN_TPM_THRESHOLD = 100)
# dict_DATA_filt2 = rnaf.filter_gene_by_coefficient_of_variation(dict_DATA_filt1, CV_THRESHOLD = 0.25, ALL_CONDITIONS= ['MX'])
dict_MAX = rnaf.filter_gene_by_coefficient_of_variation(dict_DATA_max_denoised, MEAN_TPM_THRESHOLD = 400,ALL_CONDITIONS=['MX'])['MX']
# dict_MAX = dict_DATA
# MEAN THRES = 100
##
curve = 0
f,ax = plt.subplots(7,1,sharex=True,figsize=(30,14))
for time_pt in range(1,8):
    # for curve in range(16):
    ax[time_pt-1].plot(np.array(dict_DATA_ORIGINAL['MX'][curve]['df_X_TPM'].loc[:, time_pt]))
    # ax[time_pt - 1].plot(np.array(dict_DATA_max_denoised['MX'][curve]['df_X_TPM'].loc[:,time_pt]))
    # ax[time_pt - 1].plot(np.array(dict_MAX[curve]['df_X_TPM'].loc[:, time_pt]))
    ax[time_pt - 1].set_xlim([1200,1250])
    ax[time_pt - 1].set_ylim([0, 1000])
    ax[time_pt-1].set_title('Time Point : ' + str(time_pt),fontsize=24)
ax[-1].set_xlabel('Gene Locus Tag')
f.show()

##
plt.figure()
n = np.prod(dict_DATA_ORIGINAL['MX'][curve]['Y'].shape)
t = np.arange(1,1+n*3/60,3/60)
for curves in range(16):
    plt.plot(t,np.array(dict_DATA_ORIGINAL['MX'][curve]['Y']).T.reshape(-1),color='blue')
    plt.plot(t,np.array(dict_DATA_max_denoised['MX'][curve]['Y']).T.reshape(-1), color='green')
plt.xlabel('Time [Hr]]')
plt.ylabel('Optical Density')
plt.show()
## Sorting the MAX dataset to deepDMD format

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

SYSTEM_NO = 104
storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
if os.path.exists(storage_folder):
    # get_input = input('Do you wanna delete the existing system[y/n]? ')
    get_input = 'y'
    if get_input == 'y':
        shutil.rmtree(storage_folder)
        os.mkdir(storage_folder)
    else:
        quit(0)
else:
    os.mkdir(storage_folder)

_, dict_Scaler, _ = oc.scale_train_data(dict_DMD1, 'standard',WITH_MEAN_FOR_STANDARD_SCALER_X = False, WITH_MEAN_FOR_STANDARD_SCALER_Y = False)
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
# n_genes = dict_MAX['MX'][0]['df_X_TPM'].shape[0]
# for l in range(1):
#     igene = 2219#random.randint(0,n_genes)
#     plt.figure()
#     for i in range(16):
#         plt.plot(dict_MAX[i]['df_X_TPM'].iloc[igene,:])
#     plt.title('Gene ' + str(igene))
#     plt.show()

## DMD Stats

def ADD_BIAS_ROW(X_IN,ADD_BIAS):
    if ADD_BIAS:
        X_OUT = np.concatenate([X_IN, np.ones(shape=(1, X_IN.shape[1]))], axis=0)
    else:
        X_OUT = X_IN
    return X_OUT
def ADD_BIAS_COLUMN(X_IN,ADD_BIAS):
    if ADD_BIAS:
        X_OUT = np.concatenate([X_IN, np.ones(shape=(X_IN.shape[0], 1))], axis=1)
    else:
        X_OUT = X_IN
    return X_OUT
def REMOVE_BIAS_ROW(X_IN,ADD_BIAS):
    if ADD_BIAS:
        X_OUT = X_IN[0:-1,:]
    else:
        X_OUT = X_IN
    return X_OUT
def REMOVE_BIAS_COLUMN(X_IN,ADD_BIAS):
    if ADD_BIAS:
        X_OUT = X_IN[:,0:-1]
    else:
        X_OUT = X_IN
    return X_OUT


SYSTEM_NO = 104
ADD_BIAS = False
data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
with open(data_path,'rb') as handle:
    p = pickle.load(handle)

TRAIN_PERCENT = 50
VALID_PERCENT = 25
n_tot = p['Xp'].shape[0]
n_train = np.int(np.floor(n_tot*TRAIN_PERCENT/100))
n_valid = np.int(np.floor(n_tot*(TRAIN_PERCENT+VALID_PERCENT)/100))

p['Xp'] = ADD_BIAS_COLUMN(p['Xp'],ADD_BIAS)
p['Xf'] = ADD_BIAS_COLUMN(p['Xf'],ADD_BIAS)


Xp_train = p['Xp'][0:n_train].T
Xf_train = p['Xf'][0:n_train].T
Xp_valid = p['Xp'][n_train:].T
Xf_valid = p['Xf'][n_train:].T

Yp_train = p['Yp'][0:n_train].T
Yf_train = p['Yf'][0:n_train].T
Yp_valid = p['Yp'][n_train:].T
Yf_valid = p['Yf'][n_train:].T

# Optimal Number of Principal Components
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
nPC_opt_A = np.where(ls_error_valid == np.min(ls_error_valid))[0][0] + 1
print('Optimal Principal components : ', nPC_opt_A)
A_hat_opt = np.zeros(shape=U.shape)
for i in range(nPC_opt_A):
    A_hat_opt = A_hat_opt + (1 / S[i]) * np.matmul(np.matmul(Xf_train, V[:, i:i + 1]), Uh[i:i + 1, :])

# TODO - Might have to alter the last row of A



##
E = np.linalg.eigvals(A_hat_opt)

fig = plt.figure(figsize=(3.5,6))
ax = fig.add_subplot(2, 1, 2)
circ = plt.Circle((0, 0), radius=1, edgecolor='None', facecolor='cyan')
ax.add_patch(circ)
ax.plot(np.real(E),np.imag(E),'x',linewidth=3,color='g')
ax = fig.add_subplot(2, 1, 1)
ax = sb.heatmap(A_hat_opt, cmap="RdBu")
plt.show()
# ##
# # plt.plot(ls_error_train[55:])
# # plt.plot(ls_error_valid[55:])
# plt.plot(np.array(ls_error_train[55:]) + np.array(ls_error_valid[55:]))
# plt.show()

## Performance
SCALED_PERFORMACE = True
indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
with open(indices_path,'rb') as handle:
    ls_indices = pickle.load(handle)
raw_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
with open(raw_data_path,'rb') as handle:
    dict_RAWDATA = pickle.load(handle)

last_train_curve = np.int(np.floor(len(ls_indices)*(TRAIN_PERCENT)/100))
last_valid_curve = np.int(np.floor(len(ls_indices)*(TRAIN_PERCENT+VALID_PERCENT)/100))
ls_train_indices = ls_indices[0:last_train_curve]
ls_valid_indices = ls_indices[last_train_curve:last_valid_curve]
ls_test_indices = ls_indices[last_valid_curve:]


dict_r2 = {}
for curve in ls_indices:
    dict_r2[curve] ={}
    # Getting the scaled data
    X = np.array(dict_RAWDATA[curve]['df_X_TPM'])
    Xs = oc.scale_data_using_existing_scaler_folder({'X' : X.T}, SYSTEM_NUMBER=SYSTEM_NO)['X'].T

    Xs = ADD_BIAS_ROW(Xs, ADD_BIAS)

    # 1 - step
    Xhats = copy.deepcopy(Xs[:, 0:1])
    for i in range(len(X[0]) - 1):
        Xhats = np.concatenate([Xhats, np.matmul(A_hat_opt, Xs[:, i:(i+1)])], axis=1)

    Xhat = REMOVE_BIAS_ROW(Xhats, ADD_BIAS)
    Xhat = oc.inverse_transform_X(Xhat.T, SYSTEM_NO).T

    if SCALED_PERFORMACE:
        SSE = np.sum((Xhats[:, 1:] - Xs[:, 1:]) ** 2)
        SST = np.sum(Xs[:, 1:] ** 2)
    else:
        SSE = np.sum((Xhat[:,1:] - X[:,1:]) ** 2)
        SST = np.sum(X[:,1:] ** 2)
    dict_r2[curve]['1-step'] = np.max([0, 100 * (1 - SSE / SST)])

    # n - step
    Xhatns = copy.deepcopy(Xs[:,0:1])
    for i in range(len(X[0])-1):
        Xhatns = np.concatenate([Xhatns,np.matmul(A_hat_opt,Xhatns[:,-1:])],axis=1)

    Xhatn = REMOVE_BIAS_ROW(Xhatns, ADD_BIAS)
    Xhatn = oc.inverse_transform_X(Xhatn.T, SYSTEM_NO).T

    if SCALED_PERFORMACE:
        SSEn = np.sum((Xhatns[:, 1:] - Xs[:, 1:]) ** 2)
        SSTn = np.sum(Xs[:, 1:] ** 2)
    else:
        SSEn = np.sum((Xhatn[:,1:] - X[:,1:])**2)
        SSTn = np.sum(X[:,1:] ** 2)
    dict_r2[curve]['n-step'] = np.max([0,100*(1-SSEn/SSTn)])

    if curve == ls_train_indices[0]:
        print('=======================================================')
        print('========== TRAINING STATS ==========')
        print('=======================================================')
    elif curve == ls_valid_indices[0]:
        print('=======================================================')
        print('========== VALIDATION STATS ==========')
        print('=======================================================')
    elif curve == ls_test_indices[0]:
        print('=======================================================')
        print('========== TEST STATS ==========')
        print('=======================================================')
    print('Curve : ' + str(curve), ' r2 accuracy [1-step] : ',dict_r2[curve]['1-step'] )
    print('Curve : ' + str(curve), ' r2 accuracy [n-step] : ', dict_r2[curve]['n-step'])
    print(' ')






## Output Linear

# Model Y = W*X

U,S,Vh = np.linalg.svd(Xp_train)
# plt.stem((1-np.cumsum(S**2)/np.sum(S**2))*100)
# plt.show()
V = Vh.T.conj()
Uh = U.T.conj()
C_hat = np.zeros(shape=(Yp_train.shape[0],Xp_train.shape[0]))
ls_error_train = []
ls_error_valid = []
for i in range(len(S)):
    C_hat = C_hat + (1 / S[i]) * np.matmul(np.matmul(Yp_train, V[:, i:i + 1]), Uh[i:i + 1, :])
    ls_error_train.append(np.mean(np.square((Yp_train - np.matmul(C_hat, Xp_train)))))
    if Xp_valid.shape[1] != 0:
        ls_error_valid.append(np.mean(np.square((Yp_valid - np.matmul(C_hat, Xp_valid)))))
if Xp_valid.shape[1] == 0:
    ls_error = np.array(ls_error_train)
else:
    ls_error = np.array(ls_error_train) + np.array(ls_error_valid)
# nPC_opt = np.where(ls_error == np.min(ls_error))[0][0] + 1
nPC_opt_C = np.where(ls_error_valid == np.min(ls_error_valid))[0][0] + 1
print('Optimal Principal components : ', nPC_opt_C)
C_hat_opt = np.zeros(shape=(Yp_train.shape[0],Xp_train.shape[0]))
for i in range(nPC_opt_C):
    C_hat_opt = C_hat_opt + (1 / S[i]) * np.matmul(np.matmul(Yp_train, V[:, i:i + 1]), Uh[i:i + 1, :])

## Performance of Output
SCALED_PERFORMACE = True

indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
with open(indices_path,'rb') as handle:
    ls_indices = pickle.load(handle)
raw_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
with open(raw_data_path,'rb') as handle:
    dict_RAWDATA = pickle.load(handle)


last_train_curve = np.int(np.floor(len(ls_indices)*(TRAIN_PERCENT)/100))
last_valid_curve = np.int(np.floor(len(ls_indices)*(TRAIN_PERCENT+VALID_PERCENT)/100))
ls_train_indices = ls_indices[0:last_train_curve]
ls_valid_indices = ls_indices[last_train_curve:last_valid_curve]
ls_test_indices = ls_indices[last_valid_curve:]


dict_r2 = {}
for curve in ls_indices:
    dict_r2[curve] ={}
    X = np.array(dict_RAWDATA[curve]['df_X_TPM'])
    Y = np.array(dict_RAWDATA[curve]['Y'])
    # Getting the scaled data
    Xs = oc.scale_data_using_existing_scaler_folder({'X' : X.T}, SYSTEM_NUMBER=SYSTEM_NO)['X'].T

    Xs = ADD_BIAS_ROW(Xs, ADD_BIAS)

    Ys = oc.scale_data_using_existing_scaler_folder({'Y': Y.T},SYSTEM_NUMBER=SYSTEM_NO)['Y'].T

    # 1 - step
    Xhat1s = copy.deepcopy(Xs[:, 0:1])
    for i in range(len(Xs[0]) - 1):
        Xhat1s = np.concatenate([Xhat1s, np.matmul(A_hat_opt, Xs[:, i:(i+1)])], axis=1)
    Yhat1s = np.matmul(C_hat_opt, Xhat1s)

    Xhat1 = REMOVE_BIAS_ROW(Xhat1s, ADD_BIAS)
    Xhat1 = oc.inverse_transform_X(Xhat1.T, SYSTEM_NO).T

    Yhat1 = oc.inverse_transform_Y(Yhat1s.T, SYSTEM_NO).T
    if SCALED_PERFORMACE:
        SSE = np.sum((Xhat1s[:,1:] - Xs[:,1:]) ** 2) + np.sum((Yhat1s - Ys) ** 2)
        SST = np.sum(Xs[:,1:] ** 2) + np.sum(Ys ** 2)
    else:
        SSE = np.sum((Xhat1[:, 1:] - X[:, 1:]) ** 2) + np.sum((Yhat1 - Y) ** 2)
        SST = np.sum(X[:, 1:] ** 2) + np.sum(Y ** 2)
    dict_r2[curve]['1-step'] = np.max([0, 100 * (1 - SSE / SST)])


    # n - step
    Xhatns = copy.deepcopy(Xs[:,0:1])
    for i in range(len(Xs[0])-1):
        Xhatns = np.concatenate([Xhatns,np.matmul(A_hat_opt,Xhatns[:,-1:])],axis=1)
    Yhatns = np.matmul(C_hat_opt, Xhatns)

    Xhatn = REMOVE_BIAS_ROW(Xhatns, ADD_BIAS)
    Xhatn = oc.inverse_transform_X(Xhatn.T, SYSTEM_NO).T

    Yhatn = oc.inverse_transform_Y(Yhatns.T, SYSTEM_NO).T
    if SCALED_PERFORMACE:
        SSEn = np.sum((Xhatns[:,1:] - Xs[:,1:]) ** 2) + np.sum((Yhatns - Ys) ** 2)
        SSTn = np.sum(Xs[:,1:] ** 2) + np.sum(Ys ** 2)
    else:
        SSEn = np.sum((Xhatn[:,1:] - X[:,1:])**2) + np.sum((Yhatn - Y) ** 2)
        SSTn = np.sum(X[:,1:] ** 2) + np.sum(Y ** 2)
    dict_r2[curve]['n-step'] = np.max([0,100*(1-SSEn/SSTn)])


    if curve == ls_train_indices[0]:
        print('=======================================================')
        print('========== TRAINING STATS ==========')
        print('=======================================================')
    elif curve == ls_valid_indices[0]:
        print('=======================================================')
        print('========== VALIDATION STATS ==========')
        print('=======================================================')
    elif curve == ls_test_indices[0]:
        print('=======================================================')
        print('========== TEST STATS ==========')
        print('=======================================================')
    print('Curve : ' + str(curve), ' r2 accuracy [1-step] : ',dict_r2[curve]['1-step'] )
    print('Curve : ' + str(curve), ' r2 accuracy [n-step] : ', dict_r2[curve]['n-step'])
    # print('Curve : ' + str(curve), ' r2 accuracy [n-step] : ', dict_r2[curve]['output'])
    print(' ')
    # plt.plot()

sb.heatmap(C_hat_opt, cmap="YlGnBu")
plt.xlabel('Gene Locus Tag')
plt.ylabel('Output Number')
plt.show()

##
hmap_data = np.log10(X)#[:,0:3])
hmap_data[hmap_data == np.inf] = np.nan
hmap_data[hmap_data == -np.inf] = np.nan
this_center = np.nanmean(hmap_data)
# sb.heatmap(hmap_data,vmax = 5.0,vmin = -0.5,center=this_center,cmap = sb.diverging_palette(240, 11.75, s=99, l=30.2, n=15))
sb.heatmap(hmap_data,vmax = 5.0,vmin = -0.5,center=this_center,cmap = 'YlGn')
plt.show()


## STRATEGY 1 - ORDER GENES BY THE ENEREGY CONTRIBUTED TO THE OUTPUT
CURVE_NO = 0
dict_DATA = dict_RAWDATA[CURVE_NO]
X = np.array(dict_DATA['df_X_TPM'])
Y = np.array(dict_DATA['Y'])
Xs = oc.scale_data_using_existing_scaler_folder({'X' : X.T}, SYSTEM_NUMBER=SYSTEM_NO)['X'].T
Ys = oc.scale_data_using_existing_scaler_folder({'Y': Y.T},SYSTEM_NUMBER=SYSTEM_NO)['Y'].T
dict_GENE = {}
ls_time_points = list(dict_DATA['df_X_TPM'].columns)
ls_gene = list(dict_DATA['df_X_TPM'].index)
for i in range(len(ls_gene)):
    gene = ls_gene[i]
    dict_GENE[gene] = {'Wh val': np.mean(C_hat_opt[:,i]), 'Y energy': np.sum(np.matmul(C_hat_opt[:,i:(i+1)],Xs[i:(i+1),:]))/np.sum(Ys)*100}


pd.DataFrame(dict_GENE).T.sort_values('Y energy',ascending = False)

## STRATEGY 2 - SENSITIVITY AT TIMEPOINTS 1 and 4
for i in range(len(ls_gene)):
    Xi = copy.deepcopy(X[:,0:1])
    delta_x = Xi[i][0] - Xi[i][0]/2
    Xi[i][0] = Xi[i][0]/2
    Xhats = oc.scale_data_using_existing_scaler_folder({'X': Xi.T}, SYSTEM_NUMBER=SYSTEM_NO)['X'].T
    for tp in range(len(ls_time_points)):
        Xhats = np.concatenate([Xhats, np.matmul(A_hat_opt, Xhats[:, -1:])], axis=1)
    Yhats = np.matmul(C_hat_opt,Xhats)
    Yhat = oc.inverse_transform_Y(Yhats.T, SYSTEM_NO).T
    for tp in range(len(ls_time_points)):
        dict_GENE[ls_gene[i]]['Sensitivity tp '+str(ls_time_points[tp])] = np.mean(Y[:,tp] - Yhat[:,tp])/delta_x

d = pd.DataFrame(dict_GENE).T.sort_values('Sensitivity tp 1',ascending = False)

## STRATEGY 3 - MODAL ANALYSIS
def resolve_complex_right_eigenvalues(E, W):
    eval = copy.deepcopy(np.diag(E))
    comp_modes = []
    comp_modes_conj = []
    for i1 in range(E.shape[0]):
        if np.imag(E[i1, i1]) != 0:
            print(i1)
            # Find the complex conjugate
            for i2 in range(i1 + 1, E.shape[0]):
                if eval[i2] == eval[i1].conj():
                    break
            # i1 and i2 are the indices of the complex conjugate eigenvalues
            comp_modes.append(i1)
            comp_modes_conj.append(i2)
            E[i1, i1] = np.real(eval[i1])
            E[i2, i2] = np.real(eval[i1])
            E[i1, i2] = np.imag(eval[i1])
            E[i2, i1] = - np.imag(eval[i1])
            u1 = copy.deepcopy(np.real(W[:, i1:i1 + 1]))
            w1 = copy.deepcopy(np.imag(W[:, i1:i1 + 1]))
            W[:, i1:i1 + 1] = u1
            W[:, i2:i2 + 1] = w1
    E_out = np.real(E)
    W_out = np.real(W)
    return E_out, W_out, comp_modes, comp_modes_conj


# n_modes = 4
# Ur = U[:,0:n_modes]
# A_tilde = np.matmul(Ur.T,np.matmul(A_hat_opt,Ur))
# Eraw,Vraw = np.linalg.eig(A_tilde)

Eraw, Vraw = np.linalg.eig(A_hat_opt)

E,V,comp_modes, comp_modes_conj = resolve_complex_right_eigenvalues(np.diag(Eraw), Vraw)

##
CURVE_NO = 0
dict_DATA = dict_RAWDATA[CURVE_NO]
X = np.array(dict_DATA['df_X_TPM'])
Y = np.array(dict_DATA['Y'])
Xs = oc.scale_data_using_existing_scaler_folder({'X' : X.T}, SYSTEM_NUMBER=SYSTEM_NO)['X'].T
Ys = oc.scale_data_using_existing_scaler_folder({'Y': Y.T},SYSTEM_NUMBER=SYSTEM_NO)['Y'].T

Zs = np.matmul(np.linalg.inv(V),Xs)

##
plt.figure()
for i in range(Zs.shape[0]):
    if np.mean(np.abs(Zs[i,:]))>4000:# and np.mean(np.abs(Zs[i,:]))>20:
        plt.plot([1,2,3,4,5,6,7],Zs[i,:])
    # else:
    #     print(np.mean(np.abs(Zs[i,:])))
plt.ylabel('Scaled modal values')
plt.title('Modes as a function of time')
plt.xlabel('Time [hr]')
plt.show()

## Attempting Regularization
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
from sklearn.metrics import make_scorer,r2_score
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
##
# kf = KFold(n_splits=3)
X = np.random.normal(0,5,size=(100,3))
Y = np.matmul(X,np.random.normal(0,1,size=(3,3))) + np.random.normal(0,1,size=(100,3))
kf = KFold(n_splits=5,shuffle=False,random_state=None)
# for train_index,test_index in kf.split(X):
#     X_train,X_test,Y_train,Y_test = X[train_index,:],X[test_index,:])


# cross_val_score(Ridge(),X,Y)
# my_scorer = make_scorer(r2_score,multioutput='variance_weighted')
my_scorer = make_scorer(r2_score,multioutput='uniform_average')
cross_val_score(LinearRegression(fit_intercept=False),X,Y,cv=kf.split(X), scoring= my_scorer)

##




