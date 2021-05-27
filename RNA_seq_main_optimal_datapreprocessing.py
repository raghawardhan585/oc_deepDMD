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
from sklearn.metrics import make_scorer,r2_score
plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22
##
# To get the RNAseq and OD data to RAW format of X and Y data
rnaf.organize_RNAseq_OD_to_RAWDATA(get_fitness_output = True)
# rnaf.organize_RNAseq_OD_to_RAWDATA(get_fitness_output = False)

## Open the RAW datafile


with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle', 'rb') as handle:
    dict_DATA_ORIGINAL = pickle.load(handle)
# dict_DATA = rnaf.filter_gene_by_coefficient_of_variation(dict_DATA, MEAN_TPM_THRESHOLD = 1, ALL_CONDITIONS= ['MX'])
dict_DATA_max_denoised = copy.deepcopy(dict_DATA_ORIGINAL)
# dict_DATA_max_denoised['MX'] = rnaf.denoise_using_PCA(dict_DATA_max_denoised['MX'], PCA_THRESHOLD = 99, NORMALIZE=True, PLOT_SCREE=False)

ls_MEAN_TPM_THRESHOLD = [2,3,4]
ls_CV_THRESHOLD = [0.102,0.103,0.104]
# ls_MEAN_TPM_THRESHOLD = [0,5]
# ls_CV_THRESHOLD = [0.1,1]
NO_OF_FOLDS = 16
kf = KFold(n_splits=NO_OF_FOLDS, shuffle=False, random_state=None)
my_scorer = make_scorer(r2_score,multioutput='uniform_average')

dict_score = {}
for MEAN_TPM_THRESHOLD in ls_MEAN_TPM_THRESHOLD:
    dict_score[MEAN_TPM_THRESHOLD] = {}
for MEAN_TPM_THRESHOLD,CV_THRESHOLD in itertools.product(ls_MEAN_TPM_THRESHOLD,ls_CV_THRESHOLD):
    print('MEAN TPM THRESHOLD: ',MEAN_TPM_THRESHOLD,'   CV_THRESHOLD: ',CV_THRESHOLD)
    dict_MAX = rnaf.filter_gene_by_coefficient_of_variation(copy.deepcopy(dict_DATA_max_denoised), MEAN_TPM_THRESHOLD = MEAN_TPM_THRESHOLD, CV_THRESHOLD = CV_THRESHOLD,ALL_CONDITIONS=['MX'])['MX']
    ls_all_indices = list(dict_MAX.keys())
    n_states = dict_MAX[ls_all_indices[0]]['df_X_TPM'].shape[0]
    n_outputs = dict_MAX[ls_all_indices[0]]['Y'].shape[0]
    # DMD formulation
    dict_DMD = {'Xp' : np.empty(shape=(0,n_states)), 'Xf': np.empty(shape=(0,n_states)),'Yp' : np.empty(shape=(0,n_outputs)), 'Yf' : np.empty(shape=(0,n_outputs))}
    for i in ls_all_indices:
        dict_DMD['Xp'] = np.concatenate([dict_DMD['Xp'], np.array(dict_MAX[i]['df_X_TPM'].iloc[:,0:-1]).T],axis=0)
        dict_DMD['Xf'] = np.concatenate([dict_DMD['Xf'], np.array(dict_MAX[i]['df_X_TPM'].iloc[:, 1:]).T], axis=0)
        dict_DMD['Yp'] = np.concatenate([dict_DMD['Yp'], np.array(dict_MAX[i]['Y'].iloc[:, 0:-1]).T], axis=0)
        dict_DMD['Yf'] = np.concatenate([dict_DMD['Yf'], np.array(dict_MAX[i]['Y'].iloc[:, 1:]).T], axis=0)
    # Scaling the data
    dict_DMDs, dict_Scaler, _ = oc.scale_train_data(dict_DMD, 'standard', WITH_MEAN_FOR_STANDARD_SCALER_X=True,
                                            WITH_MEAN_FOR_STANDARD_SCALER_Y=True)
    dict_score[MEAN_TPM_THRESHOLD][CV_THRESHOLD] = np.mean(cross_val_score(LinearRegression(fit_intercept=True), dict_DMDs['Xp'], dict_DMDs['Xf'],
                    cv=kf.split(dict_DMDs['Xp']), scoring=my_scorer))
df_score = np.maximum(0,pd.DataFrame(dict_score))

##
plt.figure(figsize=(6,6))
a = sb.heatmap(df_score,vmin=0,vmax=1,center=0.5,cmap=sb.diverging_palette(240, 11.75, s=99, l=30.2, n=15),annot=True)
b, t = a.axes.get_ylim()  # discover the values for bottom and top
b += 0.5  # Add 0.5 to the bottom
t -= 0.5  # Subtract 0.5 from the top
a.axes.set_ylim(b, t)  # update the ylim(bottom, top) values
a.axes.set_xlabel('mean TPM threshold')
a.axes.set_ylabel('CV threshold')
plt.show()
##
ls_MEAN_TPM_THRESHOLD = list(range(0,1000,10))
ls_del_gene = []
for i in ls_MEAN_TPM_THRESHOLD:
    dict_MAX = rnaf.filter_gene_by_coefficient_of_variation(copy.deepcopy(dict_DATA_max_denoised),MEAN_TPM_THRESHOLD=i, ALL_CONDITIONS=['MX'])['MX']
    ls_del_gene.append(len(dict_MAX[0]['df_X_TPM']))
##
plt.figure()
plt.plot(ls_MEAN_TPM_THRESHOLD,np.log10(ls_del_gene))
plt.xlabel('Mean TPM Threshold')
plt.ylabel('Genes Retained')
plt.show()