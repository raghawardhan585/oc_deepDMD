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
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
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


## Filtering the genes based on variance (a metric to quantify the dynamics)
dict_growthcurve = copy.deepcopy(dict_DATA_ORIGINAL)
ls_allconditions = ['MX','MN','NC']
n_conditions = len(ls_allconditions)
ls_allgenes = list(dict_growthcurve[ls_allconditions[0]][0]['df_X_TPM'].index)
ls_required_genes = []
dict_df_var = {}
for cond_no in range(n_conditions):
    cond = ls_allconditions[cond_no]
    df_gene_variance = pd.DataFrame([])
    for replicate in dict_growthcurve[cond].keys():
        df_temp1 = dict_growthcurve[cond][replicate]['df_X_TPM']
        # Check the differential expression across time points
        try:
            df_gene_variance.loc[:,replicate] = df_temp1.var(axis=1)
        except:
            df_gene_variance = pd.DataFrame(df_temp1.var(axis=1), columns=[replicate])
    dict_df_var[cond] = copy.deepcopy(df_gene_variance)

df_gene_mean_ofvariance = pd.DataFrame([])
for cond in ls_allconditions:
    try:
        df_gene_mean_ofvariance.loc[:, cond] = dict_df_var[cond].mean(axis=1)
    except:
        df_gene_mean_ofvariance = pd.DataFrame(dict_df_var[cond].mean(axis=1),columns = cond)


df_differential_expression = np.log2(copy.deepcopy(df_gene_mean_ofvariance).divide(df_gene_mean_ofvariance.loc[:,'MN'],axis=0))

df_differential_expression[df_differential_expression.loc[:,'MX']>5]
# ##
# plt.figure()
# ls_sorted_genes_MAX = list(dict_df_var_mean['MX'].sort_values(ascending = False).index)[100:200]
# for cond in ls_allconditions:
#     plt.plot(np.log10(np.array(dict_df_var_mean[cond][ls_sorted_genes_MAX])),label=cond)
# plt.legend()
# plt.show()


ls_filtered_genes = list(df_differential_expression[df_differential_expression.loc[:,'MX']>5].index)

# print('# Intersecting genes: ', len(list(set(ls_filtered_genes).intersection(ls_genes_biocyc))))

# Filtering genes based on the significance of the dynamics
df_temp1 = dict_growthcurve['MX'][0]['df_X_TPM'] + 1e-2

p = np.abs(np.log2(df_temp1.divide(df_temp1.iloc[:,-1],axis=0))).iloc[:,0].sort_values(ascending = False)
# plt.plot(np.array())

##

## Filtering based on the linear dynamics
ALL_CONDITIONS =['MX']
dict_data = rnaf.filter_gene_by_coefficient_of_variation(dict_DATA_max_denoised, CV_THRESHOLD = 0.1,ALL_CONDITIONS=ALL_CONDITIONS)
ls_data_indices = list(dict_data[ALL_CONDITIONS[0]].keys())

# Form the training dataset
Xp = Xf = np.array([])
for cond, i in itertools.product(ALL_CONDITIONS,ls_data_indices):
    try:
        Xp = pd.concat([Xp,dict_data[cond][i]['df_X_TPM'].iloc[:,0:-1]],axis=1)
        Xf = pd.concat([Xf, dict_data[cond][i]['df_X_TPM'].iloc[:, 1:]], axis=1)
    except:
        Xp = dict_data[cond][i]['df_X_TPM'].iloc[:,0:-1]
        Xf = dict_data[cond][i]['df_X_TPM'].iloc[:, 1:]

ls_genes = list(Xp.index)
XpT = np.array(Xp).T
XfT = np.array(Xf).T

X_scale = MinMaxScaler()
X_scale.fit(XpT)
XpTs = X_scale.transform(XpT)
XfTs = X_scale.transform(XfT)

dict_empty_all_conditions ={}
for COND in ALL_CONDITIONS:
    dict_empty_all_conditions[COND] = {}

dict_scaled_data = copy.deepcopy(dict_empty_all_conditions)
dict_unscaled_data = copy.deepcopy(dict_empty_all_conditions)
for cond,i in itertools.product(ALL_CONDITIONS,ls_data_indices):
    dict_unscaled_data[cond][i] = {'XpT': np.array(dict_data[cond][i]['df_X_TPM'].iloc[:, 0:-1]).T,
                                   'XfT': np.array(dict_data[cond][i]['df_X_TPM'].iloc[:, 1:]).T}
    dict_scaled_data[cond][i] = {'XpT': X_scale.transform(dict_unscaled_data[cond][i]['XpT']),
                                 'XfT': X_scale.transform(dict_unscaled_data[cond][i]['XfT'])}

A = LinearRegression(fit_intercept= True)
A.fit(XpTs,XfTs)

dict_score = {}
for j in range(len(ls_genes)):
    if np.mod(j,100) ==0:
        print('Gene:',j+1,'/',len(ls_genes))
    # dict_score[ls_genes[j]] = {}
    Ac = copy.deepcopy(A)
    Ac.coef_[:,j] = 0
    Ac.coef_[j,:] = 0
    Ac.intercept_[j] = 0
    # for cond, i in itertools.product(ALL_CONDITIONS,ls_data_indices):
    #     # n_step predictions
    #     XfTs_hat = dict_scaled_data[cond][i]['XpT'][0:1]
    #     for step in range(len(dict_scaled_data[cond][i]['XfT'])):
    #         XfTs_hat = np.concatenate([XfTs_hat, Ac.predict(XfTs_hat[-1:])],axis=0)
    #     XfTs_hat = XfTs_hat[1:]
    #     dict_score[ls_genes[j]][cond+'_'+str(i)] = r2_score(np.delete(dict_scaled_data[cond][i]['XfT'],j,1),np.delete(X_scale.inverse_transform(XfTs_hat),j,1))
    # Based on 1-step predictions
    XfTs_hat = Ac.predict(XpTs)
    dict_score[ls_genes[j]] = r2_score(np.delete(XfT,j,1),np.delete(X_scale.inverse_transform(XfTs_hat),j,1))
# df_score = pd.DataFrame(dict_score).T



