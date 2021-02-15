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
pd.set_option("display.max_rows", None, "display.max_columns", None)

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

# Open the RAW datafile

# with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle', 'rb') as handle:
#     dict_DATA_ORIGINAL = pickle.load(handle)
#
# dict_DATA_MAX = rnaf.filter_gene_by_coefficient_of_variation(dict_DATA_ORIGINAL, MEAN_TPM_THRESHOLD = 10,ALL_CONDITIONS=['MX'])['MX']

# ## Train Test Split
# TRAIN_PERCENT = 75
# NO_OF_FOLDS = 3
#
# ls_indices = list(dict_DATA_MAX.keys())
# random.shuffle(ls_indices)
# ls_indices_train = ls_indices[0:np.int(np.floor(len(ls_indices) * TRAIN_PERCENT / 100))]
# ls_indices_train.sort()
# ls_indices_test = list(set(ls_indices) - set(ls_indices_train))
# ls_indices_test.sort()
# ls_indices.sort()
#
# n_genes = dict_DATA_MAX[ls_indices[0]]['df_X_TPM'].shape[0]
# n_outputs = dict_DATA_MAX[ls_indices[0]]['Y'].shape[0]
#
# n_states = dict_DATA_MAX [ls_indices_train[0]]['df_X_TPM'].shape[0]
# n_outputs = dict_DATA_MAX [ls_indices_train[0]]['Y'].shape[0]
# dict_DMD_train = {'Xp' : np.empty(shape=(0,n_states)), 'Xf': np.empty(shape=(0,n_states)),'Yp' : np.empty(shape=(0,n_outputs)), 'Yf' : np.empty(shape=(0,n_outputs))}
# for i in ls_indices_train:
#     dict_DMD_train['Xp'] = np.concatenate([dict_DMD_train['Xp'], np.array(dict_DATA_MAX[i]['df_X_TPM'].iloc[:,0:-1]).T],axis=0)
#     dict_DMD_train['Xf'] = np.concatenate([dict_DMD_train['Xf'], np.array(dict_DATA_MAX[i]['df_X_TPM'].iloc[:, 1:]).T], axis=0)
#     dict_DMD_train['Yp'] = np.concatenate([dict_DMD_train['Yp'], np.array(dict_DATA_MAX[i]['Y'].iloc[:, 0:-1]).T], axis=0)
#     dict_DMD_train['Yf'] = np.concatenate([dict_DMD_train['Yf'], np.array(dict_DATA_MAX[i]['Y'].iloc[:, 1:]).T], axis=0)
#
#
# SYSTEM_NO = 104
# storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
# if os.path.exists(storage_folder):
#     # get_input = input('Do you wanna delete the existing system[y/n]? ')
#     get_input = 'y'
#     if get_input == 'y':
#         shutil.rmtree(storage_folder)
#         os.mkdir(storage_folder)
#     else:
#         quit(0)
# else:
#     os.mkdir(storage_folder)
#
# # Scaling made on training data
# _, dict_Scaler, _ = oc.scale_train_data(dict_DMD_train, 'standard')
#
# with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle', 'wb') as handle:
#     pickle.dump(dict_Scaler, handle)
# dict_DATA_OUT = oc.scale_data_using_existing_scaler_folder(dict_DMD_train, SYSTEM_NO)
# with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle', 'wb') as handle:
#     pickle.dump(dict_DATA_OUT, handle)
# with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_Data.pickle', 'wb') as handle:
#     pickle.dump(dict_DATA_MAX, handle)
# with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle', 'wb') as handle:
#     pickle.dump(ls_indices_train, handle)  # Only training and validation indices are stored
# # Store the data in Koopman
# with open('/Users/shara/Desktop/oc_deepDMD/koopman_data/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle','wb') as handle:
#     pickle.dump(dict_DATA_OUT, handle)

## DMD Train with Lasso Regression

SYSTEM_NO = 104
NO_OF_FOLDS = 12

data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
with open(data_path,'rb') as handle:
    q = pickle.load(handle)

p = copy.deepcopy(q)
data_test ={}
for items in p:
    data_test[items] = p[items][72:]
    p[items] = p[items][0:72]
# TODO -  Go for k-fold cross validation directly

ADD_BIAS = False
p['Xp'] = ADD_BIAS_COLUMN(p['Xp'],ADD_BIAS)
p['Xf'] = ADD_BIAS_COLUMN(p['Xf'],ADD_BIAS)

kf = KFold(n_splits=NO_OF_FOLDS, shuffle=False, random_state=None)
my_scorer = make_scorer(r2_score,multioutput='uniform_average')
print(cross_val_score(LinearRegression(fit_intercept=False), p['Xp'], p['Xf'], cv=kf.split(p['Xp']),scoring=my_scorer))
print(cross_val_score(Lasso(alpha= 0.02, fit_intercept=False, max_iter=50000), p['Xp'], p['Xf'],cv=kf.split(p['Xp']), scoring=my_scorer))

##
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
            model_1 = Lasso(alpha = alpha, fit_intercept=False, max_iter= 50000)
        model_1.fit(Xp_train,Xf_train)
        Xfhat_train = model_1.predict(Xp_train)
        Xfhat_valid = model_1.predict(Xp_valid)
        dict_results[alpha][iter] = r2_score(Xf_valid, Xfhat_valid)
        # print(r2_score(Xf_train,Xfhat_train))
        # print(r2_score(Xf_valid, Xfhat_valid))
        # print(r2_score(np.concatenate([Xf_train,Xf_valid],axis=0), np.concatenate([Xfhat_train,Xfhat_valid],axis=0) ))
    print('Alpha = ',alpha, ' r2 = ', [dict_results[alpha][i] for i in dict_results[alpha].keys()])



##
x_val = []
y_val = []
for alpha in dict_results.keys():
    for iter in dict_results[alpha].keys():
        x_val.append(alpha)
        y_val.append(dict_results[alpha][iter])
plt.plot(x_val,y_val,'.')
plt.show()

##



my_scorer = make_scorer(r2_score,multioutput='uniform_average')
print(cross_val_score(LinearRegression(fit_intercept=False), p['Xp'], p['Yp'], cv=kf.split(p['Xp']),scoring=my_scorer))
print(cross_val_score(Lasso(alpha= 0.001, fit_intercept=False, max_iter=100000), p['Xp'], p['Yp'],cv=kf.split(p['Xp']), scoring=my_scorer))
# cross_val_score(Lasso(alpha = 0.5, fit_intercept=False),p['Xp'],p['Xf'],cv=kf.split(p['Xp']), scoring= my_scorer)
# cross_val_score(LinearRegression(fit_intercept=False),p['Xp'],p['Xf'],cv=kf.split(p['Xp']), scoring= my_scorer)

##
# dict_results = {}
# for alpha in np.arange(0.,0.1,0.005):
for alpha in np.arange(0.1, 0.5, 0.05):
    # if alpha in np.arange(0.01,0.04,0.01):
    #     continue
    if alpha == 0:
        A = cross_val_score(LinearRegression(fit_intercept=False), p['Xp'], p['Yp'], cv=kf.split(p['Xp']),scoring=my_scorer)
    else:
        A = cross_val_score(Lasso(alpha=alpha, fit_intercept=False, max_iter=100000), p['Xp'], p['Yp'],cv=kf.split(p['Xp']), scoring=my_scorer)
    print('alpha = ',alpha)
    print(A)
    dict_results[alpha] = A
##
# df_p = pd.DataFrame(dict_results)
xval = np.array([])
yval = np.array([])
x_i = []
x_mu = []
x_std = []
for items in dict_results:
    xval = np.concatenate([xval,items*np.ones(len(dict_results[items]))])
    yval = np.concatenate([yval,dict_results[items]])
    x_i.append(items)
    x_mu.append(np.mean(dict_results[items]))
    x_std.append(np.std(dict_results[items]))
y_low = np.array(x_mu) - np.array(x_std)
y_high = np.array(x_mu) + np.array(x_std)

plt.figure(figsize=(5,4))
plt.plot(xval,yval,'.')
plt.plot(x_i,x_mu)
for i in range(len(x_i)):
    plt.plot([x_i[i],x_i[i]],[y_low[i],y_high[i]],'r')
plt.show()
## alpha vs r2

SYSTEM_NO = 104
NO_OF_FOLDS = 12
LAMBDA_1 = 0.01
LAMBDA_2 = 0.06
alpha = LAMBDA_1 + LAMBDA_2
l1_ratio = LAMBDA_1/(LAMBDA_1 + LAMBDA_2)

full_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
with open(full_data_path,'rb') as handle:
    dict_MAX = pickle.load(handle)
ocDMD_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
with open(ocDMD_data_path,'rb') as handle:
    q = pickle.load(handle)

p = copy.deepcopy(q)
data_test ={}
for items in p:
    data_test[items] = p[items][72:]
    p[items] = p[items][0:72]

kf = KFold(n_splits=NO_OF_FOLDS, shuffle=False, random_state=None)
model_1 = Lasso(alpha = LAMBDA_1, fit_intercept=False, max_iter= 50000)
# model_1 = Ridge(alpha = LAMBDA_2, fit_intercept=False, max_iter= 5000)
# model_1 = ElasticNet(alpha = alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter= 5000)
# model_1 = LinearRegression(fit_intercept=False)
model_1.fit(p['Xp'],p['Xf'])
Xfhat_train = model_1.predict(p['Xp'])
Xfhat_test = model_1.predict(data_test['Xp'])
print('Training Error r2 = ', np.round(r2_score(p['Xf'],Xfhat_train)*100,2),'%')
print('Testing Error r2  = ', np.round(r2_score(data_test['Xf'],Xfhat_test)*100,2),'%')
A = model_1.coef_
E,V = np.linalg.eig(A)

plt.figure(figsize=(6,6))
sb.heatmap(A, cmap="RdBu",center=0)
plt.show()

fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(1, 1, 1)
circ = plt.Circle((0, 0), radius=1, edgecolor='None', facecolor='cyan')
ax.add_patch(circ)
ax.plot(np.real(E),np.imag(E),'x',linewidth=3,color='g')
plt.show()

print('Rank of A :', np.linalg.matrix_rank(A))
print('Zero Elements Percentage:', np.round(np.sum(A==0)/np.prod(A.shape)*100,2),'%')

## Finding the interconnected components
# A = A.T
epsilon = 1e-2
dict_interconnect = {}
interconnect_index = 0
for i in range(len(A)):
    A_i = A[i]
    ls_connect = copy.deepcopy([j for j in range(len(A_i)) if np.abs(A_i[j]) >epsilon])
    if i not in ls_connect:
        ls_connect.append(i)
    # print(ls_connect)
    if list(dict_interconnect.keys()) == []:
        dict_interconnect[interconnect_index] = copy.deepcopy(ls_connect)
        interconnect_index = interconnect_index + 1
    else:
        ITEM_FOUND = False
        for items in dict_interconnect:
            SUB_ITEM_FOUND = False
            for item_j in ls_connect:
                if item_j in dict_interconnect[items]:
                    dict_interconnect[items] = copy.deepcopy(list(set(dict_interconnect[items]).union(set(ls_connect))))
                    SUB_ITEM_FOUND = True
                    break
            if SUB_ITEM_FOUND:
                ITEM_FOUND = True
                break
        if not ITEM_FOUND:
            # print('Item ',i,' not found!')
            dict_interconnect[interconnect_index] = ls_connect
            interconnect_index = interconnect_index + 1

print('Number of interconnected genes: ',len(dict_interconnect))


## DMD modes

Xp = p['Xp'].T
Xf = p['Xf'].T
U,S,VT = np.linalg.svd(Xp)

plt.stem(np.arange(0,11),np.concatenate([np.array([100]),((1 - np.cumsum(S**2)/np.sum(S**2))*100)[0:10]],axis=0))
plt.show()
##
Ur = U[:,0:6]
Ar = np.matmul(Ur.T,np.matmul(A,Ur))
Zp = np.matmul(Ur.T,Xp)

##
LAMBDA_1y = 0.045
LAMBDA_2y = 0.1
alphay = LAMBDA_1y + LAMBDA_2y
l1_ratioy = LAMBDA_1y/(LAMBDA_1y + LAMBDA_2y)

model_2 = Lasso(alpha = LAMBDA_1y, fit_intercept=False, max_iter= 50000)
# model_2 = Ridge(alpha = LAMBDA_2y, fit_intercept=False, max_iter= 5000)
# model_2 = ElasticNet(alpha = alphay, l1_ratio=l1_ratioy, fit_intercept=False, max_iter= 5000)
# model_2 = LinearRegression(fit_intercept=False)
model_2.fit(p['Xp'],p['Yp'])
Yphat_train = model_2.predict(p['Xp'])
Yphat_test = model_2.predict(data_test['Xp'])
print('Training Error r2 = ', np.round(r2_score(p['Yp'],Yphat_train)*100,2),'%')
print('Testing Error r2  = ', np.round(r2_score(data_test['Yp'],Yphat_test)*100,2),'%')

Wh = model_2.coef_

print('Zero Elements Percentage in Wh:', np.round(np.sum(Wh==0)/np.prod(Wh.shape)*100,2),'%')
print('Genes with no contribution to output : ',np.sum(np.abs(np.sum(Wh,axis=0))==0) )
plt.figure(figsize=(6,6))
sb.heatmap(Wh, cmap="RdYlGn",center=0)
plt.show()

## GENE RANKING - APPROACH 1 - OUTPUT MATRIX STRENGTH
df_gene_info = rnaf.get_gene_conversion_info()
ls_gene_list = list(dict_MAX[0]['df_X_TPM'].index)
dict_GENES = {gene:{'Name':df_gene_info.loc[gene,'Name'] } for gene in ls_gene_list}

for i in range(len(ls_gene_list)):
    dict_GENES[ls_gene_list[i]]['Wh_val'] = np.mean(Wh[:,i])

# df_GENES = pd.DataFrame(dict_GENES).T
# df_GENES = df_GENES.sort_values(by = 'Wh_val',ascending = False)



## GENE RANKING - APPROACH 2 - Output Energy of the actual output (not scaled)
WhT = Wh.T
CURVE_NO = 0
Xps_i = dict_MAX[CURVE_NO]['df_X_TPM'].iloc[:,0:-1].to_numpy().T
Xfs_i = dict_MAX[CURVE_NO]['df_X_TPM'].iloc[:,1:].to_numpy().T
Y0 = dict_MAX[CURVE_NO]['Y0']
Yp = dict_MAX[CURVE_NO]['Y'].iloc[:,0:-1].to_numpy().T

d = oc.scale_data_using_existing_scaler_folder({'Xp':Xps_i,'Xf':Xfs_i,'Yp':Yp}, SYSTEM_NO)
XpTs = d['Xp']
XfTs = d['Xf']
YpTs = d['Yp']

YT_true = np.cumsum(oc.inverse_transform_Y(YpTs,SYSTEM_NO).reshape(-1))

for i in range(len(p['Xp'][0])):
    # Y_i = Y0 + np.cumsum(oc.inverse_transform_Y(np.matmul(Xps[:,i:i+1],WhT[i:i+1,:]), SYSTEM_NO).reshape(-1))
    Y_i = np.cumsum(oc.inverse_transform_Y(np.matmul(XpTs[:, i:i + 1], WhT[i:i + 1, :]), SYSTEM_NO).reshape(-1))
    print(i,' : ', np.sum(Y_i))
    dict_GENES[ls_gene_list[i]]['y_energy'] = np.sum(Y_i) / np.sum(YT_true) * 100

# [:,i:i+1]
# for i in range(len(p['Xp'][0])):
#     Y_is = np.matmul(p['Xp'][:,i:i+1],WhT[i:i+1,:])
#     Y_i = oc.inverse_transform_Y(Y_is, SYSTEM_NO)
#     dict_GENES[ls_gene_list[i]]['y_energy'] = np.sum(Y_i)/np.sum(Y_true)*100

df_GENES = pd.DataFrame(dict_GENES).T
# df_GENES = df_GENES.sort_values(by = 'y_energy',ascending = False)
# df_GENES.loc[:,['Name', 'y_energy']]

# rnaf.get_gene_Uniprot_DATA(ls_all_locus_tags=list(df_GENES.index)[0:10],search_columns='genes,comment(FUNCTION)')

## Method 3 - Sensitivity of the output to the states
Y_true = Y0 + np.cumsum(oc.inverse_transform_Y(YpTs,SYSTEM_NO).reshape(-1))
# Y_true = oc.inverse_transform_Y(YpTs,SYSTEM_NO).reshape(-1)
AT = A.T
dict_y = {}
for gene_no in range(len(XpTs[0])):
    XTi = copy.deepcopy(XpTs[0:1,:])
    XTi[-1:,gene_no] = 0
    for tp in range(1,len(XpTs)):
        XTi[-1:, gene_no] = 0
        XTi = np.concatenate([XTi,np.matmul(XTi[-1:,:],AT)],axis=0)

    YTi = Y0 + np.cumsum(oc.inverse_transform_Y(np.matmul(XTi,WhT),SYSTEM_NO).reshape(-1))
    # YTi = oc.inverse_transform_Y(np.matmul(XTi,WhT),SYSTEM_NO).reshape(-1)
    dict_y[gene_no] = YTi
    dict_GENES[ls_gene_list[gene_no]]['y_r2_sens'] = r2_score(Y_true,YTi)*100

df_GENES = pd.DataFrame(dict_GENES).T
df_GENES = df_GENES.sort_values(by = 'y_r2_sens',ascending = True)
df_GENES.loc[:,['Name', 'y_r2_sens']]


##
N_GENES = 10
ls_genes_MAX = list(dict_MAX[0]['df_X_TPM'].index)
ls_genes_reqd = list(df_GENES.index[0:N_GENES])
ls_reqd_gene_indices = [ls_genes_MAX.index(items) for items in ls_genes_reqd]
ls_gene_names = df_GENES.Name.to_list()[0:N_GENES]
for i in range(len(ls_gene_names)):
    if ls_gene_names[i] == 'DUF4223 domain-containing protein CDS':
        ls_gene_names[i] = 'DUF4223 dcp'
    elif ls_gene_names[i] == 'hypothetical protein CDS':
        ls_gene_names[i] =  ls_genes_reqd[i]+ ' hp'
    elif ls_gene_names[i] == 'membrane protein CDS':
        ls_gene_names[i] = ls_genes_reqd[i] + ' mp'
    else:
        ls_gene_names[i] = ls_gene_names[i][:-4]

FONTSIZE = 40
# Wh
plt.figure(figsize=(20,20))
a = sb.heatmap(Wh[:,ls_reqd_gene_indices], cmap="RdYlGn",center=0)
b, t = a.axes.get_ylim()  # discover the values for bottom and top
b += 0.5  # Add 0.5 to the bottom
t -= 0.5  # Subtract 0.5 from the top
a.axes.set_ylim(b, t)
cbar = a.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=FONTSIZE)
a.axes.set_xticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation=90)
ylabel_all = list(range(0,3*len(Wh),3))
ylab = ['' for y in ylabel_all]
for i in range(len(ylabel_all)):
    if np.mod(i,2) ==0:
        ylab[i] = ylabel_all[i]
a.axes.set_yticklabels(ylab,{'fontsize':FONTSIZE},rotation = 0)
plt.show()

# A
plt.figure(figsize=(20,20))
a = sb.heatmap(A[:,ls_reqd_gene_indices][ls_reqd_gene_indices,:], cmap="RdYlGn",center=0)
b, t = a.axes.get_ylim()  # discover the values for bottom and top
b += 0.5  # Add 0.5 to the bottom
t -= 0.5  # Subtract 0.5 from the top
a.axes.set_ylim(b, t)
cbar = a.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=FONTSIZE)
a.axes.set_xticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation=90)
a.axes.set_yticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation = 0)
plt.show()

## Plot growth of the required genes
plt.figure()
for i in range(len(ls_reqd_gene_indices)):
    # plt.plot(dict_MAX[0]['df_X_TPM'].iloc[ls_reqd_gene_indices[i],:],label = ls_gene_names[i])
    plt.plot(np.log(dict_MAX[0]['df_X_TPM'].iloc[ls_reqd_gene_indices[i], :]), label=ls_gene_names[i])
plt.legend()
plt.show()
##
DS = 2
FONTSIZE = 12
xv = 1 + np.arange(len(dict_y[0]))*3/60
plt.figure()
plt.plot(xv[::DS], Y_true[::DS],linewidth = 4,label='True curve')
for i in range(len(ls_reqd_gene_indices)):
    plt.plot(xv[::DS], dict_y[ls_reqd_gene_indices[i]][::DS],'.',linewidth=3,label = ls_gene_names[i])
# plt.xlabel('Time (hrs)')
# plt.ylabel('OD600')
plt.xticks(fontsize = FONTSIZE)
plt.yticks(fontsize = FONTSIZE)
plt.legend(fontsize = FONTSIZE,ncol = 1)
plt.show()

##
plt.figure()
plt.plot(Y_true,linewidth = 4)
for i in dict_y:
    plt.plot(dict_y[i])
plt.show()

##


[0,2,3,4,5,7,8]:#
# ## Method 4 - Modal analysis
# n_red = 6
# U,S,VT = np.linalg.svd(XpTs.T)
# Ur = U[:,0:n_red]
# A_tilde = np.matmul(Ur.T,np.matmul(A,Ur))
# Eri,Wri = np.linalg.eig(A_tilde)
# Er,Wr,_,__ = resolve_complex_right_eigenvalues(np.diag(Eri),Wri)
# sb.heatmap(np.matmul(Wh,np.matmul(Ur,Wr)), cmap="RdBu",center=0)
# plt.show()
#
#
# Zps = np.matmul(np.linalg.inv(Wr),np.matmul(Ur.T, XpTs.T))
# Zfs = np.matmul(np.linalg.inv(Wr),np.matmul(Ur.T, XfTs.T))
#
# ##
#
# curve = 0
# with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle', 'rb') as handle:
#     dict_DATA_ORIGINAL = pickle.load(handle)
#
##
# ls_all_genes = list(dict_DATA_ORIGINAL['MX'][0]['df_X_TPM'].index)
# ls_reqd_genes = list(df_GENES.index)[0:3]
# ls_select_gene_indices = [i for i in range(len(ls_reqd_genes)) if ls_all_genes[i] in ls_reqd_genes]
curve = 0
with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle', 'rb') as handle:
    dict_DATA_ORIGINAL = pickle.load(handle)
f,ax = plt.subplots(7,1,sharex=True,figsize=(30,14))
for time_pt in range(1,8):
    # for curve in range(16):
    # max_val = np.max(np.array(dict_DATA_ORIGINAL['MX'][curve]['df_X_TPM'].loc[:, time_pt]))
    max_val = np.max(np.array(dict_MAX[curve]['df_X_TPM'].loc[:, time_pt]))
    i = 0
    for items in ls_reqd_gene_indices:
        ax[time_pt - 1].plot([items, items],[0,max_val],'r',linewidth = 3)
        i = i+1
        ax[time_pt - 1].annotate(str(i),(items,max_val),fontsize=24)
    # ax[time_pt - 1].plot(np.array(dict_DATA_ORIGINAL['MX'][curve]['df_X_TPM'].loc[:, time_pt]))
    ax[time_pt - 1].plot(np.array(dict_MAX[curve]['df_X_TPM'].loc[:, time_pt]))

    # ax[time_pt - 1].plot(np.array(dict_DATA_max_denoised['MX'][curve]['df_X_TPM'].loc[:,time_pt]))
    # ax[time_pt - 1].plot(np.array(dict_MAX[curve]['df_X_TPM'].loc[:, time_pt]))
    # ax[time_pt - 1].set_xlim([0,500])
    # ax[time_pt - 1].set_ylim([0, 1000])
    ax[time_pt-1].set_title('Time Point : ' + str(time_pt),fontsize=24)
    # break
ax[-1].set_xlabel('Gene Locus Tag')
f.show()
#
# ##
# ls_reqd_genes = list(df_GENES.index)[0:20]
# df_A = pd.DataFrame(A,index = dict_MAX[0]['df_X_TPM'].index, columns= dict_MAX[0]['df_X_TPM'].index)
# print(df_A.loc[ls_reqd_genes,ls_reqd_genes])
# sb.heatmap(df_A.loc[ls_reqd_genes,ls_reqd_genes], cmap="RdBu",center=0)
# plt.show()
#
# # ls_gene_ind = [i for ]