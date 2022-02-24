import numpy as np
import itertools
import copy
import pandas as pd

def n_step_prediction_error(model, dict_temp, train_test_valid = 'train', multioutput = 'uniform'):
    proxy_cond = list(dict_temp['scaled'].keys())[0]
    proxy_rep = list(dict_temp['scaled'][proxy_cond].keys())[0]
    # Checking for a dictionary of two models in which case, we want n-step predictions of x and y
    if isinstance(model,dict):
        print('This is yet to be written')
        return
        r2 = n_step_prediction_error_XY(model, dict_temp, train_test_valid=train_test_valid, multioutput=multioutput)
    elif len(model.coef_.shape) ==2:
        if model.coef_.shape[0] == dict_temp['scaled'][proxy_cond][proxy_rep]['XfT'].shape[1]: # Check if model output dimension matches the shape of X
            r2 = n_step_prediction_error_X(model, dict_temp, train_test_valid=train_test_valid, multioutput=multioutput)
        elif model.coef_.shape[0] == dict_temp['scaled'][proxy_cond][proxy_rep]['YfT'].shape[1]: # Check if model output dimension matches the shape of Y
            r2 = n_step_prediction_error_Y(model, dict_temp, train_test_valid=train_test_valid)
            # Since there is a single output, we only use uniform error
        else:
            print('[WARNING]: Invalid model')
            r2 = np.nan
    else:
        try:
        # Output model has a dimension of 1
            r2 = n_step_prediction_error_Y(model, dict_temp, train_test_valid=train_test_valid)
        except:
            print('[WARNING]: Invalid model')
            r2 = np.nan
    return r2

def n_step_prediction_error_X(model, dict_temp, train_test_valid = 'train', multioutput = 'uniform'):
    # n-step prediction error in training and validation datasets
    XfTs_SSE = XfTs_SST = XfTs_wi = 0
    ls_conditions = list(dict_temp['scaled'].keys())
    for cond in ls_conditions:
        X0Ts_i = XfTs_i = UpTs_i = np.array([])
        # Generate all the predictors
        for curve in dict_temp[train_test_valid]['indices']:
            try:
                X0Ts_i = np.concatenate([X0Ts_i, dict_temp['scaled'][cond][curve]['XpT'][0:1, :].reshape(1, -1, 1)],
                                        axis=2)
            except:
                X0Ts_i = dict_temp['scaled'][cond][curve]['XpT'][0:1, :].reshape(1, -1, 1)
            try:
                XfTs_i = np.concatenate([XfTs_i, dict_temp['scaled'][cond][curve]['XfT'][:, :, np.newaxis]], axis=2)
            except:
                XfTs_i = dict_temp['scaled'][cond][curve]['XfT'][:, :, np.newaxis]
            try:
                UpTs_i = np.concatenate([UpTs_i, dict_temp['scaled'][cond][curve]['UpT'][:, :, np.newaxis]], axis=2)
            except:
                UpTs_i = dict_temp['scaled'][cond][curve]['UpT'][:, :, np.newaxis]
        XT_i = copy.deepcopy(X0Ts_i)
        for time_i in range(XfTs_i.shape[0]):
            XT_i = np.concatenate([XT_i,
                                   model.predict(np.concatenate([XT_i[-1, :, :].T, UpTs_i[time_i, :, :].T], axis=1)).T[
                                   np.newaxis, :, :]], axis=0)
        XT_i = XT_i[1:, :, :]
        if multioutput == 'variance_weighted':
            try:
                XfTs_SSE = np.concatenate([XfTs_SSE, np.square(XT_i - XfTs_i).sum(axis=2).sum(axis=0)[:, np.newaxis]],
                                          axis=1)
                XfTs_SST = np.concatenate([XfTs_SST, np.square(XfTs_i- XfTs_i.mean(axis=(0, 2))[np.newaxis, :, np.newaxis]).sum(axis=2).sum(axis=0)[:, np.newaxis]], axis=1)
                XfTs_wi = np.concatenate([XfTs_wi, XfTs_i.var(axis=(0,2))[:,np.newaxis]],axis=1)
            except:
                XfTs_SSE = np.square(XT_i - XfTs_i).sum(axis=(0, 2))[:, np.newaxis]
                XfTs_SST = np.square(XfTs_i - XfTs_i.mean(axis=(0, 2))[np.newaxis, :, np.newaxis]).sum(axis=2).sum(
                    axis=0)[:, np.newaxis]
                XfTs_wi = XfTs_i.var(axis=(0, 2))[:, np.newaxis]
        elif multioutput == 'uniform':
            XfTs_SSE = XfTs_SSE + np.square(XT_i - XfTs_i).sum()
            XfTs_SST = XfTs_SST + np.square(XfTs_i - XfTs_i.mean(axis=(0, 2))[np.newaxis, :, np.newaxis]).sum()
    if multioutput == 'variance_weighted':
        r2 = 1 - np.sum((XfTs_SSE/XfTs_SST)*(XfTs_wi/XfTs_wi.sum()))
    elif multioutput == 'uniform':
        r2 = 1 - XfTs_SSE/XfTs_SST
    return r2

def n_step_prediction_error_Y(model, dict_temp, train_test_valid = 'train'):
    # prediction error in training and validation datasets
    XTs = YTs = 0
    ls_conditions = list(dict_temp['scaled'].keys())
    # Generate all the predictors
    for cond,curve in itertools.product(ls_conditions,dict_temp[train_test_valid]['indices']):
        XTs_i = np.concatenate([dict_temp['scaled'][cond][curve]['XpT'][0:1, :], dict_temp['scaled'][cond][curve]['XfT']], axis=0)
        YTs_i = np.concatenate([dict_temp['scaled'][cond][curve]['YpT'][0:1, :], dict_temp['scaled'][cond][curve]['YfT']], axis=0)
        try:
            XTs = np.concatenate([XTs, XTs_i], axis=0)
            YTs = np.concatenate([YTs, YTs_i], axis=0)
        except:
            XTs = XTs_i
            YTs = YTs_i
    YTs_est = model.predict(XTs)
    YTs_est = YTs_est.reshape(-1)
    YTs = YTs.reshape(-1)
    SSE = np.square(YTs - YTs_est).sum()
    SST = np.square(YTs - YTs.mean()).sum()
    r2 = 1 - SSE/SST
    return r2

def n_step_prediction_error_XY(model, dict_temp, train_test_valid = 'train', multioutput = 'uniform'):
    # n-step prediction error in training and validation datasets
    XfTs_SSE = XfTs_SST = XfTs_wi = 0
    ls_conditions = list(dict_temp['scaled'].keys())
    for cond in ls_conditions:
        X0Ts_i = XfTs_i = UpTs_i = np.array([])
        # Generate all the predictors
        for curve in dict_temp[train_test_valid]['indices']:
            try:
                X0Ts_i = np.concatenate([X0Ts_i, dict_temp['scaled'][cond][curve]['XpT'][0:1, :].reshape(1, -1, 1)],
                                        axis=2)
            except:
                X0Ts_i = dict_temp['scaled'][cond][curve]['XpT'][0:1, :].reshape(1, -1, 1)
            try:
                XfTs_i = np.concatenate([XfTs_i, dict_temp['scaled'][cond][curve]['XfT'][:, :, np.newaxis]], axis=2)
            except:
                XfTs_i = dict_temp['scaled'][cond][curve]['XfT'][:, :, np.newaxis]
            try:
                UpTs_i = np.concatenate([UpTs_i, dict_temp['scaled'][cond][curve]['UpT'][:, :, np.newaxis]], axis=2)
            except:
                UpTs_i = dict_temp['scaled'][cond][curve]['UpT'][:, :, np.newaxis]
        XT_i = copy.deepcopy(X0Ts_i)
        for time_i in range(XfTs_i.shape[0]):
            XT_i = np.concatenate([XT_i,
                                   model.predict(np.concatenate([XT_i[-1, :, :].T, UpTs_i[time_i, :, :].T], axis=1)).T[
                                   np.newaxis, :, :]], axis=0)
        # Computation for Y


        XT_i = XT_i[1:, :, :]

        if multioutput == 'variance_weighted':
            try:
                XfTs_SSE = np.concatenate([XfTs_SSE, np.square(XT_i - XfTs_i).sum(axis=2).sum(axis=0)[:, np.newaxis]],
                                          axis=1)
                XfTs_SST = np.concatenate([XfTs_SST, np.square(XfTs_i- XfTs_i.mean(axis=(0, 2))[np.newaxis, :, np.newaxis]).sum(axis=2).sum(axis=0)[:, np.newaxis]], axis=1)
                XfTs_wi = np.concatenate([XfTs_wi, XfTs_i.var(axis=(0,2))[:,np.newaxis]],axis=1)
            except:
                XfTs_SSE = np.square(XT_i - XfTs_i).sum(axis=(0, 2))[:, np.newaxis]
                XfTs_SST = np.square(XfTs_i - XfTs_i.mean(axis=(0, 2))[np.newaxis, :, np.newaxis]).sum(axis=2).sum(
                    axis=0)[:, np.newaxis]
                XfTs_wi = XfTs_i.var(axis=(0, 2))[:, np.newaxis]
        elif multioutput == 'uniform':
            XfTs_SSE = XfTs_SSE + np.square(XT_i - XfTs_i).sum()
            XfTs_SST = XfTs_SST + np.square(XfTs_i - XfTs_i.mean(axis=(0, 2))[np.newaxis, :, np.newaxis]).sum()
    if multioutput == 'variance_weighted':
        r2 = 1 - np.sum((XfTs_SSE/XfTs_SST)*(XfTs_wi/XfTs_wi.sum()))
    elif multioutput == 'uniform':
        r2 = 1 - XfTs_SSE/XfTs_SST
    return r2

# TODO - Move it to the appropriate folder
def get_RbTnSeq_curves(ls_gene_locus_tags, condition = 'MAX', with_respect_to_time0=True):
    # Take care of start time and end time
    df_i = pd.read_csv('DATA/RNA_1_Pput_R2A_Cas_Glu/RbTnSeq/RbTnSeq_' + condition + '.csv', index_col=0)
    if with_respect_to_time0:
        df_i = df_i.loc[:,df_i.loc['start_time',:] == 0]
    else:
        ls_reqd_cols = []
        for i in df_i.columns:
            if df_i.loc['end_time',i] == df_i.loc['start_time',i] + 1:
                ls_reqd_cols.append(i)
            elif df_i.loc['end_time',i] == 0:
                ls_reqd_cols.append(i)
        df_i = df_i.loc[:, ls_reqd_cols]
        df_i.columns = df_i.loc['end_time',:]
    return df_i.loc[ls_gene_locus_tags,:]


# TODO - come back to the DMD model later. Make this in the same framework as the LinearRegression or Lasso or RidgeRegression model
# TODO - eventually make deepDMD the same architecture
# class DMD():
#     def __init__(self):
#         self.coef_ = np.nan
#         self.n_features_in = 0
#         self.n_features_out = 0
#         return
#
#     def fit(self, XT_train, YT_train, XT_valid, YT_valid, ls_n_principal_components= -1):
#         self.n_features_in = XT_train.shape[0]
#         self.n_features_out = YT_train.shape[0]
#
#         U, s, VT = np.linalg.svd(XT_train.T)
#         for random_index in range(10):
#             BREAK_LOOP = False
#             try:
#                 if isinstance(ls_n_principal_components,list) and np.all(np.array(ls_n_principal_components)>=0):
#                     # TODO check if it is a valid list
#                     ls_nPC = np.array(ls_n_principal_components).reshape(-1)
#             except:
#                 # Implement smart selection
#                 print('Implementing smart loop iteration number ', random_index+1)
#             # Implementing the pseudo-inverse
#             for r in ls_nPC:
#                 Ur = U[:, 0:r]
#                 UrT = np.conj(Ur.T)
#                 Sr = np.diag(s[0:r])
#                 V = np.conj(VT.T)
#                 Vr = V[:, 0:r]
#                 Ahat = XfTs.T @ Vr @ np.linalg.inv(Sr) @ UrT
#
#             if BREAK_LOOP:
#                 break
#
#         self.coef_ = Ahat
#         return
#
#     def predict(self, XT, one_step=True):
#         # Checking input dimensions
#         if not XT.shape[0] == self.n_features_in:
#             print('The given input dimension is ', XT.shape[0], 'in the place of ', self.n_features_in)
#             return
#
#         return