##
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle
import tensorflow as tf
import itertools
import ocdeepdmd_simulation_examples_helper_functions as oc
import copy
import random
from scipy.stats import pearsonr as corr
import direct_nn_helper_functions as dn
from sklearn.metrics import r2_score

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.37109375, 0.67578125, 0.38671875],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.8125   , 0.609375 , 0.0703125], #[0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765],
          '#B724AE','#2C9572','#0055FF','#A6A948','#AC8A00'];
colors = np.asarray(colors);  # defines a color palette
plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22

SYS_NO = 91
# RUN_DIRECT_DEEPDMD_SUBOPT = 6
RUN_DIRECT_DEEPDMD = 72 #3 # 3,6,32,   13
RUN_SEQ_DEEPDMD = 194 #194 #156#41
RUN_DEEPDMD_X = 4 # 91
RUN_NN = 4



sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYS_NO)
run_folder_name_DEEPDMD = sys_folder_name + '/Sequential/RUN_' + str(RUN_DEEPDMD_X)
run_folder_name_SEQ_ocDEEPDMD = sys_folder_name + '/Sequential/RUN_' + str(RUN_SEQ_DEEPDMD)
run_folder_name_DIR_ocDEEPDMD = sys_folder_name + '/deepDMD/RUN_' + str(RUN_DIRECT_DEEPDMD)
# run_folder_name_DIR_ocDEEPDMD_SUBOPT = sys_folder_name + '/deepDMD/RUN_' + str(RUN_DIRECT_DEEPDMD_SUBOPT)
run_folder_name_NN = sys_folder_name + '/Direct_nn/RUN_' + str(RUN_NN)

with open(sys_folder_name + '/System_' + str(SYS_NO) + '_SimulatedData.pickle', 'rb') as handle:
    dict_data = pickle.load(handle)
with open(sys_folder_name + '/System_' + str(SYS_NO) + '_ocDeepDMDdata.pickle', 'rb') as handle:
    dict_oc_data = pickle.load(handle)
Ntrain = round(len(dict_oc_data['Xp'])/2)
for items in dict_oc_data:
    dict_oc_data[items] = dict_oc_data[items][0:Ntrain]
with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'rb') as handle:
    d_SEQ = pickle.load(handle)[RUN_SEQ_DEEPDMD]
with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'rb') as handle:
    d_DDMD_X = pickle.load(handle)[RUN_DEEPDMD_X]
with open(sys_folder_name + '/dict_predictions_deepDMD.pickle', 'rb') as handle:
    d_DDMD = pickle.load(handle)[RUN_DIRECT_DEEPDMD]
with open(sys_folder_name + '/dict_predictions_Direct_nn.pickle', 'rb') as handle:
    d_NN = pickle.load(handle)[RUN_NN]




ls_steps = list(range(1,50,1))
ls_curves = list(range(200, 300)) # test curves
Senergy_THRESHOLD = 99.99
REDUCED_MODES = False
RIGHT_EIGEN_VECTORS = True
CURVE_NO = 260 # random.choice(ls_curves)
print(CURVE_NO)
def phase_portrait_data():
    # System Parameters
    gamma_A = 1.
    gamma_B = 0.5
    delta_A = 1.
    delta_B = 1.
    alpha_A0 = 0.04
    alpha_B0 = 0.004
    alpha_A = 50.
    alpha_B = 30.
    K_A = 1.
    K_B = 1.5
    kappa_A = 1.
    kappa_B = 1.
    n = 2.
    m = 2.
    k_3n = 3.
    k_3d = 1.08

    sys_params_arc2s = (gamma_A, gamma_B, delta_A, delta_B, alpha_A0, alpha_B0, alpha_A, alpha_B, K_A, K_B, kappa_A, kappa_B, n, m)
    Ts = 0.1
    t_end = 40
    # Simulation Parameters
    dict_phase_data = {}
    X0 = np.empty(shape=(0, 2))
    t = np.arange(0, t_end, Ts)

    # Phase Space Data
    dict_phase_data = {}
    X0 = np.empty(shape=(0, 2))
    i = 0
    for x1, x2 in itertools.product(list(np.arange(1, 30., 3.5)), list(np.arange(1, 60., 8))):
        dict_phase_data[i] = {}
        x0_curr = np.array([x1, x2])
        X0 = np.concatenate([X0, np.array([[x1, x2]])], axis=0)
        dict_phase_data[i]['X'] = oc.odeint(oc.activator_repressor_clock_2states, x0_curr, t, args=sys_params_arc2s)
        dict_phase_data[i]['Y'] = k_3n * dict_phase_data[i]['X'][:, 1:2] / (k_3d + dict_phase_data[i]['X'][:, 3:4])
        i = i + 1
    return dict_phase_data
def get_dict_param(run_folder_name_curr,SYS_NO,sess):
    dict_p = {}
    saver = tf.compat.v1.train.import_meta_graph(run_folder_name_curr + '/System_' + str(SYS_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(run_folder_name_curr))
    try:
        psixpT = tf.get_collection('psixpT')[0]
        psixfT = tf.get_collection('psixfT')[0]
        xpT_feed = tf.get_collection('xpT_feed')[0]
        xfT_feed = tf.get_collection('xfT_feed')[0]
        KxT = tf.get_collection('KxT')[0]
        KxT_num = sess.run(KxT)
        dict_p['psixpT'] = psixpT
        dict_p['psixfT'] = psixfT
        dict_p['xpT_feed'] = xpT_feed
        dict_p['xfT_feed'] = xfT_feed
        dict_p['KxT_num'] = KxT_num
    except:
        print('State info not found')
    try:
        ypT_feed = tf.get_collection('ypT_feed')[0]
        yfT_feed = tf.get_collection('yfT_feed')[0]
        dict_p['ypT_feed'] = ypT_feed
        dict_p['yfT_feed'] = yfT_feed
        WhT = tf.get_collection('WhT')[0];
        WhT_num = sess.run(WhT)
        dict_p['WhT_num'] = WhT_num
    except:
        print('No output info found')
    return dict_p
def r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params_curr):
    n_states = len(dict_data[list(dict_data.keys())[0]]['X'][0])
    # n_outputs = len(dict_data[list(dict_data.keys())[0]]['Y'][0])
    # dict_rmse = {}
    dict_r2 = {}
    for CURVE_NO in ls_curves:
        # dict_rmse[CURVE_NO] = {}
        dict_r2[CURVE_NO] = {}
        dict_DATA_i = oc.scale_data_using_existing_scaler_folder(dict_data[CURVE_NO], SYS_NO)
        X_scaled = dict_DATA_i['X']
        Y_scaled = dict_DATA_i['Y']
        psiX = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: X_scaled})
        for i in ls_steps:  # iterating through each step prediction
            np_psiX_true = psiX[i:, :]
            np_psiX_pred = np.matmul(psiX[:-i, :],np.linalg.matrix_power(dict_params_curr['KxT_num'], i))  # i step prediction at each datapoint
            Y_pred = np.matmul(np_psiX_pred, dict_params_curr['WhT_num'])
            Y_true = Y_scaled[i:, :]

            X = oc.inverse_transform_X(np_psiX_true[:,0:n_states], SYS_NO)
            Y = oc.inverse_transform_Y(Y_true, SYS_NO)
            Xhat = oc.inverse_transform_X(np_psiX_pred[:,0:n_states], SYS_NO)
            Yhat = oc.inverse_transform_Y(Y_pred, SYS_NO)
            SSE = np.sum(np.square(X - Xhat)) + np.sum(np.square(Y - Yhat))
            SST = np.sum(np.square(X)) + np.sum(np.square(Y))
            dict_r2[CURVE_NO][i] = np.max([0, 1 - (SSE / SST)]) * 100
            # dict_rmse[CURVE_NO][i] = np.sqrt(np.mean(np.square(np_psiX_true - np_psiX_pred)))
            # dict_r2[CURVE_NO][i] = np.max([0, (1 - (np.sum(np.square(np_psiX_true - np_psiX_pred)) + np.sum(np.square(Y_true - Y_pred))) / (np.sum(np.square(np_psiX_true)) + np.sum(np.square(Y_true)))) * 100])
    df_r2 = pd.DataFrame(dict_r2)
    print(df_r2)
    CHECK_VAL = df_r2.iloc[-1, :].min()
    OPT_CURVE_NO = 0
    for i in ls_curves:
        if df_r2.loc[df_r2.index[-1], i] == CHECK_VAL:
            OPT_CURVE_NO = i
            break
    return df_r2, OPT_CURVE_NO
def resolve_complex_right_eigenvalues(E, W):
    eval = np.diag(E)
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

def modal_analysis(dict_oc_data,dict_data_curr,dict_params_curr,REDUCED_MODES,Senergy_THRESHOLD,RIGHT_EIGEN_VECTORS = True,SHOW_PCA_X=True):
    #Eig func evolution
    psiX = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: dict_oc_data['Xp']}).T
    # Phi0 = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: dict_data_curr['X'][0:1]})
    Phi0 = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: dict_data_curr['X']})
    # For Eigenfunctions and Observables
    n_observables = len(psiX)
    sampling_resolution = 0.1
    x1 = np.arange(-5, 5 + sampling_resolution, sampling_resolution)
    x2 = np.arange(-6, 6 + sampling_resolution, sampling_resolution)
    X1, X2 = np.meshgrid(x1, x2)
    H = np.zeros(shape=X2.shape)
    if SHOW_PCA_X:
        _, s, _T = np.linalg.svd(dict_oc_data['Xp'])
        plt.stem(np.arange(len(s)), (np.cumsum(s ** 2) / np.sum(s ** 2)) * 100)
        plt.plot([0, len(s) - 1], [100, 100])
        plt.title('just X')
        plt.show()
        plt.figure()
        _, s, _T = np.linalg.svd(psiX)
        plt.stem(np.arange(len(s)), (np.cumsum(s ** 2) / np.sum(s ** 2)) * 100)
        plt.plot([0, len(s) - 1], [100, 100])
        plt.title('psiX')
        plt.show()
    K = dict_params_curr['KxT_num'].T
    if REDUCED_MODES:
        # Minimal POD modes of psiX
        U, S, VT = np.linalg.svd(psiX)
        Senergy = np.cumsum(S ** 2) / np.sum(S ** 2) * 100
        for i in range(len(S)):
            if Senergy[i] > Senergy_THRESHOLD:
                nPC = i + 1
                break
        print('Optimal POD modes chosen : ', nPC)
        Ur = U[:, 0:nPC]
        Kred = np.matmul(np.matmul(Ur.T, K), Ur)
        if RIGHT_EIGEN_VECTORS:
            eval, W = np.linalg.eig(Kred)
            E = np.diag(eval)
            E,W, comp_modes, comp_modes_conj = resolve_complex_right_eigenvalues(E,W)
            Winv = np.linalg.inv(W)
            Phi_t = np.matmul(E, np.matmul(np.matmul(Winv, Ur.T), Phi0.T))
            # for i in range(1,len(dict_data_curr['X'])):
            #     Phi = np.concatenate([Phi,np.matmul(E,Phi[:,-1:])],axis=1)
            koop_modes = W
            # For Eigenfunctions and Observables
            PHI = np.zeros(shape=(X1.shape[0], X1.shape[1], nPC))
            PSI = np.zeros(shape=(X1.shape[0], X1.shape[1], n_observables))
            for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
                x1_i = X1[i, j]
                x2_i = X2[i, j]
                psiXT_i = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: np.array([[x1_i, x2_i]])})
                PHI[i, j, :] = np.matmul(np.matmul(Winv, Ur.T), psiXT_i.T).reshape((1, 1, -1))
                PSI[i, j, :] = psiXT_i.reshape((1, 1, -1))
                H[i, j] = 0
        else:
            #TODO - Do what happens when left eigenvectors are inserted here
            print('Meh')
    else:
        if RIGHT_EIGEN_VECTORS:
            eval, W = np.linalg.eig(K)
            E = np.diag(eval)
            E, W, comp_modes, comp_modes_conj = resolve_complex_right_eigenvalues(E, W)
            Winv = np.linalg.inv(W)
            Phi_t = np.matmul(E, np.matmul(Winv,Phi0.T))
            # for i in range(1, len(dict_data_curr['X'])):
            #     Phi = np.concatenate([Phi, np.matmul(E, Phi[:, -1:])], axis=1)
            koop_modes = W
            PHI = np.zeros(shape=(X1.shape[0], X1.shape[1], n_observables))
            PSI = np.zeros(shape=(X1.shape[0], X1.shape[1], n_observables))
            for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
                x1_i = X1[i, j]
                x2_i = X2[i, j]
                psiXT_i = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: np.array([[x1_i, x2_i]])})
                PHI[i, j, :] = np.matmul(Winv, psiXT_i.T).reshape((1, 1, -1))
                PSI[i, j, :] = psiXT_i.reshape((1, 1, -1))
                try:
                    # H[i, j] = oc.inverse_transform_Y(np.matmul(psiXT_i,dict_params_curr['WhT_num']), 91).reshape(-1)
                    H[i, j] = np.matmul(psiXT_i, dict_params_curr['WhT_num']).reshape(-1)
                except:
                    H[i,j] =0
        else:
            #TODO - Do what happens when left eigenvectors are inserted here
            print('Meh')
    return PHI,PSI,Phi_t, koop_modes, comp_modes, comp_modes_conj, X1, X2, eval, H

def r2_n_step_prediction_accuracy2(ls_steps,ls_curves,dict_data,dict_params_curr):
    n_states = len(dict_data[list(dict_data.keys())[0]]['X'][0])
    n_outputs = len(dict_data[list(dict_data.keys())[0]]['Y'][0])
    dict_X = {}
    dict_X_pred = {}
    dict_Y = {}
    dict_Y_pred = {}
    for step in ls_steps:
        dict_X[step] = np.empty(shape=(0, n_states))
        dict_X_pred[step] = np.empty(shape=(0, n_states))
        dict_Y[step] = np.empty(shape=(0, n_outputs))
        dict_Y_pred[step] = np.empty(shape=(0, n_outputs))
    for CURVE_NO in ls_curves:
        dict_DATA_i = oc.scale_data_using_existing_scaler_folder(dict_data[CURVE_NO], SYS_NO)
        X_scaled = dict_DATA_i['X']
        Y_scaled = dict_DATA_i['Y']
        for i in range(len(X_scaled) - np.max(ls_steps) - 2):
            psi_xi = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: X_scaled[i:i + 1]})
            for step in range(1,np.max(ls_steps)+1):
                psi_xi = np.matmul(psi_xi,dict_params_curr['KxT_num'])
                if step in ls_steps:
                    dict_X_pred[step] = np.concatenate([dict_X_pred[step],psi_xi[:,0:n_states]],axis=0)
                    dict_X[step] = np.concatenate([dict_X[step], X_scaled[i+step:i+step+1] ], axis=0)
                    dict_Y_pred[step] = np.concatenate([dict_Y_pred[step],np.matmul(psi_xi,dict_params_curr['WhT_num'])],axis=0)
                    dict_Y[step] = np.concatenate([dict_Y[step], Y_scaled[i + step:i + step + 1]],axis=0)
    dict_r2 = {}
    for step in ls_steps:
        # Compute the r^2
        X = oc.inverse_transform_X(dict_X[step], SYS_NO)
        Y = oc.inverse_transform_Y(dict_Y[step], SYS_NO)
        Xhat = oc.inverse_transform_X(dict_X_pred[step], SYS_NO)
        Yhat = oc.inverse_transform_Y(dict_Y_pred[step], SYS_NO)
        SSE = np.sum(np.square(X - Xhat)) + np.sum(np.square(Y - Yhat))
        SST = np.sum(np.square(X)) + np.sum(np.square(Y))
        dict_r2[step] = [np.max([0, 1 - (SSE / SST)]) * 100]
    df_r2 = pd.DataFrame(dict_r2)
    print(df_r2)
    return df_r2

##
dict_phase_data = phase_portrait_data()


dict_params = {}
sess1 = tf.InteractiveSession()
dict_params['Seq'] = get_dict_param(run_folder_name_SEQ_ocDEEPDMD,SYS_NO,sess1)
df_r2_SEQ = r2_n_step_prediction_accuracy2(ls_steps,ls_curves,dict_data,dict_params['Seq'])
# _, CURVE_NO = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Seq'])
PHI_SEQ,PSI_SEQ, Phi_t_SEQ,koop_modes_SEQ, comp_modes_SEQ, comp_modes_conj_SEQ,X1_SEQ,X2_SEQ, E_SEQ,H_SEQ = modal_analysis(dict_oc_data,dict_data[CURVE_NO],dict_params['Seq'],REDUCED_MODES = False,Senergy_THRESHOLD = 99.99,RIGHT_EIGEN_VECTORS=True,SHOW_PCA_X = False)
tf.reset_default_graph()
sess1.close()

sess2 = tf.InteractiveSession()
dict_params['Deep'] = get_dict_param(run_folder_name_DIR_ocDEEPDMD,SYS_NO,sess2)
df_r2_DEEPDMD = r2_n_step_prediction_accuracy2(ls_steps,ls_curves,dict_data,dict_params['Deep'])
# df_r2_DEEPDMD, _ = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Deep'])
PHI_DEEPDMD,PSI_DEEPDMD,Phi_t_DEEPDMD,koop_modes_DEEPDMD, comp_modes_DEEPDMD, comp_modes_conj_DEEPDMD,X1_DEEPDMD, X2_DEEPDMD, E_DEEPDMD, H_DEEPDMD = modal_analysis(dict_oc_data,dict_data[CURVE_NO],dict_params['Deep'],REDUCED_MODES = False,Senergy_THRESHOLD = 99.99,RIGHT_EIGEN_VECTORS=True,SHOW_PCA_X = False)
tf.reset_default_graph()
sess2.close()

sess3 = tf.InteractiveSession()
dict_params['Deep_X'] = get_dict_param(run_folder_name_DEEPDMD,SYS_NO,sess3)
# df_r2_DEEPDMD = r2_n_step_prediction_accuracy2(ls_steps,ls_curves,dict_data,dict_params['Deep_X'])
# df_r2_DEEPDMD, _ = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Deep'])
PHI_DEEP_X,PSI_DEEP_X,Phi_t_DEEP_X,koop_modes_DEEP_X, comp_modes_DEEP_X, comp_modes_conj_DEEP_X,X1_DEEP_X, X2_DEEP_X, E_DEEP_X, _ = modal_analysis(dict_oc_data,dict_data[CURVE_NO],dict_params['Deep_X'],REDUCED_MODES = False,Senergy_THRESHOLD = 99.99,RIGHT_EIGEN_VECTORS=True,SHOW_PCA_X = False)
tf.reset_default_graph()
sess3.close()

## Correlation between the eigenfunctions
import scipy

PHI_1_uf = copy.deepcopy(PHI_SEQ)
PHI_2_uf = copy.deepcopy(PHI_DEEPDMD)
ls_comp_modes_conj_1 = comp_modes_conj_SEQ
ls_comp_modes_conj_2 = comp_modes_conj_DEEPDMD
ls1 = list(set(list(range(PHI_1_uf.shape[2]))) - set(ls_comp_modes_conj_1))
ls2 = list(set(list(range(PHI_2_uf.shape[2]))) - set(ls_comp_modes_conj_2))
PHI_1 = PHI_1_uf[:,:,ls1]
PHI_2 = PHI_2_uf[:,:,ls2]
PHI_CORR = np.zeros(shape=(PHI_1.shape[2],PHI_2.shape[2]))
for i,j in itertools.product(range(PHI_1.shape[2]), range(PHI_2.shape[2])):
    PHI_CORR[i,j] = scipy.stats.pearsonr(PHI_1[:,:,i].reshape(-1)/np.max(PHI_1[:,:,i].reshape(-1)),PHI_2[:,:,j].reshape(-1)/np.max(PHI_2[:,:,j].reshape(-1)))[0]
PHI_CORR = np.nan_to_num(PHI_CORR,0)
PHI_CORR[-1,-1] = 1
PHI_CORR
##
# ## Figure 1
#
# FONT_SIZE = 14
# DOWNSAMPLE = 4
# LINE_WIDTH_c_d = 3
# TRUTH_MARKER_SIZE = 10
# TICK_FONT_SIZE = 9
# HEADER_SIZE = 21
# plt.figure(figsize=(15,10))
# plt.rcParams["axes.edgecolor"] = "black"
# plt.rcParams["axes.linewidth"] = 1
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["mathtext.fontset"] = 'cm'
#
# plt.subplot2grid((10,16), (0,0), colspan=5, rowspan=4)
# alpha = 1.0
# epsilon = alpha - 0.01
# arrow_length = 1.2
# ls_pts = list(range(0,1))
# for i in list(dict_phase_data.keys())[0:]:
#     for j in ls_pts:
#         if np.abs(dict_phase_data[i]['X'][j, 0]) > 1 or j==0:
#             plt.plot(dict_phase_data[i]['X'][j, 0], dict_phase_data[i]['X'][j, 1], 'o',color='salmon',fillstyle='none',markersize=5)
#     plt.plot(dict_phase_data[i]['X'][:, 0], dict_phase_data[i]['X'][:, 1], color='tab:blue',linewidth=0.3)
#     if np.mod(i,1)==0:
#         for j in ls_pts:
#             dist = np.sqrt((dict_phase_data[i]['X'][j, 0] - dict_phase_data[i]['X'][j + 1, 0]) ** 2 + (dict_phase_data[i]['X'][j, 1] - dict_phase_data[i]['X'][j + 1, 1]) ** 2)
#             x = dict_phase_data[i]['X'][j, 0]
#             y = dict_phase_data[i]['X'][j, 1]
#             dx = (dict_phase_data[i]['X'][j + 1, 0] - dict_phase_data[i]['X'][j, 0]) * arrow_length
#             dy = (dict_phase_data[i]['X'][j + 1, 1] - dict_phase_data[i]['X'][j, 1]) * arrow_length
#             # print(x,' ',y,' ',dist)
#             if dist<0.1:
#                 plt.arrow(x,y,dx,dy,head_width = 0.02,head_length=0.03,alpha=1,color='tab:green')
#             else:
#                 plt.arrow(x, y, dx, dy, head_width=1., head_length=0.9, alpha=1, color='tab:green')
# plt.xlabel('$x_1$',fontsize = FONT_SIZE)
# plt.ylabel('$x_2$',fontsize = FONT_SIZE)
# plt.plot(dict_phase_data[0]['X'][200:,0],dict_phase_data[0]['X'][200:,1],color='tab:red',markersize=10)
# plt.plot(dict_phase_data[5]['X'][200:,0],dict_phase_data[5]['X'][200:,1],color='tab:red',markersize=10)
# plt.plot(dict_phase_data[10]['X'][200:,0],dict_phase_data[10]['X'][200:,1],color='tab:red',markersize=10)
# # plt.xlim([0,0.6])
# # plt.ylim([0,1.65])
# plt.xlim([-0.5,30])
# plt.ylim([-0.5,60])
# plt.xticks(fontsize = TICK_FONT_SIZE)
# plt.yticks(fontsize = TICK_FONT_SIZE)
# plt.title('(a)',fontsize = HEADER_SIZE,loc='left')
#
#
#
# plt.subplot2grid((10,16), (5,0), colspan=4, rowspan=2)
# n_states = d_SEQ[CURVE_NO]['X'].shape[1]
# n_outputs = d_SEQ[CURVE_NO]['Y'].shape[1]
# pl_max = 0
# pl_min = 0
# for i in range(n_states):
#     x_scale = 10**np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['X'][:,i]))))
#     l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1) + (r'$[\times 10^{{{}}}]$').format(np.int(np.log10(x_scale))))
#     plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['X']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['X'][0::DOWNSAMPLE,i]/x_scale,'.',color = colors[i],markersize = TRUTH_MARKER_SIZE)
#     plt.plot(d_SEQ[CURVE_NO]['X_est_one_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
#     plt.plot(d_DDMD[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='solid', color=colors[i])
#     plt.plot(d_HAM[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
#     pl_max = np.max([pl_max,np.max(d_SEQ[CURVE_NO]['X'][:,i]/x_scale)])
#     pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['X'][:, i] / x_scale)])
# for i in range(n_outputs):
#     y_scale = 10 ** np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['Y'][:, i]))))
#     plt.plot([], color=colors[n_states+i], label=('$y_{}$').format(i + 1) + (r'$[\times 10^{{{}}}]$').format(np.int(np.log10(y_scale))))
#     plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['Y']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['Y'][0::DOWNSAMPLE,i]/y_scale, '.',color = colors[n_states+i],markersize = TRUTH_MARKER_SIZE)
#     plt.plot(d_SEQ[CURVE_NO]['Y_est_one_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
#     plt.plot(d_DDMD[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='solid', color=colors[n_states+i])
#     plt.plot(d_HAM[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='dashdot', color=colors[n_states+i])
#     pl_max = np.max([pl_max, np.max(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
#     pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
# l1 = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =3)
# plt.gca().add_artist(l1)
# # l1 = plt.legend(loc='upper center',fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =3)
# # plt.xlabel('Time Index(k)',fontsize = FONT_SIZE)
# # plt.ylabel('x,y [1 -step]',fontsize = FONT_SIZE)
# plt.text(32,2,'[1-step]',fontsize = FONT_SIZE)
# plt.ylim([pl_min-0.1,pl_max+0.1])
# plt.xticks([])
# plt.yticks(fontsize = TICK_FONT_SIZE)
# plt.xlim([-0.5,100.5])
# plt.title('(b)',fontsize = HEADER_SIZE,loc='left')
# plt.subplot2grid((10,16), (8,0), colspan=4, rowspan=2)
# pl_max = 0
# pl_min = 0
# for i in range(n_states):
#     x_scale = 10**np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['X'][:,i]))))
#     # l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(x_scale))))
#     plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['X']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['X'][0::DOWNSAMPLE,i]/x_scale,'.',color = colors[i],markersize = TRUTH_MARKER_SIZE)
#     plt.plot(d_SEQ[CURVE_NO]['X_est_n_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
#     plt.plot(d_DDMD[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle='solid', color=colors[i])
#     plt.plot(d_HAM[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
#     pl_max = np.max([pl_max,np.max(d_SEQ[CURVE_NO]['X'][:,i]/x_scale)])
#     pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['X'][:, i] / x_scale)])
# for i in range(n_outputs):
#     y_scale = 10 ** np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['Y'][:, i]))))
#     # plt.plot([], color=colors[n_states+i], label=('$y_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(y_scale))))
#     plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['Y']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['Y'][0::DOWNSAMPLE,i]/y_scale, '.',color = colors[n_states+i],markersize = TRUTH_MARKER_SIZE)
#     plt.plot(d_SEQ[CURVE_NO]['Y_est_n_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
#     plt.plot(d_DDMD[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle='solid', color=colors[n_states+i])
#     plt.plot(d_HAM[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle='dashdot', color=colors[n_states+i])
#     pl_max = np.max([pl_max, np.max(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
#     pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
# # l1 = plt.legend(loc='lower right',fontsize = 14)
# # plt.gca().add_artist(l1)
# a1, = plt.plot([],'.',markersize = TRUTH_MARKER_SIZE,label='Truth',color = 'grey')
# a2, = plt.plot([], linestyle = 'dashed',linewidth = 1,label='Seq ocdDMD',color = 'grey')
# a3, = plt.plot([], linestyle ='solid',linewidth = 1,label='Dir ocdDMD',color = 'grey')
# a4, = plt.plot([], linestyle ='dashdot',linewidth = 1,label='Hamm nn-model',color = 'grey')
# l1 = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =2)
# plt.gca().add_artist(l1)
# # l2 = plt.legend((a1,a2,a3),('Truth','Sequential oc-deepDMD','direct oc-deepDMD','Hammerstein model'),loc = "upper right",fontsize = FONT_SIZE)
# plt.xlabel('$k$ (time index)',fontsize = FONT_SIZE)
# plt.text(32,2,'[n-step]',fontsize = FONT_SIZE)
# # plt.ylabel('x,y [n -step]',fontsize = FONT_SIZE)
# plt.text(-20,2,'States and Outputs',rotation = 90,fontsize = FONT_SIZE)
# # plt.title('(b)',fontsize = FONT_SIZE)
# plt.ylim([pl_min-0.1,pl_max+0.1])
# plt.xticks(fontsize = TICK_FONT_SIZE)
# plt.yticks(fontsize = TICK_FONT_SIZE)
# plt.xlim([-0.5,100.5])
#
#
#
#
# plt.subplot2grid((10,16), (0,6), colspan=5, rowspan=3)
# # plt.bar(df_r2_SEQ.index,df_r2_SEQ.mean(axis=1),color = colors[1],label='Seq ocdDMD')
# # plt.plot(df_r2_DEEPDMD.index,df_r2_DEEPDMD.mean(axis=1),color = colors[0],label='dir ocdDMD', linewidth = LINE_WIDTH_c_d )
# plt.bar(df_r2_SEQ.columns.to_numpy(),df_r2_SEQ.to_numpy().reshape(-1),color = colors[1],label='Seq ocdDMD')
# plt.plot(df_r2_DEEPDMD.columns.to_numpy(),df_r2_DEEPDMD.to_numpy().reshape(-1),color = colors[0],label='dir ocdDMD', linewidth = LINE_WIDTH_c_d )
# plt.plot(df_r2_HAM.columns.to_numpy(),df_r2_HAM.to_numpy().reshape(-1),color = colors[2],label='Hamm nn-model',linewidth = LINE_WIDTH_c_d )
# plt.xlim([0.5,49.5])
# plt.ylim([85,101])
# STEPS = 10
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =2)
# plt.xticks(fontsize = TICK_FONT_SIZE)
# plt.yticks(fontsize = TICK_FONT_SIZE)
# plt.xticks(ticks=np.arange(10, 51, step=STEPS),labels=range(10,51,STEPS))
# plt.xlabel('# Prediction Steps',fontsize = FONT_SIZE)
# plt.ylabel('$r^2$(in %)',fontsize = FONT_SIZE)
# plt.title('(c)',fontsize = HEADER_SIZE,loc='left')
#
# plt.subplot2grid((10,16), (0,12), colspan=4, rowspan=3)
# p=0
# for i in range(Phi_t_SEQ.shape[0]):
#     if i in comp_modes_conj_SEQ:
#         continue
#     elif i in comp_modes_SEQ:
#         # plt.plot(Phi[i, :],label = 'lala')
#         plt.plot(Phi_t_SEQ[i,:],label='$\phi_{{{},{}}}(x)$'.format(i+1,comp_modes_conj_SEQ[comp_modes_SEQ.index(i)]+1), linewidth = LINE_WIDTH_c_d )
#         p = p+1
#     else:
#         plt.plot(Phi_t_SEQ[i, :], label='$\phi_{{{}}}(x)$'.format(i + 1), linewidth = LINE_WIDTH_c_d )
#         p = p+1
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =np.int(np.ceil(p/2)))
# plt.xlabel('$k$ (time index)',fontsize = FONT_SIZE)
# plt.ylabel('$\phi[k]$',fontsize = FONT_SIZE)
# plt.title('(d)',fontsize = HEADER_SIZE,loc='left')
# plt.xticks(fontsize = TICK_FONT_SIZE)
# plt.yticks(fontsize = TICK_FONT_SIZE)
# plt.xlim([0,100])
#
# p=0
# for i in range(PHI_SEQ.shape[2]):
#     title = ''
#     if p == 0:
#         f = plt.subplot2grid((10, 16), (5, 6-1), colspan=3, rowspan=2)
#     elif p == 1:
#         f = plt.subplot2grid((10, 16), (5, 10-1), colspan=3, rowspan=2)
#         # title = title + '(e)\n'
#     elif p == 2:
#         f = plt.subplot2grid((10, 16), (5, 14-1), colspan=3, rowspan=2)
#     elif p == 3:
#         f = plt.subplot2grid((10, 16), (8, 6-1), colspan=3, rowspan=2)
#     elif p == 4:
#         f = plt.subplot2grid((10, 16), (8, 10-1), colspan=3, rowspan=2)
#     elif p==5:
#         f = plt.subplot2grid((10, 16), (8, 14-1), colspan=3, rowspan=2)
#     elif p==6:
#         break
#     if i in comp_modes_conj_SEQ:
#         continue
#     elif i in comp_modes_SEQ:
#         c = f.pcolor(X1_SEQ,X2_SEQ,PHI_SEQ[:,:,i],cmap='rainbow', vmin=np.min(PHI_SEQ[:,:,i]), vmax=np.max(PHI_SEQ[:,:,i]))
#         plt.colorbar(c,ax = f)
#         plt.xlabel('$x_1$ \n' + '$\lambda=$' + str(round(np.real(E_SEQ[i]),2)) + r'$\pm$' + str(round(np.imag(E_SEQ[i]),2)), fontsize=FONT_SIZE)
#         plt.ylabel('$x_2$', fontsize=FONT_SIZE)
#         plt.xticks([-4, 0, 4])
#         plt.yticks([-4, 0, 4])
#         plt.title(title + '$\phi_{{{},{}}}(x)$'.format(i+1,comp_modes_conj_SEQ[comp_modes_SEQ.index(i)]+1), fontsize=FONT_SIZE)
#         # plt.text(-3.5,3.5,'$\lambda=$' + str(round(np.real(E_SEQ[i]),2)) + r'$\pm$' + str(round(np.imag(E_SEQ[i]),2)), fontsize=FONT_SIZE)
#         p = p+1
#     else:
#         c = f.pcolor(X1_SEQ, X2_SEQ, PHI_SEQ[:, :, i], cmap='rainbow', vmin=np.min(PHI_SEQ[:, :, i]),vmax=np.max(PHI_SEQ[:, :, i]))
#         plt.colorbar(c, ax=f)
#         plt.xlabel('$x_1$\n' + '$\lambda=$' + str(round(np.real(E_SEQ[i]),2)), fontsize=FONT_SIZE)
#         plt.ylabel('$x_2$', fontsize=FONT_SIZE)
#         plt.xticks([-4, 0, 4])
#         plt.yticks([-4, 0, 4])
#         plt.title(title + '$\phi_{{{}}}(x)$'.format(i + 1), fontsize=FONT_SIZE )
#         # plt.text(-3.5,3.5,'$\lambda=$' + str(round(np.real(E_SEQ[i]),2)), fontsize=FONT_SIZE)
#         p = p+1
#     if p ==1:
#         f.text(-5,5.5,'(e)',fontsize = HEADER_SIZE)
#
# # plt.savefig('Plots/eg2_2StateActivatorRespressorClock.svg')
# # plt.savefig('Plots/eg2_2StateActivatorRespressorClock_pycharm.png')
# plt.show()

##

# p=0
# for i in range(PHI_SEQ.shape[2]):
#     title = ''
#     if p == 0:
#         f = plt.subplot2grid((10, 16), (5, 6-1), colspan=3, rowspan=2)
#     elif p == 1:
#         f = plt.subplot2grid((10, 16), (5, 10-1), colspan=3, rowspan=2)
#         title = title + '(e)\n'
#     elif p == 2:
#         f = plt.subplot2grid((10, 16), (5, 14-1), colspan=3, rowspan=2)
#     elif p == 3:
#         f = plt.subplot2grid((10, 16), (8, 8-1), colspan=3, rowspan=2)
#     elif p == 4:
#         f = plt.subplot2grid((10, 16), (8, 12-1), colspan=3, rowspan=2)
#     elif p==5:
#         break
#     if i in comp_modes_conj_SEQ:
#         continue
#     elif i in comp_modes_SEQ:
#         c = f.pcolor(X1_SEQ,X2_SEQ,PHI_SEQ[:,:,i],cmap='rainbow', vmin=np.min(PHI_SEQ[:,:,i]), vmax=np.max(PHI_SEQ[:,:,i]))
#         plt.colorbar(c,ax = f)
#         plt.xlabel('$x_1$', fontsize=FONT_SIZE)
#         plt.ylabel('$x_2$', fontsize=FONT_SIZE)
#         plt.title(title + '$\phi_{{{},{}}}(x)$'.format(i+1,comp_modes_conj_SEQ[comp_modes_SEQ.index(i)]+1) + ',$\lambda=$' + str(round(E_SEQ[i,i],3)) + '+j' + str(round(E_SEQ[i,i+1],3)), fontsize=FONT_SIZE)
#         p = p+1
#     else:
#         c = f.pcolor(X1_SEQ, X2_SEQ, PHI_SEQ[:, :, i], cmap='rainbow', vmin=np.min(PHI_SEQ[:, :, i]),vmax=np.max(PHI_SEQ[:, :, i]))
#         plt.colorbar(c, ax=f)
#         plt.xlabel('$x_1$', fontsize=FONT_SIZE)
#         plt.ylabel('$x_2$', fontsize=FONT_SIZE)
#         plt.title(title + '$\phi_{{{}}}(x)$'.format(i + 1) + ',$\lambda=$' + str(round(E_SEQ[i,i],3)), fontsize=FONT_SIZE)
#         p = p+1
#
# plt.savefig('Plots/eg2_2StateActivatorRespressorClock.svg')
# plt.savefig('Plots/eg2_2StateActivatorRespressorClock.png')
# plt.show()


# ## Dynamic Modes
# Senergy_THRESHOLD = 99.9
# # Required Variables - K, psiX,
# CURVE_NO = 0
# psiX = d[CURVE_NO]['psiX'].T
# K = dict_params['KxT_num'].T
# psiXp_data = psiX[:,0:-1]
# psiXf_data = psiX[:,1:]
# # Minimal POD modes of psiX
# U,S,VT = np.linalg.svd(psiXp_data)
# Senergy = np.cumsum(S**2)/np.sum(S**2)*100
# for i in range(len(S)):
#     if Senergy[i] > Senergy_THRESHOLD:
#         nPC = i+1
#         break
# print('Optimal POD modes chosen : ', nPC)
# Ur = U[:,0:nPC]
# plt.figure()
# _,s,_T = np.linalg.svd(d[CURVE_NO]['X'])
# plt.stem(np.arange(len(s)),(np.cumsum(s**2)/np.sum(s**2))*100)
# plt.plot([0,len(s)-1],[100,100])
# plt.title('just X')
# plt.show()
# plt.figure()
# _,s,_T = np.linalg.svd(d[CURVE_NO]['psiX'])
# plt.stem(np.arange(len(s)),(np.cumsum(s**2)/np.sum(s**2))*100)
# plt.plot([0,len(s)-1],[100,100])
# plt.title('psiX')
# plt.show()
# # Reduced K - Kred
# Kred = np.matmul(np.matmul(Ur.T,K),Ur)
# # Eigendecomposition of Kred - Right eigenvectors
# # eval,W = np.linalg.eig(Kred.T)
# eval,W = np.linalg.eig(Kred)
# E = np.diag(eval)
# comp_modes =[]
# comp_modes_conj =[]
# for i1 in range(len(eval)):
#     if np.imag(E[i1,i1]) != 0:
#         print(i1)
#         # Find the complex conjugate
#         for i2 in range(i1+1,len(eval)):
#             if eval[i2] == eval[i1].conj():
#                 break
#         # i1 and i2 are the indices of the complex conjugate eigenvalues
#         comp_modes.append(i1)
#         comp_modes_conj.append(i2)
#         E[i1,i1] = np.real(eval[i1])
#         E[i2, i2] = np.real(eval[i1])
#         E[i1, i2] = np.imag(eval[i1])
#         E[i2, i1] = - np.imag(eval[i1])
#         u1 = copy.deepcopy(np.real(W[:, i1:i1 + 1]))
#         w1 = copy.deepcopy(np.imag(W[:, i1:i1 + 1]))
#         W[:, i1:i1 + 1] = u1
#         W[:, i2:i2 + 1] = w1
# E = np.real(E)
# W = np.real(W)
# Winv = np.linalg.inv(W)
# # Koopman eigenfunctions
# Phi = np.matmul(E,np.matmul(np.matmul(Winv,Ur.T),psiXp_data))
# # plt.figure()
# # plt.plot(Phi.T)
# # plt.legend(np.arange(nPC))
# # plt.show()
#
# ## Eigenfunctions and Observables
# n_observables = psiXp_data.shape[0]
# sampling_resolution = 0.1
# x1 = np.arange(-5, 5 + sampling_resolution, sampling_resolution)
# x2 = np.arange(-5, 5 + sampling_resolution, sampling_resolution)
# X1, X2 = np.meshgrid(x1, x2)
# PHI = np.zeros(shape=(X1.shape[0], X1.shape[1],nPC))
# PSI = np.zeros(shape=(X1.shape[0], X1.shape[1],n_observables))
# for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
#     x1_i = X1[i, j]
#     x2_i = X2[i, j]
#     psiXT_i = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: np.array([[x1_i, x2_i]])})
#     PHI[i, j, :] = np.matmul(np.matmul(Winv,Ur.T),psiXT_i.T).reshape((1,1,-1))
#     PSI[i, j, :] = psiXT_i.reshape((1, 1, -1))
# ## Observables
# f,ax = plt.subplots(1,n_observables,figsize = (2*n_observables,1.5))
# for i in range(n_observables):
#     c = ax[i].pcolor(X1,X2,PSI[:,:,i],cmap='rainbow', vmin=np.min(PSI[:,:,i]), vmax=np.max(PSI[:,:,i]))
#     f.colorbar(c,ax = ax[i])
# f.show()
# ## Eigenfunctions
# f,ax = plt.subplots(1,nPC,figsize = (2*nPC,1.5))
# for i in range(nPC):
#     c = ax[i].pcolor(X1,X2,PHI[:,:,i],cmap='rainbow', vmin=np.min(PHI[:,:,i]), vmax=np.max(PHI[:,:,i]))
#     f.colorbar(c,ax = ax[i])
# f.show()
# # Koopman modes - UW
#
#
# # Dynamic modes - Lambda*inv(W)*U.T*psiX
#
# ## [Phase Portrait]
#
# # System Parameters
# gamma_A = 1.
# gamma_B = 0.5
# delta_A = 1.
# delta_B = 1.
# alpha_A0= 0.04
# alpha_B0= 0.004
# alpha_A = 50.
# alpha_B = 30.
# K_A = 1.
# K_B = 1.5
# kappa_A = 1.
# kappa_B = 1.
# n = 2.
# m = 2.
# k_3n = 3.
# k_3d = 1.08
#
#
# sys_params_arc2s = (gamma_A,gamma_B,delta_A,delta_B,alpha_A0,alpha_B0,alpha_A,alpha_B,K_A,K_B,kappa_A,kappa_B,n,m)
# Ts = 0.1
# t_end = 40
# # Simulation Parameters
# dict_data = {}
# X0 = np.empty(shape=(0, 2))
# t = np.arange(0, t_end, Ts)
#
# # Phase Space Data
# dict_data = {}
# X0 = np.empty(shape=(0, 2))
# i=0
# for x1,x2 in itertools.product(list(np.arange(1,30.,3.5)), list(np.arange(1,60.,8))):
#     dict_data[i]={}
#     x0_curr =  np.array([x1,x2])
#     X0 = np.concatenate([X0, np.array([[x1,x2]])], axis=0)
#     dict_data[i]['X'] = oc.odeint(oc.activator_repressor_clock_2states, x0_curr, t, args=sys_params_arc2s)
#     dict_data[i]['Y'] = k_3n * dict_data[i]['X'][:, 1:2] / (k_3d + dict_data[i]['X'][:, 3:4])
#     i = i+1
#
# ## [R2 function of prediction steps] Calculate the accuracy as a funcation of the number of steps predicted
# CURVE_NO = 0
# ls_steps = list(range(1,50,1))
# dict_rmse = {}
# dict_r2 = {}
# for CURVE_NO in range(200,300):
#     dict_rmse[CURVE_NO] = {}
#     dict_r2[CURVE_NO] = {}
#     dict_DATA_i = oc.scale_data_using_existing_scaler_folder(d[CURVE_NO], SYS_NO)
#     X_scaled = dict_DATA_i['X']
#     Y_scaled = dict_DATA_i['Y']
#     psiX = psixfT.eval(feed_dict={xfT_feed: X_scaled})
#     for i in ls_steps:  # iterating through each step prediction
#         np_psiX_true = psiX[i:, :]
#         np_psiX_pred = np.matmul(psiX[:-i, :], np.linalg.matrix_power(KxT_num, i))  # i step prediction at each datapoint
#         Y_pred = np.matmul(np_psiX_pred, WhT_num)
#         Y_true = Y_scaled[i:, :]
#         dict_rmse[CURVE_NO][i] = np.sqrt(np.mean(np.square(np_psiX_true - np_psiX_pred)))
#         dict_r2[CURVE_NO][i] = np.max([0, (1 - (np.sum(np.square(np_psiX_true - np_psiX_pred))+np.sum(np.square(Y_true - Y_pred))) / (np.sum(np.square(np_psiX_true))+np.sum(np.square(Y_true)))) * 100])
# df_r2 = pd.DataFrame(dict_r2)
# print(df_r2)
# CHECK_VAL =df_r2.iloc[-1,:].max()
# for i in range(200,300):
#     if df_r2.loc[df_r2.index[-1],i] == CHECK_VAL:
#         CURVE_NO = i
#         break
# ## Figure 1
# plt.figure(figsize=(18,7))
# plt.subplot2grid((7,18), (0,0), colspan=6, rowspan=4)
# alpha = 1.0
# epsilon = alpha - 0.01
# arrow_length = 1.2
# ls_pts = list(range(0,1))
# for i in list(dict_data.keys())[0:]:
#     for j in ls_pts:
#         if np.abs(dict_data[i]['X'][j, 0]) > 1 or j==0:
#             plt.plot(dict_data[i]['X'][j, 0], dict_data[i]['X'][j, 1], 'o',color='salmon',fillstyle='none',markersize=5)
#     plt.plot(dict_data[i]['X'][:, 0], dict_data[i]['X'][:, 1], color='tab:blue',linewidth=0.3)
#     if np.mod(i,1)==0:
#         for j in ls_pts:
#             dist = np.sqrt((dict_data[i]['X'][j, 0] - dict_data[i]['X'][j + 1, 0]) ** 2 + (dict_data[i]['X'][j, 1] - dict_data[i]['X'][j + 1, 1]) ** 2)
#             x = dict_data[i]['X'][j, 0]
#             y = dict_data[i]['X'][j, 1]
#             dx = (dict_data[i]['X'][j + 1, 0] - dict_data[i]['X'][j, 0]) * arrow_length
#             dy = (dict_data[i]['X'][j + 1, 1] - dict_data[i]['X'][j, 1]) * arrow_length
#             # print(x,' ',y,' ',dist)
#             if dist<0.1:
#                 plt.arrow(x,y,dx,dy,head_width = 0.02,head_length=0.03,alpha=1,color='tab:green')
#             else:
#                 plt.arrow(x, y, dx, dy, head_width=1., head_length=0.9, alpha=1, color='tab:green')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.plot(dict_data[0]['X'][200:,0],dict_data[0]['X'][200:,1],color='tab:red',markersize=10)
# plt.plot(dict_data[5]['X'][200:,0],dict_data[5]['X'][200:,1],color='tab:red',markersize=10)
# plt.plot(dict_data[10]['X'][200:,0],dict_data[10]['X'][200:,1],color='tab:red',markersize=10)
# # plt.xlim([0,0.6])
# # plt.ylim([0,1.65])
# plt.xlim([-0.1,30])
# plt.ylim([-0.1,60])
#
#
# # CURVE_NO = 0
# plt.subplot2grid((7,18), (0,6), colspan=6, rowspan=4)
# n_states = d[CURVE_NO]['X'].shape[1]
# n_outputs = d[CURVE_NO]['Y'].shape[1]
# for i in range(n_states):
#     x_scale = 10**np.round(np.log10(np.max(np.abs(d[CURVE_NO]['X'][:,i]))))
#     l1_i, = plt.plot(0, color=colors[i],label=('$x_{}$').format(i + 1) + ('$[x10^{}]$').format(np.int(np.log10(x_scale))))
#     plt.plot(d[CURVE_NO]['X'][:,i]/x_scale,'.',color = colors[i],linewidth = 5)
#     plt.plot(d[CURVE_NO]['X_est_one_step'][:, i]/x_scale,linestyle = 'solid',color=colors[i])
#     plt.plot(d[CURVE_NO]['X_est_n_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
# for i in range(n_outputs):
#     y_scale = 10 ** np.round(np.log10(np.max(np.abs(d[CURVE_NO]['Y'][:, i]))))
#     plt.plot(0, color=colors[i], label=('$y_{}$').format(i + 1) + ('$[x10^{}]$').format(np.int(np.log10(y_scale))))
#     plt.plot(d[CURVE_NO]['Y'][:,i]/y_scale, '.',color = colors[n_states+i],linewidth = 5)
#     plt.plot(d[CURVE_NO]['Y_est_one_step'][:, i]/y_scale, linestyle ='solid',color=colors[n_states+i])
#     plt.plot(d[CURVE_NO]['Y_est_n_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
# l1 = plt.legend(loc="upper right")
# plt.gca().add_artist(l1)
# a1, = plt.plot(0,'.',linewidth = 5,label='Truth',color = 'grey')
# a2, = plt.plot(0, linestyle ='solid',linewidth = 1,label='1-step',color = 'grey')
# a3, = plt.plot(0, linestyle = 'dashed',linewidth = 1,label='n-step',color = 'grey')
# l2 = plt.legend((a1,a2,a3),('Truth','1-step','n-step'),loc = "lower right")
# plt.xlabel('Time Index(k)')
# plt.ylabel('States and Outputs')
# plt.title('(b)')
#
#
# plt.subplot2grid((7,18), (0,12), colspan=6, rowspan=4)
# for i in range(nPC):
#     if i in comp_modes_conj:
#         continue
#     elif i in comp_modes:
#         # plt.plot(Phi[i, :],label = 'lala')
#         plt.plot(Phi[i,:],label='$\phi_{{{},{}}}(x)$'.format(i+1,comp_modes_conj[comp_modes.index(i)]+1))
#     else:
#         plt.plot(Phi[i, :], label='$\phi_{}(x)$'.format(i + 1))
# plt.legend()
# plt.xlabel('Time Index(k)')
# plt.ylabel('Evolution of eigenfunctions')
# plt.title('(c)')
#
#
# plt.subplot2grid((7,18), (4,0), colspan=3, rowspan=2)
# plt.bar(df_r2.index,df_r2.mean(axis=1))
# plt.xlim([0.5,50.5])
# plt.ylim([50,100])
# STEPS = 10
# plt.xticks(ticks=np.arange(10, 51, step=STEPS),labels=range(10,51,STEPS))
# plt.xlabel('# Prediction Steps \n (d)')
# plt.ylabel('$r^2$(in %)')
#
#
#
# for i in range(nPC):
#     f = plt.subplot2grid((7,18), (4, 3*(i+1)), colspan=3, rowspan=2)
#     c = f.pcolor(X1,X2,PHI[:,:,i],cmap='rainbow', vmin=np.min(PHI[:,:,i]), vmax=np.max(PHI[:,:,i]))
#     if i!=0:
#         plt.yticks([])
#     else:
#         plt.ylabel('$x_2$')
#     if i ==2:
#         plt.xlabel('$x_1$ \n (e)')
#         plt.title('$\lambda_{} =$'.format(i+1) + str(round(eval[i],3)))
#     else:
#         plt.title('$\lambda_{} =$'.format(i+1) + str(round(eval[i],3)))
#         plt.xlabel('$x_1$')
#     plt.colorbar(c,ax = f)


# plt.show()

##

NORMALIZE = True
title = ''
FONT_SIZE = 32
max_eigs = 9
f,ax = plt.subplots(2,7,sharex=True,sharey=True,figsize=(3*max_eigs,10), gridspec_kw={'hspace': 0.35})
for row_i in range(2):
    if row_i ==0:
        # x DMD modes
        comp_modes_conj = comp_modes_conj_DEEP_X
        comp_modes = comp_modes_DEEP_X
        PHI = PHI_DEEP_X
        PSI = PSI_DEEP_X
        X1 = X1_DEEP_X
        X2 = X2_DEEP_X
        E = E_DEEP_X
        y0_title = '\n deepDMD \n \n'
    elif row_i ==1:
        # Dir ocdDMD modes
        comp_modes_conj = comp_modes_conj_DEEPDMD
        comp_modes = comp_modes_DEEPDMD
        PHI = PHI_DEEPDMD
        PSI = PSI_DEEPDMD
        X1 = X1_DEEPDMD
        X2 = X2_DEEPDMD
        E = E_DEEPDMD
        H = H_DEEPDMD
        y0_title = 'Direct \n OC-deepDMD \n \n'
    for i in range(6):
            c = ax[row_i, i].pcolor(X1, X2, PSI[:, :, i] / np.max(np.abs(PSI[:, :, i])), cmap='rainbow', vmin=-1,vmax=1)
            # f.colorbar(c, ax=ax[row_i,p])
            # ax[row_i,p].set_xlabel('$x_1$\n' + '$\lambda=$' + str(round(np.real(E[i]), 2)), fontsize=FONT_SIZE)
            ax[row_i, i].set_title(title + '$\psi_{{{}}}(x)$'.format(i + 1), fontsize=FONT_SIZE)
            # plt.text(-3.5,3.5,'$\lambda=$' + str(round(np.real(E_SEQ[i]),2)), fontsize=FONT_SIZE)
            # ax[row_i,p].set_ylabel('$x_2$', fontsize=FONT_SIZE)
            ax[row_i, i].set_xticks([-3, 0, 3])
            ax[row_i, i].set_xticklabels([-3, 0, 3], fontsize=FONT_SIZE)
            ax[row_i, i].set_yticks([-3, 0, 3])
            ax[row_i, i].set_yticklabels([-3, 0, 3], fontsize=FONT_SIZE)
            if i==0:
                ax[row_i, i].set_ylabel(y0_title + '$x_2$', fontsize=FONT_SIZE)
            if row_i ==1:
                ax[row_i, i].set_xlabel('$x_1$', fontsize=FONT_SIZE)
    if row_i == 1:
        ax[row_i,6].pcolor(X1, X2, H[:, :] / np.max(np.abs(H[:, :])), cmap='rainbow', vmin=-1,vmax=1)
        ax[row_i, 6].set_title(title + '$h(x)$', fontsize=FONT_SIZE)
        ax[row_i, 6].set_xticks([-3, 0, 3])
        ax[row_i, 6].set_xticklabels([-3, 0, 3], fontsize=FONT_SIZE)
        ax[row_i, 6].set_yticks([-3, 0, 3])
        ax[row_i, 6].set_yticklabels([-3, 0, 3], fontsize=FONT_SIZE)
        ax[row_i, 6].set_xlabel('$x_1$', fontsize=FONT_SIZE)
    else:
        ax[row_i, 6].axis('off')
cbar = f.colorbar(c, ax=ax.ravel().tolist(), shrink=0.95,  ticks = [-1.,-0.5,0,0.5,1.])
plt.savefig('Plots/eg2_MEMS_obs.png')

f.show()


##

NORMALIZE = True
title = ''
FONT_SIZE = 32
max_eigs = 9
f,ax = plt.subplots(2,5,sharex=True,sharey=True,figsize=(3*max_eigs,10), gridspec_kw={'hspace': 0.35})
for row_i in range(2):
    if row_i ==0:
        # x DMD modes
        comp_modes_conj = comp_modes_conj_DEEP_X
        comp_modes = comp_modes_DEEP_X
        PHI = PHI_DEEP_X
        PSI = PSI_DEEP_X
        X1 = X1_DEEP_X
        X2 = X2_DEEP_X
        E = E_DEEP_X
        y0_title = '\n deepDMD \n \n'
    elif row_i ==1:
        # Dir ocdDMD modes
        comp_modes_conj = comp_modes_conj_DEEPDMD
        comp_modes = comp_modes_DEEPDMD
        PHI = PHI_DEEPDMD
        PSI = PSI_DEEPDMD
        X1 = X1_DEEPDMD
        X2 = X2_DEEPDMD
        E = E_DEEPDMD
        H = H_DEEPDMD
        y0_title = 'Direct \n OC-deepDMD \n \n'
    p = 0
    for i in range(PHI.shape[-1]):
        if i in comp_modes_conj:
            continue
        elif i in comp_modes:
            c = ax[row_i, p].pcolor(X1, X2, PHI[:, :, i] / np.max(np.abs(PHI[:, :, i])), cmap='rainbow', vmin=-1,vmax=1)
            # f.colorbar(c, ax=ax[row_i,p])
            # ax[row_i,p].set_xlabel('$x_1$ \n' + '$\lambda=$' + str(round(np.real(E[i]), 2)) + r'$\pm$' + 'j' + str(round(np.imag(E[i]), 2)), fontsize=FONT_SIZE)
            ax[row_i, p].set_title(title + '$\phi_{{{},{}}}(x)$'.format(i + 1, comp_modes_conj[
                comp_modes.index(i)] + 1) + '\n $\lambda=$' + str(round(np.real(E[i]), 2)) + r'$\pm$' + 'j' + str(
                round(np.imag(E[i]), 2)), fontsize=FONT_SIZE)
            # plt.text(-3.5,3.5,'$\lambda=$' + str(round(np.real(E_SEQ[i]),2)) + r'$\pm$' + str(round(np.imag(E_SEQ[i]),2)), fontsize=FONT_SIZE)
            # ax[row_i, p].set_ylabel('$x_2$', fontsize=FONT_SIZE)
        else:
            c = ax[row_i, p].pcolor(X1, X2, PHI[:, :, i] / np.max(np.abs(PHI[:, :, i])), cmap='rainbow', vmin=-1,vmax=1)
            # f.colorbar(c, ax=ax[row_i,p])
            # ax[row_i,p].set_xlabel('$x_1$\n' + '$\lambda=$' + str(round(np.real(E[i]), 2)), fontsize=FONT_SIZE)
            ax[row_i, p].set_title(
                title + '$\phi_{{{}}}(x)$'.format(i + 1) + '\n $\lambda=$' + str(round(np.real(E[i]), 2)),
                fontsize=FONT_SIZE)
            # plt.text(-3.5,3.5,'$\lambda=$' + str(round(np.real(E_SEQ[i]),2)), fontsize=FONT_SIZE)
            # ax[row_i,p].set_ylabel('$x_2$', fontsize=FONT_SIZE)
        if p == 0:
            ax[row_i, p].set_ylabel(y0_title + '$x_2$', fontsize=FONT_SIZE)
        if row_i == 1:
            ax[row_i, p].set_xlabel('$x_1$', fontsize=FONT_SIZE)
        ax[row_i, p].set_xticks([-3, 0, 3])
        ax[row_i, p].set_xticklabels([-3, 0, 3], fontsize=FONT_SIZE)
        ax[row_i, p].set_yticks([-3, 0, 3])
        ax[row_i, p].set_yticklabels([-3, 0, 3], fontsize=FONT_SIZE)
        p = p + 1
    if row_i == 1:
        ax[row_i, 4].pcolor(X1, X2, H[:, :] / np.max(np.abs(H[:, :])), cmap='rainbow', vmin=-1,vmax=1)
        ax[row_i, 4].set_title(title + '$h(x)$ \n', fontsize=FONT_SIZE)
        ax[row_i, 4].set_xticks([-3, 0, 3])
        ax[row_i, 4].set_xticklabels([-3, 0, 3], fontsize=FONT_SIZE)
        ax[row_i, 4].set_yticks([-3, 0, 3])
        ax[row_i, 4].set_yticklabels([-3, 0, 3], fontsize=FONT_SIZE)
        ax[row_i, 4].set_xlabel('$x_1$', fontsize=FONT_SIZE)
    else:
        ax[row_i, 4].axis('off')
cbar = f.colorbar(c, ax=ax.ravel().tolist(), shrink=0.95,  ticks = [-1.,-0.5,0,0.5,1.])
f.savefig('Plots/eg2_MEMS_eigfunc.png')
f.show()

######


NORMALIZE = True
title = ''
FONT_SIZE = 44
max_eigs = np.max([PHI_DEEP_X.shape[-1] - len(comp_modes_DEEP_X),PHI_SEQ.shape[-1] - len(comp_modes_SEQ),PHI_DEEPDMD.shape[-1] - len(comp_modes_conj_DEEPDMD)]) + 1
# max_eigs = np.max([PHI_DEEP_X.shape[-1],PHI_DEEPDMD.shape[-1]])
# plt.figure(figsize=(30,5))
f,ax = plt.subplots(3,max_eigs,sharex=True,sharey=True,figsize=(8*max_eigs,20), gridspec_kw={'hspace': 0.35})
for row_i in range(3):
    if row_i ==0:
        # x DMD modes
        comp_modes_conj = comp_modes_conj_DEEP_X
        comp_modes = comp_modes_DEEP_X
        PHI = PHI_DEEP_X
        X1 = X1_DEEP_X
        X2 = X2_DEEP_X
        E = E_DEEP_X
        y0_title = '\n deepDMD \n \n'
    elif row_i ==1:
        # Seq ocdDMD modes
        comp_modes_conj = comp_modes_conj_SEQ
        comp_modes = comp_modes_SEQ
        PHI = PHI_SEQ
        X1 = X1_SEQ
        X2 = X2_SEQ
        E = E_SEQ
        H = H_SEQ
        y0_title = 'Sequential \n OC-deepDMD \n \n'
    elif row_i ==2:
        # Dir ocdDMD modes
        comp_modes_conj = comp_modes_conj_DEEPDMD
        comp_modes = comp_modes_DEEPDMD
        PHI = PHI_DEEPDMD
        X1 = X1_DEEPDMD
        X2 = X2_DEEPDMD
        E = E_DEEPDMD
        H = H_DEEPDMD
        y0_title = 'Direct \n OC-deepDMD \n \n'
    p = 0
    if row_i ==0:
        c = ax[0, 0].pcolor(X1, X2, PHI[:, :, 4] / np.max(np.abs(PHI[:, :, 4])), cmap='rainbow', vmin=-1, vmax=1)
        ax[0, 0].set_title(title + '$\phi_{{{}}}(x)$'.format(1) + '\n $\lambda=$' + str(round(np.real(E[4]), 2)),fontsize=FONT_SIZE)
        ax[0, 0].set_ylabel(y0_title , fontsize=FONT_SIZE)
        ax[0, 0].set_yticks([-8, 0, 8])
        ax[0, 0].set_yticklabels([-8, 0, 8], fontsize=FONT_SIZE)
        ax[0, 1].axis('off')
        ax[0, 2].pcolor(X1, X2, PHI[:, :, 2] / np.max(np.abs(PHI[:, :, 2])), cmap='rainbow', vmin=-1, vmax=1)
        ax[0, 2].set_title(title + '$\phi_{{{},{}}}(x)$'.format(2,3) + '\n $\lambda=$' + str(round(np.real(E[2]), 2)) + r'$\pm$' + 'j' + str(round(np.imag(E[2]), 2)),fontsize=FONT_SIZE)
        ax[0, 3].pcolor(X1, X2, PHI[:, :, 0] / np.max(np.abs(PHI[:, :, 0])), cmap='rainbow', vmin=-1, vmax=1)
        ax[0, 3].set_title(title + '$\phi_{{{},{}}}(x)$'.format(4,5) + '\n $\lambda=$' + str(round(np.real(E[0]), 2)) + r'$\pm$' + 'j' + str(
            round(np.imag(E[0]), 2)), fontsize=FONT_SIZE)
        ax[0, 4].axis('off')
        ax[0, 5].pcolor(X1, X2, PHI[:, :, 5] / np.max(np.abs(PHI[:, :, 5])), cmap='rainbow', vmin=-1, vmax=1)
        ax[0, 5].set_title(title + '$\phi_{{{}}}(x)$'.format(6) + '\n $\lambda=$' + str(round(np.real(E[5]), 2)),fontsize=FONT_SIZE)
        ax[0, 6].axis('off')
    else:
        for i in range(PHI.shape[-1]):
            if i in comp_modes_conj:
                continue
            elif i in comp_modes:
                if NORMALIZE:
                    c = ax[row_i, p].pcolor(X1, X2, PHI[:, :, i] / np.max(np.abs(PHI[:, :, i])), cmap='rainbow', vmin=-1, vmax=1)
                else:
                    c = ax[row_i, p].pcolor(X1, X2, PHI[:, :, i], cmap='rainbow', vmin=np.min(PHI[:, :, i]),vmax=np.max(PHI[:, :, i]))
                # f.colorbar(c, ax=ax[row_i,p])
                # ax[row_i,p].set_xlabel('$x_1$ \n' + '$\lambda=$' + str(round(np.real(E[i]), 2)) + r'$\pm$' + 'j' + str(round(np.imag(E[i]), 2)), fontsize=FONT_SIZE)
                ax[row_i,p].set_title(title + '$\phi_{{{},{}}}(x)$'.format(i + 1, comp_modes_conj[comp_modes.index(i)] + 1) + '\n $\lambda=$' + str(round(np.real(E[i]), 2)) + r'$\pm$' + 'j' + str(round(np.imag(E[i]), 2)),fontsize=FONT_SIZE)
                # plt.text(-3.5,3.5,'$\lambda=$' + str(round(np.real(E_SEQ[i]),2)) + r'$\pm$' + str(round(np.imag(E_SEQ[i]),2)), fontsize=FONT_SIZE)
                # ax[row_i, p].set_ylabel('$x_2$', fontsize=FONT_SIZE)
                ax[row_i, p].set_xticks([-8, 0, 8])
                ax[row_i, p].set_xticklabels([-8, 0, 8], fontsize=FONT_SIZE)
                ax[row_i, p].set_yticks([-8, 0, 8])
                ax[row_i, p].set_yticklabels([-8, 0, 8], fontsize=FONT_SIZE)
                p = p + 1
            else:
                if NORMALIZE:
                    c = ax[row_i, p].pcolor(X1, X2, PHI[:, :, i]/ np.max(np.abs(PHI[:, :, i])), cmap='rainbow', vmin=-1,vmax=1)
                else:
                    c = ax[row_i,p].pcolor(X1, X2, PHI[:, :, i], cmap='rainbow', vmin=np.min(PHI[:, :, i]),vmax=np.max(PHI[:, :, i]))
                # f.colorbar(c, ax=ax[row_i,p])
                # ax[row_i,p].set_xlabel('$x_1$\n' + '$\lambda=$' + str(round(np.real(E[i]), 2)), fontsize=FONT_SIZE)
                ax[row_i,p].set_title(title + '$\phi_{{{}}}(x)$'.format(i + 1) + '\n $\lambda=$' + str(round(np.real(E[i]), 2)), fontsize=FONT_SIZE)
                # plt.text(-3.5,3.5,'$\lambda=$' + str(round(np.real(E_SEQ[i]),2)), fontsize=FONT_SIZE)
                # ax[row_i,p].set_ylabel('$x_2$', fontsize=FONT_SIZE)
                ax[row_i, p].set_xticks([-8, 0, 8])
                ax[row_i, p].set_xticklabels([-8, 0, 8], fontsize=FONT_SIZE)
                ax[row_i, p].set_yticks([-8, 0, 8])
                ax[row_i, p].set_yticklabels([-8, 0, 8], fontsize=FONT_SIZE)
                p = p+1
            # if row_i == 2 and p == 4:
            #     ax[row_i, p-1].set_xlabel('$x_1$', fontsize=FONT_SIZE)
            if row_i ==1 and p == 1:
                ax[row_i, p-1].set_ylabel(y0_title + '$x_2$', fontsize=FONT_SIZE)
            if row_i ==2 and p==1:
                ax[row_i, p-1].set_ylabel(y0_title, fontsize=FONT_SIZE)
            # else:
            #     print('None')
                # ax[row_i, p].set_visible(False)
                # p = p + 1
        # Plot the output
        if row_i == 1 or row_i == 2:
            c = ax[row_i, -1].pcolor(X1, X2, H / np.max(np.abs(H)), cmap='rainbow', vmin=-1, vmax=1)
            ax[row_i, -1].set_title('$h(x)$\n', fontsize=FONT_SIZE)
# ax[-1, -1].set_xlabel('$x_1$', fontsize=FONT_SIZE)
ax[-1, -1].set_xticks([-8, 0, 8])
ax[-1, -1].set_xticklabels([-8, 0, 8], fontsize=FONT_SIZE)



cbar = f.colorbar(c, ax=ax.ravel().tolist(), shrink=0.95,  ticks = [-1.,-0.5,0,0.5,1.])
# ax[0,-1].axis('off')
# ax[0,-2].axis('off')
# ax[0,-3].axis('off')
cbar.ax.set_yticklabels([-1.,-0.5,0,0.5,1.], fontsize=FONT_SIZE)
plt.savefig('Plots/eg2_MEMS_eigfunc.png')
f.show()
exit()

## PLOT FOR PAPER

title = ''
FONT_SIZE = 44
max_eigs = np.max([PHI_DEEP_X.shape[-1] - len(comp_modes_DEEP_X),PHI_SEQ.shape[-1] - len(comp_modes_SEQ),PHI_DEEPDMD.shape[-1] - len(comp_modes_conj_DEEPDMD)]) + 1
# max_eigs = np.max([PHI_DEEP_X.shape[-1],PHI_DEEPDMD.shape[-1]])
# plt.figure(figsize=(30,5))
f,ax = plt.subplots(3,max_eigs,sharex=True,sharey=True,figsize=(8*max_eigs,20), gridspec_kw={'hspace': 0.35})
for row_i in range(3):
    if row_i ==0:
        # x DMD modes
        comp_modes_conj = comp_modes_conj_DEEP_X
        comp_modes = comp_modes_DEEP_X
        PHI = PHI_DEEP_X
        X1 = X1_DEEP_X
        X2 = X2_DEEP_X
        E = E_DEEP_X
        y0_title = '\n deepDMD \n \n'
    elif row_i ==1:
        # Seq ocdDMD modes
        comp_modes_conj = comp_modes_conj_SEQ
        comp_modes = comp_modes_SEQ
        PHI = PHI_SEQ
        X1 = X1_SEQ
        X2 = X2_SEQ
        E = E_SEQ
        H = H_SEQ
        y0_title = 'Sequential \n OC-deepDMD \n \n'
    elif row_i ==2:
        # Dir ocdDMD modes
        comp_modes_conj = comp_modes_conj_DEEPDMD
        comp_modes = comp_modes_DEEPDMD
        PHI = PHI_DEEPDMD
        X1 = X1_DEEPDMD
        X2 = X2_DEEPDMD
        E = E_DEEPDMD
        H = H_DEEPDMD
        y0_title = 'Direct \n OC-deepDMD \n \n'
    p = 0
    if row_i ==0:
        c = ax[0, 0].pcolor(X1, X2, PHI[:, :, 4] / np.max(np.abs(PHI[:, :, 4])), cmap='rainbow', vmin=-1, vmax=1)
        ax[0, 0].set_title(title + '$\phi_{{{}}}(x)$'.format(1) + '\n $\lambda=$' + str(round(np.real(E[4]), 2)),fontsize=FONT_SIZE)
        ax[0, 0].set_ylabel(y0_title , fontsize=FONT_SIZE)
        ax[0, 0].set_yticks([-8, 0, 8])
        ax[0, 0].set_yticklabels([-8, 0, 8], fontsize=FONT_SIZE)
        ax[0, 1].pcolor(X1, X2, PHI[:, :, 2] / np.max(np.abs(PHI[:, :, 2])), cmap='rainbow', vmin=-1, vmax=1)
        ax[0, 1].set_title(title + '$\phi_{{{},{}}}(x)$'.format(2,3) + '\n $\lambda=$' + str(round(np.real(E[2]), 2)) + r'$\pm$' + 'j' + str(round(np.imag(E[2]), 2)),fontsize=FONT_SIZE)
        ax[0, 2].pcolor(X1, X2, PHI[:, :, 0] / np.max(np.abs(PHI[:, :, 0])), cmap='rainbow', vmin=-1, vmax=1)
        ax[0, 2].set_title(title + '$\phi_{{{},{}}}(x)$'.format(4,5) + '\n $\lambda=$' + str(round(np.real(E[0]), 2)) + r'$\pm$' + 'j' + str(
            round(np.imag(E[0]), 2)), fontsize=FONT_SIZE)
        ax[0, 3].pcolor(X1, X2, PHI[:, :, 5] / np.max(np.abs(PHI[:, :, 5])), cmap='rainbow', vmin=-1, vmax=1)
        ax[0, 3].set_title(title + '$\phi_{{{}}}(x)$'.format(6) + '\n $\lambda=$' + str(round(np.real(E[5]), 2)),fontsize=FONT_SIZE)
        ax[0, 4].axis('off')
        ax[0, 5].axis('off')
        ax[0, 6].axis('off')
    elif row_i == 1:
        c = ax[1, 0].pcolor(X1, X2, PHI[:, :, 0] / np.max(np.abs(PHI[:, :, 0])), cmap='rainbow', vmin=-1, vmax=1)
        ax[1, 0].set_title(title + '$\phi_{{{}}}(x)$'.format(1) + '\n $\lambda=$' + str(round(np.real(E[0]), 2)),fontsize=FONT_SIZE)
        ax[1, 0].set_ylabel(y0_title  + '$x_2$', fontsize=FONT_SIZE)
        ax[1, 0].set_yticks([-8, 0, 8])
        ax[1, 0].set_yticklabels([-8, 0, 8], fontsize=FONT_SIZE)

        ax[1, 1].pcolor(X1, X2, PHI[:, :, 3] / np.max(np.abs(PHI[:, :, 3])), cmap='rainbow', vmin=-1, vmax=1)
        ax[1, 1].set_title(title + '$\phi_{{{},{}}}(x)$'.format(2,3) + '\n $\lambda=$' + str(round(np.real(E[3]), 2)) + r'$\pm$' + 'j' + str(round(np.imag(E[3]), 2)),fontsize=FONT_SIZE)

        ax[1, 2].pcolor(X1, X2, PHI[:, :, 5] / np.max(np.abs(PHI[:, :, 5])), cmap='rainbow', vmin=-1, vmax=1)
        ax[1, 2].set_title(title + '$\phi_{{{},{}}}(x)$'.format(4,5) + '\n $\lambda=$' + str(round(np.real(E[5]), 2)) + r'$\pm$' + 'j' + str(
            round(np.imag(E[5]), 2)), fontsize=FONT_SIZE)

        ax[1, 3].pcolor(X1, X2, PHI[:, :, 8] / np.max(np.abs(PHI[:, :, 8])), cmap='rainbow', vmin=-1, vmax=1)
        ax[1, 3].set_title(title + '$\phi_{{{}}}(x)$'.format(6) + '\n $\lambda=$' + str(round(np.real(E[8]), 2)),fontsize=FONT_SIZE)

        ax[1, 4].pcolor(X1, X2, PHI[:, :, 1] / np.max(np.abs(PHI[:, :, 1])), cmap='rainbow', vmin=-1, vmax=1)
        ax[1, 4].set_title(title + '$\phi_{{{},{}}}(x)$'.format(7, 8) + '\n $\lambda=$' + str(
            round(np.real(E[1]), 2)) + r'$\pm$' + 'j' + str(
            round(np.imag(E[1]), 2)), fontsize=FONT_SIZE)
        ax[1, 5].pcolor(X1, X2, PHI[:, :, 7] / np.max(np.abs(PHI[:, :, 7])), cmap='rainbow', vmin=-1, vmax=1)
        ax[1, 5].set_title(title + '$\phi_{{{}}}(x)$'.format(9) + '\n $\lambda=$' + str(round(np.real(E[7]), 2)),
                           fontsize=FONT_SIZE)
        ax[1, 6].pcolor(X1, X2, H / np.max(np.abs(H)), cmap='rainbow', vmin=-1, vmax=1)
        ax[1, 6].set_title(title + '$h(x)$' + '\n', fontsize=FONT_SIZE)
    elif row_i == 2:
        c = ax[2, 0].pcolor(X1, X2, PHI[:, :, 0] / np.max(np.abs(PHI[:, :, 0])), cmap='rainbow', vmin=-1, vmax=1)
        ax[2, 0].set_title(title + '$\phi_{{{}}}(x)$'.format(1) + '\n $\lambda=$' + str(round(np.real(E[0]), 2)),
                           fontsize=FONT_SIZE)
        ax[2, 0].set_ylabel(y0_title, fontsize=FONT_SIZE)
        ax[2, 0].set_yticks([-8, 0, 8])
        ax[2, 0].set_yticklabels([-8, 0, 8], fontsize=FONT_SIZE)

        ax[2, 1].pcolor(X1, X2, PHI[:, :, 3] / np.max(np.abs(PHI[:, :, 3])), cmap='rainbow', vmin=-1, vmax=1)
        ax[2, 1].set_title(title + '$\phi_{{{},{}}}(x)$'.format(2, 3) + '\n $\lambda=$' + str(
            round(np.real(E[3]), 2)) + r'$\pm$' + 'j' + str(round(np.imag(E[3]), 2)), fontsize=FONT_SIZE)

        ax[2, 2].pcolor(X1, X2, PHI[:, :, 5] / np.max(np.abs(PHI[:, :, 5])), cmap='rainbow', vmin=-1, vmax=1)
        ax[2, 2].set_title(title + '$\phi_{{{},{}}}(x)$'.format(4, 5) + '\n $\lambda=$' + str(
            round(np.real(E[5]), 2)) + r'$\pm$' + 'j' + str(
            round(np.imag(E[5]), 2)), fontsize=FONT_SIZE)

        ax[2, 3].pcolor(X1, X2, PHI[:, :, 8] / np.max(np.abs(PHI[:, :, 8])), cmap='rainbow', vmin=-1, vmax=1)
        ax[2, 3].set_title(title + '$\phi_{{{}}}(x)$'.format(6) + '\n $\lambda=$' + str(round(np.real(E[8]), 2)),
                           fontsize=FONT_SIZE)
        ax[2,3].set_xlabel('$x_1$', fontsize=FONT_SIZE)

        ax[2, 4].pcolor(X1, X2, PHI[:, :, 1] / np.max(np.abs(PHI[:, :, 1])), cmap='rainbow', vmin=-1, vmax=1)
        ax[2, 4].set_title(title + '$\phi_{{{},{}}}(x)$'.format(7, 8) + '\n $\lambda=$' + str(
            round(np.real(E[1]), 2)) + r'$\pm$' + 'j' + str(
            round(np.imag(E[1]), 2)), fontsize=FONT_SIZE)
        ax[2, 5].pcolor(X1, X2, PHI[:, :, 7] / np.max(np.abs(PHI[:, :, 7])), cmap='rainbow', vmin=-1, vmax=1)
        ax[2, 5].set_title(title + '$\phi_{{{}}}(x)$'.format(9) + '\n $\lambda=$' + str(round(np.real(E[7]), 2)),
                           fontsize=FONT_SIZE)
        ax[2, 6].pcolor(X1, X2, H / np.max(np.abs(H)), cmap='rainbow', vmin=-1, vmax=1)
        ax[2, 6].set_title(title + '$h(x)$' + '\n', fontsize=FONT_SIZE)
        for i in range(7):
            ax[2, i].set_xticks([-8, 0, 8])
            ax[2, i].set_xticklabels([-8, 0, 8], fontsize=FONT_SIZE)



cbar = f.colorbar(c, ax=ax.ravel().tolist(), shrink=0.95,  ticks = [-1.,-0.5,0,0.5,1.])
# ax[0,-1].axis('off')
# ax[0,-2].axis('off')
# ax[0,-3].axis('off')
cbar.ax.set_yticklabels([-1.,-0.5,0,0.5,1.], fontsize=FONT_SIZE)
plt.savefig('Plots/eg2_MEMS_eigfunc.png')
f.show()

## ONLY INTERPOLATED INITIAL CONDITIONS

def get_sys_params():
    # System Parameters
    m = 1
    k1_l = 1
    c = 0.4
    k3_nl = 2
    Vs = 5
    d = 3
    sys_params_MEMS_accel = (m,k1_l,c,k3_nl)
    return sys_params_MEMS_accel,Vs,d


# x0_1_vals = np.arange(-0.5,3.6,0.5) # assumed scaled
# x0_2_vals = np.arange(-1,2.1,0.5) # assumed scaled
SUBPLOTS = True
PLOT_INITIAL_CONDITION_AS_RED_DOTS = True
SUBPLOT_LINEWIDTH = 0.1
x0_1_vals =  np.arange(-4,4.1,1) # assumed scaled  np.array([-4.,4])
x0_2_vals =  np.arange(-6,6.1,1) # assumed scaled np.array([-6.,6])
N_STEPS = 10
ARROW_PARAMS = {'head width':0.3,'head length': 0.3, 'alpha': 1, 'arrow length': 0.15}
f,ax = plt.subplots(2,2,sharey=True,sharex=True,figsize=(12,12))

dict_params={}
dict_phase_data ={}
for items in ['Theo','Seq','Deep','nn']:
    dict_phase_data[items] = np.empty(shape=(0,2))
    if items is 'Theo':
        sys_params_MEMS_accel, _, _ = get_sys_params()
        Ts = 0.5
        t_end = (N_STEPS+1)*Ts
        t = np.arange(0, t_end, Ts)
        X0 = np.empty(shape=(0,2))
        for x0_1, x0_2 in itertools.product(x0_1_vals, x0_2_vals):
            X0 = np.concatenate([X0,np.array([[x0_1, x0_2]])],axis=0)
            x0_unscaled = oc.inverse_transform_X(np.array([x0_1, x0_2]), SYS_NO)
            x_unscaled = oc.odeint(oc.MEMS_accelerometer, x0_unscaled, t, args=sys_params_MEMS_accel)
            x_scaled = oc.scale_data_using_existing_scaler_folder({'X': x_unscaled}, SYS_NO)['X']
            dict_phase_data[items] = np.concatenate([dict_phase_data[items] , x_scaled],axis=0)
            ax[0,0].plot(x_scaled[:, 0], x_scaled[:, 1],color = colors[8], linewidth=SUBPLOT_LINEWIDTH)
            dx = (x_scaled[1, 0] - x_scaled[0, 0]) * ARROW_PARAMS['arrow length']
            dy = (x_scaled[1, 1] - x_scaled[0, 1]) * ARROW_PARAMS['arrow length']
            ax[0, 0].arrow(x_scaled[0, 0], x_scaled[0, 1], dx, dy, head_width=ARROW_PARAMS['head width'],
                           head_length=ARROW_PARAMS['head length'], alpha=ARROW_PARAMS['alpha'], color=colors[8])#'tab:green')
            ax[0,0].set_xticks([-4,0,4])
            ax[0,0].set_xlim([-5, 5])
            ax[0,0].set_yticks([-4, 0, 4])
            ax[0,0].set_ylim([-6.5, 6.5])
            ax[0,0].set_title('Simulated System\n')
            ax[0,0].set_ylabel('$x_2$')
            # ax[0,0].set_xlabel('$x_1$')
            # break
        if PLOT_INITIAL_CONDITION_AS_RED_DOTS:
            ax[0,0].plot(X0[:,0], X0[:,1], 'o', color='salmon',fillstyle='none', markersize=3)

    elif items in ['nn']:
        sess_temp = tf.InteractiveSession()
        dict_params[items] = dn.get_all_run_info(SYS_NO, RUN_NN, sess_temp)
        for x0_1, x0_2 in itertools.product(x0_1_vals, x0_2_vals):
            x_scaled = np.array([[x0_1,x0_2]])
            for i in range(N_STEPS):
                x_scaled = np.concatenate([x_scaled,dict_params[items]['f'].eval(feed_dict={dict_params[items]['xp_feed']:x_scaled[-1:]})],axis=0)
            dict_phase_data[items] = np.concatenate([dict_phase_data[items], x_scaled], axis=0)
            ax[0,1].plot(x_scaled[:, 0], x_scaled[:, 1],color = colors[2], linewidth=SUBPLOT_LINEWIDTH)
            dx = (x_scaled[1, 0] - x_scaled[0, 0]) * ARROW_PARAMS['arrow length']
            dy = (x_scaled[1, 1] - x_scaled[0, 1]) * ARROW_PARAMS['arrow length']
            ax[0, 1].arrow(x_scaled[0, 0], x_scaled[0, 1], dx, dy, head_width=ARROW_PARAMS['head width'],
                           head_length=ARROW_PARAMS['head length'], alpha=ARROW_PARAMS['alpha'],color=colors[2])  # 'tab:green')
            ax[0,1].set_xticks([-4,0,4])
            ax[0,1].set_xlim([-5, 5])
            ax[0,1].set_yticks([-4, 0, 4])
            ax[0,1].set_ylim([-6.5, 6.5])
            ax[0,1].set_title('Neural network \n $\\rho =$')
            # ax[0,1].set_ylabel('$x_2$')
            # ax[0,0].set_xlabel('$x_1$')
            ax[0, 1].plot(X0[:, 0], X0[:, 1], 'o', color='salmon', fillstyle='none', markersize=3)
        tf.reset_default_graph()
        sess_temp.close()
    else:
        if items == 'Deep':
            run_folder = run_folder_name_DIR_ocDEEPDMD
        elif items == 'Seq':
            run_folder = run_folder_name_SEQ_ocDEEPDMD
        sess_temp = tf.InteractiveSession()
        dict_params[items] = get_dict_param(run_folder,SYS_NO,sess_temp)
        for x0_1,x0_2 in itertools.product(x0_1_vals,x0_2_vals):
            psi_x = dict_params[items]['psixpT'].eval(feed_dict = {dict_params[items]['xpT_feed']: np.array([[x0_1,x0_2]])})
            for i in range(N_STEPS):
                psi_x = np.concatenate([psi_x, np.matmul(psi_x[-1:],dict_params[items]['KxT_num'])])
            dict_phase_data[items] = np.concatenate([dict_phase_data[items], psi_x[:, 0:2]], axis=0)
            if items == 'Deep':
                ax[1,0].plot(psi_x[:, 0], psi_x[:, 1], color=colors[5], linewidth=SUBPLOT_LINEWIDTH)
                dx = (psi_x[1, 0] - psi_x[0, 0]) * ARROW_PARAMS['arrow length']
                dy = (psi_x[1, 1] - psi_x[0, 1]) * ARROW_PARAMS['arrow length']
                ax[1, 0].arrow(psi_x[0, 0], psi_x[0, 1], dx, dy, head_width=ARROW_PARAMS['head width'],
                               head_length=ARROW_PARAMS['head length'], alpha=ARROW_PARAMS['alpha'],color=colors[5])  # 'tab:green')
                ax[1,0].set_xticks([-4, 0, 4])
                ax[1,0].set_xlim([-5, 5])
                ax[1,0].set_yticks([-4, 0, 4])
                ax[1,0].set_ylim([-6.5, 6.5])
                ax[1,0].set_ylabel('$x_2$')
                ax[1,0].set_xlabel('$x_1$')
            elif items == 'Seq':
                ax[1,1].plot(psi_x[:, 0], psi_x[:, 1], color=colors[7], linewidth=SUBPLOT_LINEWIDTH)
                dx = (psi_x[1, 0] - psi_x[0, 0]) * ARROW_PARAMS['arrow length']
                dy = (psi_x[1, 1] - psi_x[0, 1]) * ARROW_PARAMS['arrow length']
                ax[1, 1].arrow(psi_x[0, 0], psi_x[0, 1], dx, dy, head_width=ARROW_PARAMS['head width'],
                               head_length=ARROW_PARAMS['head length'], alpha=ARROW_PARAMS['alpha'], color=colors[7])
                # ax[1,1].plot(X0[:, 0], X0[:, 1], 'o', color='salmon', fillstyle='none', markersize=5)
                ax[1,1].set_xticks([-4, 0, 4])
                ax[1,1].set_xlim([-5, 5])
                ax[1,1].set_yticks([-4, 0, 4])
                ax[1,1].set_ylim([-6.5, 6.5])
                ax[1,1].set_title('Sequential ocdeepDMD')
                ax[1,1].set_xlabel('$x_1$')
            # break
        print('=========================')
        tf.reset_default_graph()
        sess_temp.close()
# plt.xlim([-0.5,4])
# plt.ylim([-0.5,2.5])
ax[1,0].plot(X0[:, 0], X0[:, 1], 'o', color='salmon', fillstyle='none', markersize=3)
ax[1,1].plot(X0[:, 0], X0[:, 1], 'o', color='salmon', fillstyle='none', markersize=3)
ax[0,1].set_title('Neural network \n $r^2 =$ ' + str(round(r2_score(dict_phase_data['Theo'].reshape(-1),dict_phase_data['nn'].reshape(-1)),3)))
ax[1,0].set_title('Direct ocdeepDMD \n $r^2 =$ ' + str(round(r2_score(dict_phase_data['Theo'].reshape(-1),dict_phase_data['Deep'].reshape(-1)),3)))
ax[1,1].set_title('Sequential ocdeepDMD \n $r^2 =$ ' +str(round(r2_score(dict_phase_data['Theo'].reshape(-1),dict_phase_data['Seq'].reshape(-1)),3)))
# plt.savefig('MEMS_phase_portrait.png')
plt.show()


print(corr(dict_phase_data['Theo'].reshape(-1),dict_phase_data['nn'].reshape(-1))[0])
print(corr(dict_phase_data['Theo'].reshape(-1),dict_phase_data['Deep'].reshape(-1))[0])
print(corr(dict_phase_data['Theo'].reshape(-1),dict_phase_data['Seq'].reshape(-1))[0])
tf.reset_default_graph()
sess_temp.close()
## OUTPUT S


sampling_resolution = 0.5
x1 = np.arange(-4, 4 + sampling_resolution, sampling_resolution)
x2 = np.arange(-6, 6 + sampling_resolution, sampling_resolution)
X1, X2 = np.meshgrid(x1, x2)

f,ax = plt.subplots(2,2,sharey=True,sharex=True,figsize=(9,9))

dict_output_data ={}
for items in ['Theo','Seq','Deep','nn']:
    sess_temp = tf.InteractiveSession()
    dict_output_data[items] = np.empty(shape=(0,2))
    if items is 'Theo':
        _, Vs, d = get_sys_params()
        Ts = 0.5
        t_end = (N_STEPS+1)*Ts
        t = np.arange(0, t_end, Ts)
        Y_theo = np.zeros(shape=(X1.shape[0], X1.shape[1]))
        for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
            x1_i = X1[i, j]
            x2_i = X2[i, j]
            x0_unscaled = oc.inverse_transform_X(np.array([x1_i, x2_i]), SYS_NO)
            Y_theo[i,j] = -x0_unscaled[0] * Vs / (d + x0_unscaled[0])
        ax[0,0].pcolor(X1, X2, Y_theo, cmap='rainbow')
        ax[0,0].set_xticks([-4,0,4])
        ax[0,0].set_xlim([-4, 4])
        ax[0,0].set_yticks([-4, 0, 4])
        ax[0,0].set_ylim([-6, 6])
        ax[0,0].set_title('Simulated \n System')
        ax[0,0].set_ylabel('$x_2$')
    elif items in ['nn']:
        dict_params[items] = dn.get_all_run_info(SYS_NO, RUN_NN, sess_temp)
        Y_nn = np.zeros(shape=(X1.shape[0], X1.shape[1]))
        for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
            x1_i = X1[i, j]
            x2_i = X2[i, j]
            x0_scaled = np.array([[x1_i, x2_i]])
            Y_nn[i,j] = dict_params[items]['g'].eval(feed_dict={dict_params[items]['xp_feed']:x0_scaled})
        ax[0, 1].pcolor(X1, X2, Y_nn, cmap='rainbow')
        ax[0,1].set_xticks([-4,0,4])
        ax[0,1].set_xlim([-4, 4])
        ax[0,1].set_yticks([-4, 0, 4])
        ax[0,1].set_ylim([-6, 6])
    elif items in ['Deep']:
        dict_params[items] = get_dict_param(run_folder_name_DIR_ocDEEPDMD,SYS_NO,sess_temp)
        Y_deep = np.zeros(shape=(X1.shape[0], X1.shape[1]))
        for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
            x1_i = X1[i, j]
            x2_i = X2[i, j]
            x0_scaled = np.array([[x1_i, x2_i]])
            psi_x = dict_params[items]['psixpT'].eval(feed_dict = {dict_params[items]['xpT_feed']: x0_scaled})
            Y_deep[i,j] = np.matmul(psi_x,dict_params[items]['WhT_num'])
        ax[1,0].pcolor(X1, X2, Y_deep, cmap='rainbow')
        ax[1,0].set_xticks([-4,0,4])
        ax[1,0].set_xlim([-4, 4])
        ax[1,0].set_yticks([-4, 0, 4])
        ax[1,0].set_ylim([-6, 6])
        ax[1, 0].set_xlabel('$x_1$')
        ax[1, 0].set_ylabel('$x_2$')
    elif items in ['Seq']:
        dict_params[items] = get_dict_param(run_folder_name_SEQ_ocDEEPDMD,SYS_NO,sess_temp)
        Y_seq = np.zeros(shape=(X1.shape[0], X1.shape[1]))
        for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
            x1_i = X1[i, j]
            x2_i = X2[i, j]
            x0_scaled = np.array([[x1_i, x2_i]])
            psi_x = dict_params[items]['psixpT'].eval(feed_dict={dict_params[items]['xpT_feed']: x0_scaled})
            Y_seq[i,j] = np.matmul(psi_x,dict_params[items]['WhT_num'])
        ax[1,1].pcolor(X1, X2, Y_seq, cmap='rainbow')
        ax[1,1].set_xticks([-4,0,4])
        ax[1,1].set_xlim([-4, 4])
        ax[1,1].set_yticks([-4, 0, 4])
        ax[1,1].set_ylim([-6, 6])
        ax[1,1].set_xlabel('$x_1$')
    tf.reset_default_graph()
    sess_temp.close()

ax[0,1].set_title('Neural network \n $\\rho =$ ' + str(round(corr(Y_theo.reshape(-1),Y_nn.reshape(-1))[0],3)))
ax[1,0].set_title('Direct ocdeepDMD \n $\\rho =$ ' + str(round(corr(Y_theo.reshape(-1),Y_deep.reshape(-1))[0],3)))
ax[1,1].set_title('Sequential ocdeepDMD \n $\\rho =$ ' +str(round(corr(Y_theo.reshape(-1),Y_seq.reshape(-1))[0],3)))

plt.show()

print(corr(Y_theo.reshape(-1),Y_nn.reshape(-1))[0])
print(corr(Y_theo.reshape(-1),Y_deep.reshape(-1))[0])
print(corr(Y_theo.reshape(-1),Y_seq.reshape(-1))[0])






## OUTPUT STUFF
# Wh_SEQ = np.matmul(dict_params['Seq']['WhT_num'].T,koop_modes_SEQ)
Wh_DEEP = np.matmul(dict_params['Deep']['WhT_num'].T,koop_modes_DEEPDMD)
# Wh_SEQ = Wh_SEQ/ np.max(np.abs(Wh_SEQ))
Wh_DEEP = Wh_DEEP/ np.max(np.abs(Wh_DEEP))
# Wh_SEQ = np.abs(Wh_SEQ)
Wh_DEEP =np.abs(Wh_DEEP)
fig,ax = plt.subplots()
# ax = fig.add_axes([0,0,1,1])
# X = np.arange(1,1+len(Wh_SEQ[0]))
X = np.arange(1,len(Wh_DEEP[0])+1)
# x_labels = [str(round(items.real,2) + round(items.imag,2) *1j) for items in np.diag(E_SEQ)]
# ax.bar(X + 0.00,Wh_SEQ.reshape(-1),color='b',label = 'Sequential ocdeepDMD', width = 0.25)
ax.bar(X + 0.,Wh_DEEP.reshape(-1),color='g',label = 'Direct ocdeepDMD', width = 0.25)
ax.set_xticks(X+0.)
ax.set_xticklabels(X)
# ax.set_xticklabels(x_labels)
ax.set_xlabel('Eigenvalue $\lambda$')
ax.set_ylabel('Relative $W_h$ value')
ax.legend()
plt.show()


## EXTRAPOLATED INITIAL CONDITIONS


# def get_sys_params():
#     # System Parameters
#     m = 1
#     k1_l = 1
#     c = 0.4
#     k3_nl = 2
#     Vs = 5
#     d = 3
#     sys_params_MEMS_accel = (m,k1_l,c,k3_nl)
#     return sys_params_MEMS_accel,Vs,d
#
#
# # x0_1_vals = np.arange(-0.5,3.6,0.5) # assumed scaled
# # x0_2_vals = np.arange(-1,2.1,0.5) # assumed scaled
# SUBPLOTS = True
# SUBPLOT_LINEWIDTH = 0.7
# x0_1_vals =  np.arange(-4,4.1,2) # assumed scaled  np.array([-4.,4])
# x0_2_vals =  np.arange(-6,6.1,3) # assumed scaled np.array([-6.,6])
# N_STEPS = 10
# if SUBPLOTS:
#     f,ax = plt.subplots(2,4,sharey=True,sharex=True,figsize=(18,9))
# else:
#     plt.figure()
#
# dict_phase_data ={}
# for items in ['Theo','Seq','Deep','nn']:
#     dict_phase_data[items] = np.empty(shape=(0,2))
#     if items is 'Theo':
#         sys_params_MEMS_accel, _, _ = get_sys_params()
#         Ts = 0.5
#         t_end = (N_STEPS+1)*Ts
#         t = np.arange(0, t_end, Ts)
#         X0 = np.empty(shape=(0,2))
#         for x0_1, x0_2 in itertools.product(x0_1_vals, x0_2_vals):
#             X0 = np.concatenate([X0,np.array([[x0_1, x0_2]])],axis=0)
#             x0_unscaled = oc.inverse_transform_X(np.array([x0_1, x0_2]), SYS_NO)
#             x_unscaled = oc.odeint(oc.MEMS_accelerometer, x0_unscaled, t, args=sys_params_MEMS_accel)
#             x_scaled = oc.scale_data_using_existing_scaler_folder({'X': x_unscaled}, SYS_NO)['X']
#             dict_phase_data[items] = np.concatenate([dict_phase_data[items] , x_scaled],axis=0)
#             if SUBPLOTS:
#                 ax[0,0].plot(x_scaled[:, 0], x_scaled[:, 1],color = colors[8], linewidth=SUBPLOT_LINEWIDTH)
#                 # ax[0,0].set_xticks([-4,0,4])
#                 # ax[0,0].set_xlim([-5, 5])
#                 # ax[0,0].set_yticks([-4, 0, 4])
#                 # ax[0,0].set_ylim([-6.5, 6.5])
#                 ax[0,0].set_title('Simulated \n System')
#                 ax[0,0].set_ylabel('$x_2$')
#                 # ax[0,0].set_xlabel('$x_1$')
#             else:
#                 plt.plot(x_scaled[:, 0], x_scaled[:, 1],color = colors[0])
#             # break
#         if SUBPLOTS:
#             ax[0,0].plot(X0[:,0], X0[:,1], 'o', color='salmon',fillstyle='none', markersize=5)
#         else:
#             plt.plot(X0[:,0], X0[:,1], 'o', color='salmon',fillstyle='none', markersize=5)
#     elif items in ['nn']:
#         sess_temp = tf.InteractiveSession()
#         dict_params[items] = dn.get_all_run_info(SYS_NO, RUN_NN, sess_temp)
#         for x0_1, x0_2 in itertools.product(x0_1_vals, x0_2_vals):
#             x_scaled = np.array([[x0_1,x0_2]])
#             for i in range(N_STEPS):
#                 x_scaled = np.concatenate([x_scaled,dict_params[items]['f'].eval(feed_dict={dict_params[items]['xp_feed']:x_scaled[-1:]})],axis=0)
#             dict_phase_data[items] = np.concatenate([dict_phase_data[items], x_scaled], axis=0)
#             if SUBPLOTS:
#                 ax[0,1].plot(x_scaled[:, 0], x_scaled[:, 1],color = colors[2], linewidth=SUBPLOT_LINEWIDTH)
#                 # ax[0,1].set_xticks([-4,0,4])
#                 # ax[0,1].set_xlim([-5, 5])
#                 # ax[0,1].set_yticks([-4, 0, 4])
#                 # ax[0,1].set_ylim([-6.5, 6.5])
#                 # ax[0,1].set_ylabel('$x_2$')
#                 # ax[0,0].set_xlabel('$x_1$')
#                 ax[0, 1].plot(X0[:, 0], X0[:, 1], 'o', color='salmon', fillstyle='none', markersize=5)
#             else:
#                 plt.plot(x_scaled[:, 0], x_scaled[:, 1],color = colors[0])
#         tf.reset_default_graph()
#         sess_temp.close()
#     else:
#         if items == 'Deep':
#             run_folder = run_folder_name_DIR_ocDEEPDMD
#         elif items == 'Seq':
#             run_folder = run_folder_name_SEQ_ocDEEPDMD
#         sess_temp = tf.InteractiveSession()
#         dict_params[items] = get_dict_param(run_folder,SYS_NO,sess_temp)
#         for x0_1,x0_2 in itertools.product(x0_1_vals,x0_2_vals):
#             psi_x = dict_params[items]['psixpT'].eval(feed_dict = {dict_params[items]['xpT_feed']: np.array([[x0_1,x0_2]])})
#             for i in range(N_STEPS):
#                 psi_x = np.concatenate([psi_x, np.matmul(psi_x[-1:],dict_params[items]['KxT_num'])])
#             dict_phase_data[items] = np.concatenate([dict_phase_data[items], psi_x[:, 0:2]], axis=0)
#             if SUBPLOTS:
#                 if items == 'Deep':
#                     ax[0,2].plot(psi_x[:, 0], psi_x[:, 1], color=colors[5], linewidth=SUBPLOT_LINEWIDTH)
#                     # ax[1,0].set_xticks([-4, 0, 4])
#                     # ax[1,0].set_xlim([-5, 5])
#                     # ax[1,0].set_yticks([-4, 0, 4])
#                     # ax[1,0].set_ylim([-6.5, 6.5])
#                     # ax[0,2].set_ylabel('$x_2$')
#                     # ax[0,2].set_xlabel('$x_1$')
#                 elif items == 'Seq':
#                     ax[0,3].plot(psi_x[:, 0], psi_x[:, 1], color=colors[7], linewidth=SUBPLOT_LINEWIDTH)
#                     # ax[1,1].plot(X0[:, 0], X0[:, 1], 'o', color='salmon', fillstyle='none', markersize=5)
#                     # ax[1,1].set_xticks([-4, 0, 4])
#                     # ax[1,1].set_xlim([-5, 5])
#                     # ax[1,1].set_yticks([-4, 0, 4])
#                     # ax[1,1].set_ylim([-6.5, 6.5])
#                     # ax[0,3].set_title('Sequential ocdeepDMD')
#                     # ax[0,3].set_xlabel('$x_1$')
#             else:
#                 if items == 'Deep':
#                     plt.plot(psi_x[:, 0], psi_x[:, 1], color=colors[4], linewidth=0.5)
#                 elif items == 'Seq':
#                     plt.plot(psi_x[:, 0], psi_x[:, 1], color=colors[7], linewidth=1)
#             # break
#         print('=========================')
#         tf.reset_default_graph()
#         sess_temp.close()
# # plt.xlim([-0.5,4])
# # plt.ylim([-0.5,2.5])
# ax[0,2].plot(X0[:, 0], X0[:, 1], 'o', color='salmon', fillstyle='none', markersize=5)
# ax[0,3].plot(X0[:, 0], X0[:, 1], 'o', color='salmon', fillstyle='none', markersize=5)
# ax[0,1].set_title('Neural network \n $\\rho =$ ' + str(round(corr(dict_phase_data['Theo'].reshape(-1),dict_phase_data['nn'].reshape(-1))[0],3)))
# ax[0,2].set_title('Direct ocdeepDMD \n $\\rho =$ ' + str(round(corr(dict_phase_data['Theo'].reshape(-1),dict_phase_data['Deep'].reshape(-1))[0],3)))
# ax[0,3].set_title('Sequential ocdeepDMD \n $\\rho =$ ' +str(round(corr(dict_phase_data['Theo'].reshape(-1),dict_phase_data['Seq'].reshape(-1))[0],3)))
#
# print('Interpolated Initial Conitions:')
# print(corr(dict_phase_data['Theo'].reshape(-1),dict_phase_data['nn'].reshape(-1))[0])
# print(corr(dict_phase_data['Theo'].reshape(-1),dict_phase_data['Deep'].reshape(-1))[0])
# print(corr(dict_phase_data['Theo'].reshape(-1),dict_phase_data['Seq'].reshape(-1))[0])
#
# X_LIM = [np.floor(np.min(dict_phase_data['Theo'][:,0])), np.ceil(np.max(dict_phase_data['Theo'][:,0]))]
# Y_LIM = [np.floor(np.min(dict_phase_data['Theo'][:,1])), np.ceil(np.max(dict_phase_data['Theo'][:,1]))]
# X_TICKS = np.arange(X_LIM[0]+1,X_LIM[1],1)
# Y_TICKS = np.arange(Y_LIM[0]+1,Y_LIM[1],1)
#
#
# x0_1_vals =  list(set(np.arange(-10,-4.1,2)).union(set(np.arange(2,10.1,2)))) # np.arange(-4,4.1,2) # assumed scaled  np.array([-4.,4])
# x0_2_vals =  list(set(np.arange(-10,-6,2)).union(set(np.arange(6,10.1,2)))) #np.arange(-6,6.1,3) # assumed scaled np.array([-6.,6])
#
# dict_params = {}
# dict_phase_data_e ={}
# for items in ['Theo','Seq','Deep','nn']:
#     dict_phase_data_e[items] = np.empty(shape=(0,2))
#     if items is 'Theo':
#         sys_params_MEMS_accel, _, _ = get_sys_params()
#         Ts = 0.5
#         t_end = (N_STEPS+1)*Ts
#         t = np.arange(0, t_end, Ts)
#         X0 = np.empty(shape=(0,2))
#         for x0_1, x0_2 in itertools.product(x0_1_vals, x0_2_vals):
#             X0 = np.concatenate([X0,np.array([[x0_1, x0_2]])],axis=0)
#             x0_unscaled = oc.inverse_transform_X(np.array([x0_1, x0_2]), SYS_NO)
#             x_unscaled = oc.odeint(oc.MEMS_accelerometer, x0_unscaled, t, args=sys_params_MEMS_accel)
#             x_scaled = oc.scale_data_using_existing_scaler_folder({'X': x_unscaled}, SYS_NO)['X']
#             dict_phase_data_e[items] = np.concatenate([dict_phase_data_e[items] , x_scaled],axis=0)
#             if SUBPLOTS:
#                 ax[1,0].plot(x_scaled[:, 0], x_scaled[:, 1],color = colors[8], linewidth=SUBPLOT_LINEWIDTH)
#                 # ax[0,0].set_xticks([-4,0,4])
#                 # ax[0,0].set_xlim([-5, 5])
#                 # ax[0,0].set_yticks([-4, 0, 4])
#                 # ax[0,0].set_ylim([-6.5, 6.5])
#                 ax[1,0].set_title('Simulated \n System')
#                 ax[1,0].set_ylabel('$x_2$')
#                 ax[1,0].set_xlabel('$x_1$')
#             else:
#                 plt.plot(x_scaled[:, 0], x_scaled[:, 1],color = colors[0])
#             # break
#         if SUBPLOTS:
#             ax[1,0].plot(X0[:,0], X0[:,1], 'o', color='salmon',fillstyle='none', markersize=5)
#         else:
#             plt.plot(X0[:,0], X0[:,1], 'o', color='salmon',fillstyle='none', markersize=5)
#     elif items in ['nn']:
#         sess_temp = tf.InteractiveSession()
#         dict_params[items] = dn.get_all_run_info(SYS_NO, RUN_NN, sess_temp)
#         for x0_1, x0_2 in itertools.product(x0_1_vals, x0_2_vals):
#             x_scaled = np.array([[x0_1,x0_2]])
#             for i in range(N_STEPS):
#                 x_scaled = np.concatenate([x_scaled,dict_params[items]['f'].eval(feed_dict={dict_params[items]['xp_feed']:x_scaled[-1:]})],axis=0)
#             dict_phase_data_e[items] = np.concatenate([dict_phase_data_e[items], x_scaled], axis=0)
#             if SUBPLOTS:
#                 ax[1,1].plot(x_scaled[:, 0], x_scaled[:, 1],color = colors[2], linewidth=SUBPLOT_LINEWIDTH)
#                 # ax[0,1].set_xticks([-4,0,4])
#                 # ax[0,1].set_xlim([-5, 5])
#                 # ax[0,1].set_yticks([-4, 0, 4])
#                 # ax[0,1].set_ylim([-6.5, 6.5])
#                 ax[1,1].set_title('Neural network \n $\\rho =$')
#                 # ax[0,1].set_ylabel('$x_2$')
#                 ax[1,1].set_xlabel('$x_1$')
#                 ax[1, 1].plot(X0[:, 0], X0[:, 1], 'o', color='salmon', fillstyle='none', markersize=5)
#             else:
#                 plt.plot(x_scaled[:, 0], x_scaled[:, 1],color = colors[0])
#         tf.reset_default_graph()
#         sess_temp.close()
#     else:
#         if items == 'Deep':
#             run_folder = run_folder_name_DIR_ocDEEPDMD
#         elif items == 'Seq':
#             run_folder = run_folder_name_SEQ_ocDEEPDMD
#         sess_temp = tf.InteractiveSession()
#         dict_params[items] = get_dict_param(run_folder,SYS_NO,sess_temp)
#         for x0_1,x0_2 in itertools.product(x0_1_vals,x0_2_vals):
#             psi_x = dict_params[items]['psixpT'].eval(feed_dict = {dict_params[items]['xpT_feed']: np.array([[x0_1,x0_2]])})
#             for i in range(N_STEPS):
#                 psi_x = np.concatenate([psi_x, np.matmul(psi_x[-1:],dict_params[items]['KxT_num'])])
#             dict_phase_data_e[items] = np.concatenate([dict_phase_data_e[items], psi_x[:, 0:2]], axis=0)
#             if SUBPLOTS:
#                 if items == 'Deep':
#                     ax[1,2].plot(psi_x[:, 0], psi_x[:, 1], color=colors[5], linewidth=SUBPLOT_LINEWIDTH)
#                     # ax[1,0].set_xticks([-4, 0, 4])
#                     # ax[1,0].set_xlim([-5, 5])
#                     # ax[1,0].set_yticks([-4, 0, 4])
#                     # ax[1,0].set_ylim([-6.5, 6.5])
#                     # ax[1,2].set_ylabel('$x_2$')
#                     ax[1,2].set_xlabel('$x_1$')
#                 elif items == 'Seq':
#                     ax[1,3].plot(psi_x[:, 0], psi_x[:, 1], color=colors[7], linewidth=SUBPLOT_LINEWIDTH)
#                     # ax[1,1].plot(X0[:, 0], X0[:, 1], 'o', color='salmon', fillstyle='none', markersize=5)
#                     # ax[1,1].set_xticks([-4, 0, 4])
#                     # ax[1,1].set_xlim([-5, 5])
#                     # ax[1,1].set_yticks([-4, 0, 4])
#                     # ax[1,1].set_ylim([-6.5, 6.5])
#                     # ax[1,3].set_title('Sequential ocdeepDMD')
#                     ax[1,3].set_xlabel('$x_1$')
#             else:
#                 if items == 'Deep':
#                     plt.plot(psi_x[:, 0], psi_x[:, 1], color=colors[4], linewidth=0.5)
#                 elif items == 'Seq':
#                     plt.plot(psi_x[:, 0], psi_x[:, 1], color=colors[7], linewidth=1)
#             # break
#         print('=========================')
#         tf.reset_default_graph()
#         sess_temp.close()
# # plt.xlim([-0.5,4])
# # plt.ylim([-0.5,2.5])
# ax[1,2].plot(X0[:, 0], X0[:, 1], 'o', color='salmon', fillstyle='none', markersize=5)
# ax[1,3].plot(X0[:, 0], X0[:, 1], 'o', color='salmon', fillstyle='none', markersize=5)
# ax[1,1].set_title('Neural network \n $\\rho =$ ' + str(round(corr(dict_phase_data_e['Theo'].reshape(-1),dict_phase_data_e['nn'].reshape(-1))[0],3)))
# ax[1,2].set_title('Direct ocdeepDMD \n $\\rho =$ ' + str(round(corr(dict_phase_data_e['Theo'].reshape(-1),dict_phase_data_e['Deep'].reshape(-1))[0],3)))
# ax[1,3].set_title('Sequential ocdeepDMD \n $\\rho =$ ' +str(round(corr(dict_phase_data_e['Theo'].reshape(-1),dict_phase_data_e['Seq'].reshape(-1))[0],3)))
#
#
# X_LIM1 = [np.min([ X_LIM[0],np.floor(np.min(dict_phase_data_e['Theo'][:,0]))]), np.max([ X_LIM[1],np.ceil(np.max(dict_phase_data_e['Theo'][:,0]))])]
# Y_LIM1 = [np.min([ Y_LIM[0],np.floor(np.min(dict_phase_data_e['Theo'][:,1]))]), np.max([ Y_LIM[1],np.ceil(np.max(dict_phase_data_e['Theo'][:,1]))])]
# X_TICKS1 = np.arange(X_LIM1[0]+1,X_LIM1[1],5)
# Y_TICKS1 = np.arange(Y_LIM1[0]+1,Y_LIM1[1],5)
# for axes_plot in ax.reshape(-1):
#     axes_plot.set_xticks(X_TICKS1)
#     axes_plot.set_xlim(X_LIM1)
#     axes_plot.set_yticks(Y_TICKS1)
#     axes_plot.set_ylim(Y_LIM1)
#
# plt.show()
#
# print('Extrapolated Initial Conditions')
# print(corr(dict_phase_data_e['Theo'].reshape(-1),dict_phase_data_e['nn'].reshape(-1))[0])
# print(corr(dict_phase_data_e['Theo'].reshape(-1),dict_phase_data_e['Deep'].reshape(-1))[0])
# print(corr(dict_phase_data_e['Theo'].reshape(-1),dict_phase_data_e['Seq'].reshape(-1))[0])



## COMPUTE THE STATISTICS REQUIRED FOR THE TABLE
x_sim_test = np.empty(shape=(0,2))
x_dir_test = np.empty(shape=(0,2))
x_seq_test = np.empty(shape=(0,2))
x_nn_test = np.empty(shape=(0,2))
y_sim_test = np.empty(shape=(0,1))
y_dir_test = np.empty(shape=(0,1))
y_seq_test = np.empty(shape=(0,1))
y_nn_test = np.empty(shape=(0,1))
for i in range(200,300):
    x_sim_test = np.concatenate([x_sim_test,d_DDMD[i]['X']],axis=0)
    x_nn_test = np.concatenate([x_nn_test,d_NN[i]['X_one_step']],axis=0)
    x_dir_test = np.concatenate([x_dir_test, d_DDMD[i]['X_one_step']], axis=0)
    x_seq_test = np.concatenate([x_seq_test, d_SEQ[i]['X_est_one_step']], axis=0)
    y_sim_test = np.concatenate([y_sim_test, d_DDMD[i]['Y']], axis=0)
    y_nn_test = np.concatenate([y_nn_test, d_NN[i]['Y_one_step']], axis=0)
    y_dir_test = np.concatenate([y_dir_test, d_DDMD[i]['Y_one_step']], axis=0)
    y_seq_test = np.concatenate([y_seq_test, d_SEQ[i]['Y_est_one_step']], axis=0)

print('STATE STATS')
print('Neural Network:',round(r2_score(x_sim_test,x_nn_test)*100,2))
print('Direct ocdeepDMD:',round(r2_score(x_sim_test,x_dir_test)*100,2))
print('Sequential ocdeepDMD:',round(r2_score(x_sim_test,x_seq_test)*100,2))
print('---------------------------------------')
print('OUTPPUT STATS')
print('Neural Network:',round(r2_score(y_sim_test,y_nn_test)*100,2))
print('Direct ocdeepDMD:',round(r2_score(y_sim_test,y_dir_test)*100,2))
print('Sequential ocdeepDMD:',round(r2_score(y_sim_test,y_seq_test)*100,2))
print('---------------------------------------')
##

CURVE_NO = 294
n_states = d_SEQ[CURVE_NO]['X'].shape[1]
n_outputs = d_SEQ[CURVE_NO]['Y'].shape[1]
DOWNSAMPLE =1
TRUTH_MARKER_SIZE = 15
TICK_FONT_SIZE = 12
HEADER_SIZE = 21
FONT_SIZE = 16
pl_max = 0
pl_min = 0
plt.figure(figsize=(7.2,4.8))
for i in range(n_states):
    x_scale = 10**np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['X'][:,i]))))
    # l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1) + (r'$[\times 10^{{{}}}]$').format(np.int(np.log10(x_scale))))
    l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1))
    plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['X']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['X'][0::DOWNSAMPLE,i]/x_scale,'.',color = colors[i],markersize = TRUTH_MARKER_SIZE)
    plt.plot(d_NN[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle=':', color=colors[i],linewidth = 2)
    # plt.plot(d_DDMD_X[CURVE_NO]['X_est_n_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i],linewidth = 2)
    # plt.plot(d_DDMD_SUBOPT[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
    plt.plot(d_DDMD[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle='dashed', color=colors[i])
    plt.plot(d_SEQ[CURVE_NO]['X_est_n_step'][:, i] / x_scale, linestyle='solid', color=colors[i])
    # plt.plot(d_NN[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
    # plt.plot(d_HAM[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
    pl_max = np.max([pl_max,np.max(d_SEQ[CURVE_NO]['X'][:,i]/x_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['X'][:, i] / x_scale)])
for i in range(n_outputs):
    y_scale = 10 ** np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['Y'][:, i]))))
    # plt.plot([], color=colors[n_states+i], label=('$y_{}$').format(i + 1) + (r'$[\times 10^{{{}}}]$').format(np.int(np.log10(y_scale))))
    plt.plot([], color=colors[n_states + i],label=('$y_{}$').format(i + 1))
    plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['Y']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['Y'][0::DOWNSAMPLE,i]/y_scale, '.',color = colors[n_states+i],markersize = TRUTH_MARKER_SIZE)
    # plt.plot(d_DDMD_SUBOPT[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle='dashdot', color=colors[n_states + i])
    plt.plot(d_NN[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle=':', color=colors[n_states + i],linewidth = 2)
    plt.plot(d_DDMD[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle='dashed', color=colors[n_states+i])
    plt.plot(d_SEQ[CURVE_NO]['Y_est_n_step'][:, i] / y_scale, linestyle='solid', color=colors[n_states + i])
    # plt.plot(d_NN[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='dashdot', color=colors[n_states+i])
    # plt.plot(d_HAM[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='dashdot', color=colors[n_states + i])
    pl_max = np.max([pl_max, np.max(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
l1 = plt.legend(loc='upper right',fancybox=True, shadow=True,fontsize = FONT_SIZE,ncol =3,bbox_to_anchor=(0.97,0.97))
# l1 = plt.legend(loc='upper right',fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE)
plt.gca().add_artist(l1)
# l1 = plt.legend(loc='upper center',fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =3)
# plt.xlabel('Time Index(k)',fontsize = FONT_SIZE)
# plt.ylabel('x,y [1 -step]',fontsize = FONT_SIZE)
# plt.text(20,-1,'[1-step]',fontsize = FONT_SIZE)

a0, = plt.plot([],'.',markersize = TRUTH_MARKER_SIZE,label='Truth',color = 'grey')
a2, = plt.plot([], linestyle =':',linewidth = 2,label='Model 1',color = 'grey')
a3, = plt.plot([], linestyle = 'dashed',linewidth = 1,label='Model 2',color = 'grey')
a4, = plt.plot([], linestyle ='solid',linewidth = 1,label='Model 3',color = 'grey')


l2 = plt.legend((a0,a2,a3,a4),('Truth','Nonlinear state-space','Direct OC-deepDMD','Sequential OC-deepDMD'),loc = "lower center",fontsize = FONT_SIZE,ncol=2,bbox_to_anchor=(0.5,1.005))
plt.xlabel('Time (sec)',fontsize = FONT_SIZE)
plt.ylabel('States and Outputs',fontsize = FONT_SIZE)
# plt.text(-6,0,'States and Outputs',rotation = 90,fontsize = FONT_SIZE)
# plt.title('(b)',fontsize = FONT_SIZE)
plt.ylim([-1.5,1.8])
plt.xticks([0,4,8,12,16,20],[np.int(items) for items in (np.array([0,4,8,12,16,20]) * 0.5)],fontsize = TICK_FONT_SIZE)
plt.yticks([-1,0,1],fontsize = TICK_FONT_SIZE)
plt.xlim([-0.1,19.1])

# plt.savefig('Plots/eg1_TheoreticalExample_fit.svg')
# plt.savefig('Plots/MEMS_fit.png',  bbox_inches='tight')
# plt.title('(b)',fontsize = HEADER_SIZE,loc='left')
plt.show()
# plt.subplot2grid((10,16), (8,0), colspan=4, rowspan=2)

## Observability rank
Wh_SEQ = dict_params['Seq']['WhT_num'].T
K_SEQ = dict_params['Seq']['KxT_num'].T
Wh_DIR = dict_params['Deep']['WhT_num'].T
K_DIR = dict_params['Deep']['KxT_num'].T
N = 10

Ob_SEQ = np.empty(shape=(0,len(Wh_SEQ[0])))
for i in range(N):
    Ob_SEQ = np.concatenate([Ob_SEQ,np.matmul(Wh_SEQ,np.linalg.matrix_power(K_SEQ,i))],axis=0)

print('Raw rank of Observability Matrix :', np.linalg.matrix_rank(Ob_SEQ))
_,S,_ = np.linalg.svd(Ob_SEQ)
print('Singular Values: ', S)

S_sig = ( np.cumsum(S**2)/np.sum(S**2))*100
for items in S_sig:
    print(round(items,2))

# Reconstruction of the initial conditions
N_steps = len(d_SEQ[1]['X'])

Ob_SEQ  = np.empty(shape=(0,len(Wh_SEQ[0])))
Ob_DIR  = np.empty(shape=(0,len(Wh_DIR[0])))
for i in range(N_steps):
    Ob_SEQ  = np.concatenate([Ob_SEQ ,np.matmul(Wh_SEQ,np.linalg.matrix_power(K_SEQ,i))],axis=0)
    Ob_DIR = np.concatenate([Ob_DIR, np.matmul(Wh_DIR, np.linalg.matrix_power(K_DIR, i))], axis=0)
Ob_SEQ_pseodoinverse = np.matmul(np.linalg.inv(np.matmul(Ob_SEQ.T,Ob_SEQ)),Ob_SEQ.T)
Ob_DIR_pseodoinverse = np.matmul(np.linalg.inv(np.matmul(Ob_DIR.T,Ob_DIR)),Ob_DIR.T)
psiX0_SEQ_sim = np.empty(shape=(len(d_SEQ[0]['psiX'][0]),0))
psiX0_SEQ = np.empty(shape=(len(d_SEQ[0]['psiX'][0]),0))
psiX0_DIR_sim = np.empty(shape=(len(d_DDMD[0]['psiX'][0]),0))
psiX0_DIR = np.empty(shape=(len(d_DDMD[0]['psiX'][0]),0))
for i in range(300):
    psiX0_SEQ_sim = np.concatenate([psiX0_SEQ_sim,d_SEQ[i]['psiX'][0:1].T],axis=1)
    psiX0_SEQ = np.concatenate([psiX0_SEQ, np.matmul(Ob_SEQ_pseodoinverse,d_SEQ[i]['Y_est_n_step'])],axis=1)
    psiX0_DIR_sim = np.concatenate([psiX0_DIR_sim, d_DDMD[i]['psiX'][0:1].T],axis=1)
    psiX0_DIR = np.concatenate([psiX0_DIR, np.matmul(Ob_DIR_pseodoinverse, d_DDMD[i]['Y_n_step'])],axis=1)

print('---------------------------')
print('SEQUENTIAL RECONSTRUCTION ACCURACY')
print('---------------------------')
for i in range(len(psiX0_SEQ)):
    print('psi' + str(i+1) + ' :',r2_score(psiX0_SEQ_sim[i],psiX0_SEQ[i]))
print('---------------------------')
print('SEQUENTIAL RECONSTRUCTION ACCURACY')
print('---------------------------')
for i in range(len(psiX0_SEQ)):
    print('psi' + str(i+1) + ' :',r2_score(psiX0_DIR_sim[i],psiX0_DIR[i]))
print('---------------------------')




##

