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

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette


SYS_NO = 11
# RUN_NO = 78
# RUN_NO_HAMMERSTEIN_X = 10
# RUN_NO_HAMMERSTEIN_Y = 23
# RUN_NO_DEEPDMD = 52 #1

RUN_DIRECT_DEEPDMD_SUBOPT = 6
RUN_DIRECT_DEEPDMD = 49#31
RUN_SEQ_DEEPDMD = 78#59
DIR_DEEPDMD_X = 5
# RUN_NN = 4



sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYS_NO)
run_folder_name_DEEPDMD = sys_folder_name + '/Sequential/RUN_' + str(DIR_DEEPDMD_X)
run_folder_name_SEQ_ocDEEPDMD = sys_folder_name + '/Sequential/RUN_' + str(RUN_SEQ_DEEPDMD)
run_folder_name_DIR_ocDEEPDMD = sys_folder_name + '/deepDMD/RUN_' + str(RUN_DIRECT_DEEPDMD)
run_folder_name_DIR_ocDEEPDMD_SUBOPT = sys_folder_name + '/deepDMD/RUN_' + str(RUN_DIRECT_DEEPDMD_SUBOPT)
# run_folder_name_NN = sys_folder_name + '/Direct_nn/RUN_' + str(RUN_NN)

# run_folder_name_HAM_Y = sys_folder_name + '/Hammerstein/RUN_' + str(RUN_NO_HAMMERSTEIN_Y)
# run_folder_name_HAM_X = sys_folder_name + '/Hammerstein/RUN_' + str(RUN_NO_HAMMERSTEIN_X)


with open(sys_folder_name + '/System_' + str(SYS_NO) + '_SimulatedData.pickle', 'rb') as handle:
    dict_data = pickle.load(handle)
with open(sys_folder_name + '/System_' + str(SYS_NO) + '_ocDeepDMDdata.pickle', 'rb') as handle:
    dict_oc_data = pickle.load(handle)
Ntrain = round(len(dict_oc_data['Xp'])/2)
for items in dict_oc_data:
    dict_oc_data[items] = dict_oc_data[items][0:Ntrain]
# with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'rb') as handle:
#     d_SEQ = pickle.load(handle)[RUN_SEQ_DEEPDMD]
with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'rb') as handle:
    d_DDMD_X = pickle.load(handle)[DIR_DEEPDMD_X]
with open(sys_folder_name + '/dict_predictions_deepDMD.pickle', 'rb') as handle:
    d_DDMD = pickle.load(handle)[RUN_DIRECT_DEEPDMD]
# with open(sys_folder_name + '/dict_predictions_Direct_nn.pickle', 'rb') as handle:
#     d_NN = pickle.load(handle)[RUN_NN]

##
ls_steps = list(range(1,15,1))
ls_curves = list(range(160, 240)) # test curves
Senergy_THRESHOLD = 99.99
REDUCED_MODES = False
RIGHT_EIGEN_VECTORS = True
CURVE_NO = 221 # random.choice(ls_curves)
print(CURVE_NO)
def get_transform_matrices(dict_data):
    # Sorting into deep DMD format
    N_CURVES = len(dict_data)
    ls_all_indices = np.arange(int(np.ceil(2 / 3 * N_CURVES)))  # We take 2/3rd of the data - The training and validation set
    # random.shuffle(ls_all_indices) # Not required as the initial conditions are already shuffled
    print('[INFO]: Shape of Y : ', dict_data[0]['Y'].shape)
    n_data_pts = dict_data[0]['X'].shape[0]
    Xp = np.empty((0, dict_data[0]['X'].shape[1]))
    Xf = np.empty((0, dict_data[0]['X'].shape[1]))
    Yp = np.empty((0, dict_data[0]['Y'].shape[1]))
    Yf = np.empty((0, dict_data[0]['Y'].shape[1]))
    for i in ls_all_indices:
        Xp = np.concatenate([Xp, dict_data[i]['X'][0:-1, :]], axis=0)
        Xf = np.concatenate([Xf, dict_data[i]['X'][1:, :]], axis=0)
        Yp = np.concatenate([Yp, dict_data[i]['Y'][0:-1, :]], axis=0)
        Yf = np.concatenate([Yf, dict_data[i]['Y'][1:, :]], axis=0)
    dict_DATA_RAW = {'Xp': Xp, 'Xf': Xf, 'Yp': Yp, 'Yf': Yf}
    n_train = int(np.ceil(len(dict_DATA_RAW['Xp']) / 2))  # Segregate half of data as training
    dict_DATA_TRAIN_RAW = {'Xp': dict_DATA_RAW['Xp'][0:n_train], 'Xf': dict_DATA_RAW['Xf'][0:n_train],
                           'Yp': dict_DATA_RAW['Yp'][0:n_train], 'Yf': dict_DATA_RAW['Yf'][0:n_train]}
    # _, dict_Scaler, _ = scale_train_data(dict_DATA_TRAIN_RAW, 'min max')
    _, dict_Scaler, transform_matrices = oc.scale_train_data(dict_DATA_TRAIN_RAW, 'standard')
    Px = transform_matrices['X_PT']
    bx = transform_matrices['X_bT'].T
    Py = transform_matrices['Y_PT']
    by = transform_matrices['Y_bT'].T
    return Px,bx,Py,by

def get_K(dict_data,WITH_OUTPUT = True,TRANSFORMED = True):
    a11, a21, a22, gamma = get_sys_params()
    if WITH_OUTPUT:
        K_11 = np.array([[a11, 0], [a21, a22]])
        K_12 = np.array([[0, 0, 0], [gamma, 0, 0]])
        K_21 = np.array([[0, 0], [0, 0] , [0,0]])
        K_22 = np.array([[a11 ** 2, 0, 0], [a11 * a21, a11 * a22, a11 * gamma],[0, 0, a11 ** 3]])
    else:
        K_11 = np.array([[a11, 0], [a21, a22]])
        K_12 = np.array([[0], [gamma]])
        K_21 = np.array([[0, 0]])
        K_22 = np.array([[a11 ** 2]])
    if TRANSFORMED:
        Px,bx,Py,by = get_transform_matrices(dict_data)
        # Transforming the matrix
        K_11_t = np.matmul(Px,np.matmul(K_11,np.linalg.inv(Px)))
        K_12_t = np.matmul(Px,K_12)
        K_13_t = np.matmul(np.eye(len(K_11)) - K_11_t,bx)
        K_21_t = np.matmul(K_21,np.linalg.inv(Px))
        K_22_t = K_22
        K_23_t = np.matmul(K_21_t,bx)
        # Forming the complete matrix
        K_1_t = np.concatenate([np.concatenate([K_11_t,K_12_t],axis=1),K_13_t],axis=1)
        K_2_t = np.concatenate([np.concatenate([K_21_t,K_22_t],axis=1),K_23_t],axis=1)
        K_3_t = np.zeros(shape=(1,len(K_1_t[0])))
        K_3_t[-1,-1] = 1
        K = np.concatenate([np.concatenate([K_1_t,K_2_t],axis=0),K_3_t],axis=0)
    else:
        K_1 = np.concatenate([K_11, K_12], axis=1)
        K_2 = np.concatenate([K_21, K_22], axis=1)
        K = np.concatenate([K_1, K_2], axis=0)
    return K

def get_sys_params():
    a11 = 0.9
    a21 = -0.4
    a22 = -0.8
    gamma = -0.9
    return a11,a21,a22,gamma


def phase_portrait_data(TRANSFORMED= True):
    # System Parameters
    a11, a21, a22, gamma = get_sys_params()
    A = np.array([[a11,0.],[a21,a22]])
    # Simulation Parameters
    N_data_points = 30
    sys_params = {'A':A , 'gamma': gamma, 'N_data_points': N_data_points}
    # Phase Space Data
    dict_phase_data = {}
    X0 = np.empty(shape=(0, 2))
    i=0
    for x1,x2 in itertools.product(list(np.arange(-10,11,4)), list(np.arange(-125,125,80))):
        sys_params['x0'] = np.array([[x1,x2]])
        X0 = np.concatenate([X0, sys_params['x0']], axis=0)
        dict_phase_data[i] = oc.sim_sys_1_2(sys_params)
        i = i+1
    # Theoretical results
    if TRANSFORMED:
        K_t = get_K(dict_data,WITH_OUTPUT = True,TRANSFORMED = True)
    else:
        K_t = get_K(dict_data,WITH_OUTPUT = True,TRANSFORMED = False)
    eval_t, W_t = np.linalg.eig(K_t)
    idx = eval_t.argsort()
    eval_t = eval_t[idx]
    W_t = W_t[:, idx]
    E = np.diag(eval_t)
    E, W_t, comp_modes, comp_modes_conj = resolve_complex_right_eigenvalues(E, W_t)
    Wi_t = np.linalg.inv(W_t)
    sampling_resolution = 0.5
    x1 = np.arange(-10, 10.5, 0.5)
    # x2 = np.arange(-10, 10.5, 0.5)
    x2 = np.arange(-150, 20, 4)
    X1, X2 = np.meshgrid(x1, x2)
    if TRANSFORMED:
        PHI_theo = np.zeros(shape=(X1.shape[0], X1.shape[1], 6))
        PSI_theo = np.zeros(shape=(X1.shape[0], X1.shape[1], 6))
    else:
        PHI_theo = np.zeros(shape=(X1.shape[0], X1.shape[1], 5))
        PSI_theo = np.zeros(shape=(X1.shape[0], X1.shape[1], 5))

    for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
        x1_i = X1[i, j]
        x2_i = X2[i, j]
        if TRANSFORMED:
            psiXT_i = np.array(([[x1_i, x2_i, x1_i ** 2, x1_i * x2_i, x1_i ** 3, 1]]))
        else:
            psiXT_i = np.array(([[x1_i, x2_i, x1_i ** 2, x1_i * x2_i, x1_i ** 3]]))
        PHI_theo[i, j, :] = np.matmul(Wi_t, psiXT_i.T).reshape((1, 1, -1))
        PSI_theo[i, j, :] = psiXT_i.reshape((1, 1, -1))

    if TRANSFORMED:
        K_t3 = get_K(dict_data, WITH_OUTPUT=False, TRANSFORMED=True)
    else:
        K_t3 = get_K(dict_data, WITH_OUTPUT=False, TRANSFORMED=False)
    # K_t3 = np.array([[a11, 0, 0], [a21, a22, gamma], [0, 0, a11 ** 2]])
    eval_t3, W_t3 = np.linalg.eig(K_t3)
    idx = eval_t3.argsort()
    eval_t3 = eval_t3[idx]
    W_t3 = W_t3[:, idx]
    E3 = np.diag(eval_t3)
    E3, W_t3, comp_modes3, comp_modes_conj3 = resolve_complex_right_eigenvalues(E3, W_t3)
    Wi_t3 = np.linalg.inv(W_t3)
    sampling_resolution = 0.5
    x1_3 = np.arange(-10, 10.5, 0.5)
    # x2_3 = np.arange(-10, 10.5, 0.5)
    x2_3 = np.arange(-150, 20, 4)
    X1_3, X2_3 = np.meshgrid(x1_3, x2_3)
    if TRANSFORMED:
        PHI_theo3 = np.zeros(shape=(X1_3.shape[0], X1_3.shape[1], 4))
        PSI_theo3 = np.zeros(shape=(X1_3.shape[0], X1_3.shape[1], 4))
    else:
        PHI_theo3 = np.zeros(shape=(X1_3.shape[0], X1_3.shape[1], 3))
        PSI_theo3 = np.zeros(shape=(X1_3.shape[0], X1_3.shape[1], 3))
    for i, j in itertools.product(range(X1_3.shape[0]), range(X1_3.shape[1])):
        x1_i3 = X1_3[i, j]
        x2_i3 = X2_3[i, j]
        if TRANSFORMED:
            psiXT_i3 = np.array(([[x1_i3, x2_i3, x1_i3 ** 2,1]]))
        else:
            psiXT_i3 = np.array(([[x1_i3, x2_i3, x1_i3 ** 2]]))
        PHI_theo3[i, j, :] = np.matmul(Wi_t3, psiXT_i3.T).reshape((1, 1, -1))
        PSI_theo3[i, j, :] = psiXT_i3.reshape((1, 1, -1))
    return dict_phase_data, PHI_theo, PSI_theo, X1, X2, E, W_t, comp_modes, comp_modes_conj, PHI_theo3, PSI_theo3, X1_3, X2_3, E3, W_t3, comp_modes3, comp_modes_conj3

def get_dict_param(run_folder_name_curr,SYS_NO,sess,nn=False):
    dict_p = {}
    saver = tf.compat.v1.train.import_meta_graph(run_folder_name_curr + '/System_' + str(SYS_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(run_folder_name_curr))
    if not nn:
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
    else:
        try:
            dict_p['xpT_feed'] = tf.get_collection('xp_feed')[0]
            dict_p['f'] = tf.get_collection('f')[0]
            dict_p['g'] = tf.get_collection('g')[0]
        except:
            print('Neural network model info not found')
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
    # sampling_resolution = 0.1
    # x1 = np.arange(-5, 5 + sampling_resolution, sampling_resolution)
    # x2 = np.arange(-5, 5 + sampling_resolution, sampling_resolution)
    sampling_resolution = 0.5
    x1 = np.arange(-10, 10 + sampling_resolution, sampling_resolution)
    # x2 = np.arange(-10, 10 + sampling_resolution, sampling_resolution)
    x2 = np.arange(-150, 20 , 4)
    X1, X2 = np.meshgrid(x1, x2)
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
            idx = eval.argsort()
            eval = eval[idx]
            W = W[:, idx]
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
        else:
            #TODO - Do what happens when left eigenvectors are inserted here
            print('Meh')
    else:
        if RIGHT_EIGEN_VECTORS:
            eval, W = np.linalg.eig(K)
            idx = eval.argsort()
            eval = eval[idx]
            W = W[:, idx]
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
        else:
            #TODO - Do what happens when left eigenvectors are inserted here
            print('Meh')
    return PHI,PSI,Phi_t, koop_modes, comp_modes, comp_modes_conj, X1, X2, eval
# def r2_n_step_prediction_accuracy_ham(ls_steps,ls_curves,dict_data):
#     sess3 = tf.InteractiveSession()
#     saver = tf.compat.v1.train.import_meta_graph(run_folder_name_HAM_X + '/System_' + str(SYS_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
#     saver.restore(sess3, tf.train.latest_checkpoint(run_folder_name_HAM_X))
#     dict_params_x = {}
#     dict_params_x['psix'] = tf.get_collection('psix')[0]
#     dict_params_x['x_feed'] = tf.get_collection('x_feed')[0]
#     dict_params_x['AT'] = tf.get_collection('AT')[0]
#     dict_params_x['AT_num'] = sess3.run(dict_params_x['AT'])
#     # Initialization
#     n_states = len(dict_data[list(dict_data.keys())[0]]['X'][0])
#     n_outputs = len(dict_data[list(dict_data.keys())[0]]['Y'][0])
#     dict_X = {}
#     dict_X_pred = {}
#     dict_Y = {}
#     dict_Y_pred = {}
#     for step in ls_steps:
#         dict_X[step] = np.empty(shape=(0,n_states))
#         dict_X_pred[step] = np.empty(shape=(0, n_states))
#         dict_Y[step] = np.empty(shape=(0, n_outputs))
#         dict_Y_pred[step] = np.empty(shape=(0, n_outputs))
#     # Getting/Sorting the x component
#     for CURVE_NO in ls_curves:
#         dict_DATA_i = oc.scale_data_using_existing_scaler_folder(dict_data[CURVE_NO], SYS_NO)
#         X_scaled = dict_DATA_i['X']
#         Y_scaled = dict_DATA_i['Y']
#         for i in range(len(X_scaled) - np.max(ls_steps) - 1):
#             xi = X_scaled[i:i+1]
#             for step in range(1,1+np.max(ls_steps)):
#                 xi = np.matmul(xi,dict_params_x['AT_num']) + dict_params_x['psix'].eval(feed_dict={dict_params_x['x_feed']: xi})
#                 if step in ls_steps:
#                     dict_X_pred[step] = np.concatenate([dict_X_pred[step],xi],axis=0)
#                     dict_X[step] = np.concatenate([dict_X[step], X_scaled[i+step:i+step+1] ], axis=0)
#                     dict_Y[step] = np.concatenate([dict_Y[step], Y_scaled[i + step:i + step + 1]],axis=0)
#     tf.reset_default_graph()
#     sess3.close()
#
#     sess4 = tf.InteractiveSession()
#     saver = tf.compat.v1.train.import_meta_graph(run_folder_name_HAM_Y + '/System_' + str(SYS_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
#     saver.restore(sess4, tf.train.latest_checkpoint(run_folder_name_HAM_Y))
#     dict_params_y = {}
#     dict_params_y['psix'] = tf.get_collection('psix')[0]
#     dict_params_y['x_feed'] = tf.get_collection('x_feed')[0]
#     dict_params_y['CT'] = tf.get_collection('AT')[0]
#     dict_params_y['CT_num'] = sess4.run(dict_params_y['CT'])
#     dict_r2 = {}
#     for step in ls_steps:
#         dict_Y_pred[step] = np.matmul(dict_X_pred[step],dict_params_y['CT_num']) + dict_params_y['psix'].eval(feed_dict={dict_params_y['x_feed']: dict_X_pred[step]})
#         # Compute the r^2
#         X = oc.inverse_transform_X(dict_X[step], SYS_NO)
#         Y = oc.inverse_transform_Y(dict_Y[step], SYS_NO)
#         Xhat = oc.inverse_transform_X(dict_X_pred[step], SYS_NO)
#         Yhat = oc.inverse_transform_Y(dict_Y_pred[step], SYS_NO)
#         SSE = np.sum(np.square(X - Xhat)) + np.sum(np.square(Y - Yhat))
#         SST = np.sum(np.square(X)) + np.sum(np.square(Y))
#         dict_r2[step] = [np.max([0, 1- (SSE/SST)])*100]
#     tf.reset_default_graph()
#     sess4.close()
#     df_r2 = pd.DataFrame(dict_r2)
#     print(df_r2)
#     return df_r2
def r2_n_step_prediction_accuracy2(ls_steps,ls_curves,dict_data,dict_params_curr,with_output = True, nn= False):
    n_states = len(dict_data[list(dict_data.keys())[0]]['X'][0])
    dict_X = {}
    dict_X_pred = {}
    if with_output:
        n_outputs = len(dict_data[list(dict_data.keys())[0]]['Y'][0])
        dict_Y = {}
        dict_Y_pred = {}
    for step in ls_steps:
        dict_X[step] = np.empty(shape=(0, n_states))
        dict_X_pred[step] = np.empty(shape=(0, n_states))
        if with_output:
            dict_Y[step] = np.empty(shape=(0, n_outputs))
            dict_Y_pred[step] = np.empty(shape=(0, n_outputs))
    for CURVE_NO in ls_curves:
        dict_DATA_i = oc.scale_data_using_existing_scaler_folder(dict_data[CURVE_NO], SYS_NO)
        X_scaled = dict_DATA_i['X']
        if with_output:
            Y_scaled = dict_DATA_i['Y']
        for i in range(len(X_scaled) - np.max(ls_steps) - 2):
            if nn:
                xi = X_scaled[i:i + 1]
            else:
                psi_xi = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: X_scaled[i:i + 1]})
            for step in range(1,np.max(ls_steps)+1):

                if step in ls_steps:
                    if nn:
                        xi = dict_params_curr['f'].eval(feed_dict={dict_params_curr['xpT_feed']: xi})
                        dict_X_pred[step] = np.concatenate([dict_X_pred[step],xi], axis=0)
                    else:
                        psi_xi = np.matmul(psi_xi, dict_params_curr['KxT_num'])
                        dict_X_pred[step] = np.concatenate([dict_X_pred[step],psi_xi[:,0:n_states]],axis=0)
                    dict_X[step] = np.concatenate([dict_X[step], X_scaled[i+step:i+step+1] ], axis=0)
                    if with_output:
                        if nn:
                            dict_Y_pred[step] = np.concatenate([dict_Y_pred[step], dict_params_curr['g'].eval(feed_dict={dict_params_curr['xpT_feed']: xi})], axis=0)
                        else:
                            dict_Y_pred[step] = np.concatenate([dict_Y_pred[step],np.matmul(psi_xi,dict_params_curr['WhT_num'])],axis=0)
                        dict_Y[step] = np.concatenate([dict_Y[step], Y_scaled[i + step:i + step + 1]],axis=0)
    dict_r2 = {}
    for step in ls_steps:
        # Compute the r^2
        X = oc.inverse_transform_X(dict_X[step], SYS_NO)
        Xhat = oc.inverse_transform_X(dict_X_pred[step], SYS_NO)
        SSE = np.sum(np.square(X - Xhat))
        SST = np.sum(np.square(X))
        if with_output:
            Y = oc.inverse_transform_Y(dict_Y[step], SYS_NO)
            Yhat = oc.inverse_transform_Y(dict_Y_pred[step], SYS_NO)
            SSE = SSE + np.sum(np.square(Y - Yhat))
            SST = SST + np.sum(np.square(Y))
        dict_r2[step] = [np.max([0, 1 - (SSE / SST)]) * 100]
    df_r2 = pd.DataFrame(dict_r2)
    print(df_r2)
    return df_r2


TRANSFORMATION_STATUS = True
dict_phase_data, PHI_theo, PSI_theo, X1_theo, X2_theo, E_theo, W_theo, comp_modes_theo, comp_modes_conj_theo, PHI_theo3, PSI_theo3, X1_theo3, X2_theo3, E_theo3, W_theo3, comp_modes_theo3, comp_modes_conj_theo3 = phase_portrait_data(TRANSFORMATION_STATUS )
# ##
#
dict_params = {}
sess1 = tf.InteractiveSession()
dict_params['DeepX'] = get_dict_param(run_folder_name_DEEPDMD ,SYS_NO,sess1)
# df_r2_SEQ = r2_n_step_prediction_accuracy2(ls_steps,ls_curves,dict_data,dict_params['DeepX'],with_output=False)
# _, CURVE_NO = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Seq'])
PHI_DEEP_X,PSI_DEEP_X, Phi_t_DEEP_X,koop_modes_DEEP_X, comp_modes_DEEP_X, comp_modes_conj_DEEP_X,X1_DEEP_X,X2_DEEP_X, E_DEEP_X = modal_analysis(dict_oc_data,dict_data[CURVE_NO],dict_params['DeepX'],REDUCED_MODES = False,Senergy_THRESHOLD = 99.9,RIGHT_EIGEN_VECTORS=True,SHOW_PCA_X = False)
tf.reset_default_graph()
sess1.close()

# dict_params = {}
sess2 = tf.InteractiveSession()
dict_params['Seq'] = get_dict_param(run_folder_name_SEQ_ocDEEPDMD ,SYS_NO,sess2)
df_r2_SEQ = r2_n_step_prediction_accuracy2(ls_steps,ls_curves,dict_data,dict_params['Seq'])
# _, CURVE_NO = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Seq'])
PHI_SEQ,PSI_SEQ, Phi_t_SEQ,koop_modes_SEQ, comp_modes_SEQ, comp_modes_conj_SEQ,X1_SEQ,X2_SEQ, E_SEQ = modal_analysis(dict_oc_data,dict_data[CURVE_NO],dict_params['Seq'],REDUCED_MODES = False,Senergy_THRESHOLD = 99.9,RIGHT_EIGEN_VECTORS=True,SHOW_PCA_X = False)
tf.reset_default_graph()
sess2.close()

sess3 = tf.InteractiveSession()
dict_params['Deep'] = get_dict_param(run_folder_name_DIR_ocDEEPDMD ,SYS_NO,sess3)
df_r2_DEEPDMD = r2_n_step_prediction_accuracy2(ls_steps,ls_curves,dict_data,dict_params['Deep'])
# df_r2_DEEPDMD, _ = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Deep'])
PHI_DEEPDMD,PSI_DEEPDMD,Phi_t_DEEPDMD,koop_modes_DEEPDMD, comp_modes_DEEPDMD, comp_modes_conj_DEEPDMD,X1_DEEPDMD, X2_DEEPDMD, E_DEEPDMD = modal_analysis(dict_oc_data,dict_data[CURVE_NO],dict_params['Deep'],REDUCED_MODES = False,Senergy_THRESHOLD = 99.99,RIGHT_EIGEN_VECTORS=True,SHOW_PCA_X = False)
tf.reset_default_graph()
sess3.close()

sess5 = tf.InteractiveSession()
dict_params['Deep_SUBOPT'] = get_dict_param(run_folder_name_DIR_ocDEEPDMD_SUBOPT ,SYS_NO,sess5)
df_r2_DEEPDMD_SUBOPT = r2_n_step_prediction_accuracy2(ls_steps,ls_curves,dict_data,dict_params['Deep_SUBOPT'])
# df_r2_DEEPDMD, _ = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Deep'])
PHI_DEEPDMD_SUBOPT,PSI_DEEPDMD_SUBOPT,Phi_t_DEEPDMD_SUBOPT,koop_modes_DEEPDMD_SUBOPT, comp_modes_DEEPDMD_SUBOPT, comp_modes_conj_DEEPDMD_SUBOPT,X1_DEEPDMD_SUBOPT, X2_DEEPDMD_SUBOPT, E_DEEPDMD_SUBOPT = modal_analysis(dict_oc_data,dict_data[CURVE_NO],dict_params['Deep_SUBOPT'],REDUCED_MODES = False,Senergy_THRESHOLD = 99.99,RIGHT_EIGEN_VECTORS=True,SHOW_PCA_X = False)
tf.reset_default_graph()
sess5.close()
#
# sess4 = tf.InteractiveSession()
# dict_params['nn'] = get_dict_param(run_folder_name_NN ,SYS_NO,sess4,nn= True)
# df_r2_NN = r2_n_step_prediction_accuracy2(ls_steps,ls_curves,dict_data,dict_params['nn'],nn=True)
# # df_r2_DEEPDMD, _ = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Deep'])
# tf.reset_default_graph()
# sess4.close()

# df_r2_HAM = r2_n_step_prediction_accuracy_ham(ls_steps,ls_curves,dict_data)

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

## Figure 1

# FONT_SIZE = 14
# DOWNSAMPLE = 4
# LINE_WIDTH_c_d = 3
# TRUTH_MARKER_SIZE = 15
# TICK_FONT_SIZE = 9
# HEADER_SIZE = 21
#
# COL_SIZE = 16
# ROW_SIZE = 10
# SPACE_BETWEEN_ROWS = 2
#
# plt.figure(figsize=(COL_SIZE,ROW_SIZE))
# plt.rcParams["axes.edgecolor"] = "black"
# plt.rcParams["axes.linewidth"] = 1
# plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["mathtext.fontset"] = 'cm'
#
#
# FIRST_ROW_X_SPAN = np.int(COL_SIZE/4)
# FIRST_ROW_Y_SPAN = np.int((ROW_SIZE-SPACE_BETWEEN_ROWS)/2)
# plt.subplot2grid((ROW_SIZE,COL_SIZE), (0,0), colspan=5, rowspan=FIRST_ROW_Y_SPAN)
# alpha = 1.0
# epsilon = alpha - 0.01
# arrow_length = 0.3
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
#             if dist<2:
#                 plt.arrow(x,y,dx,dy,head_width = 0.1,head_length=0.5,alpha=1,color='tab:green')
#             else:
#                 plt.arrow(x, y, dx, dy, head_width=0.3, head_length=3, alpha=1, color='tab:green')
# plt.xlabel('$x_1$',fontsize = FONT_SIZE)
# plt.ylabel('$x_2$',fontsize = FONT_SIZE)
# plt.plot([0],[0],'o',color='tab:red',markersize=10)
# plt.xlim([-10,10])
# plt.ylim([-126,125])
# plt.xticks(fontsize = TICK_FONT_SIZE)
# plt.yticks(fontsize = TICK_FONT_SIZE)
# plt.title('(a)',fontsize = HEADER_SIZE,loc='left')
# plt.show()

# # plt.subplot2grid((10,16), (5,0), colspan=4, rowspan=2)
# # n_states = d_SEQ[CURVE_NO]['X'].shape[1]
# # n_outputs = d_SEQ[CURVE_NO]['Y'].shape[1]
# # pl_max = 0
# # pl_min = 0
# # for i in range(n_states):
# #     x_scale = 10**np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['X'][:,i]))))
# #     l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1) + (r'$[\times 10^{{{}}}]$').format(np.int(np.log10(x_scale))))
# #     plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['X']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['X'][0::DOWNSAMPLE,i]/x_scale,'.',color = colors[i],markersize = TRUTH_MARKER_SIZE)
# #     plt.plot(d_SEQ[CURVE_NO]['X_est_one_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
# #     plt.plot(d_DDMD[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='solid', color=colors[i])
# #     plt.plot(d_NN[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
# #     # plt.plot(d_HAM[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
# #     pl_max = np.max([pl_max,np.max(d_SEQ[CURVE_NO]['X'][:,i]/x_scale)])
# #     pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['X'][:, i] / x_scale)])
# # for i in range(n_outputs):
# #     y_scale = 10 ** np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['Y'][:, i]))))
# #     plt.plot([], color=colors[n_states+i], label=('$y_{}$').format(i + 1) + (r'$[\times 10^{{{}}}]$').format(np.int(np.log10(y_scale))))
# #     plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['Y']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['Y'][0::DOWNSAMPLE,i]/y_scale, '.',color = colors[n_states+i],markersize = TRUTH_MARKER_SIZE)
# #     plt.plot(d_SEQ[CURVE_NO]['Y_est_one_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
# #     plt.plot(d_DDMD[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='solid', color=colors[n_states+i])
# #     plt.plot(d_NN[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='dashdot', color=colors[n_states+i])
# #     # plt.plot(d_HAM[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='dashdot', color=colors[n_states + i])
# #     pl_max = np.max([pl_max, np.max(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
# #     pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
# # l1 = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =3)
# # plt.gca().add_artist(l1)
# # # l1 = plt.legend(loc='upper center',fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =3)
# # # plt.xlabel('Time Index(k)',fontsize = FONT_SIZE)
# # # plt.ylabel('x,y [1 -step]',fontsize = FONT_SIZE)
# # plt.text(20,-1,'[1-step]',fontsize = FONT_SIZE)
# # plt.ylim([pl_min-0.1,pl_max+0.1])
# # plt.xticks([])
# # plt.yticks(fontsize = TICK_FONT_SIZE)
# # plt.xlim([-0.5,29.5])
# # plt.title('(b)',fontsize = HEADER_SIZE,loc='left')
# # plt.subplot2grid((10,16), (8,0), colspan=4, rowspan=2)
# # pl_max = 0
# # pl_min = 0
# # for i in range(n_states):
# #     x_scale = 10**np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['X'][:,i]))))
# #     # l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(x_scale))))
# #     plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['X']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['X'][0::DOWNSAMPLE,i]/x_scale,'.',color = colors[i],markersize = TRUTH_MARKER_SIZE)
# #     plt.plot(d_SEQ[CURVE_NO]['X_est_n_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
# #     plt.plot(d_DDMD[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle='solid', color=colors[i])
# #     plt.plot(d_NN[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
# #     # plt.plot(d_HAM[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
# #     pl_max = np.max([pl_max,np.max(d_SEQ[CURVE_NO]['X'][:,i]/x_scale)])
# #     pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['X'][:, i] / x_scale)])
# # for i in range(n_outputs):
# #     y_scale = 10 ** np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['Y'][:, i]))))
# #     # plt.plot([], color=colors[n_states+i], label=('$y_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(y_scale))))
# #     plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['Y']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['Y'][0::DOWNSAMPLE,i]/y_scale, '.',color = colors[n_states+i],markersize = TRUTH_MARKER_SIZE)
# #     plt.plot(d_SEQ[CURVE_NO]['Y_est_n_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
# #     plt.plot(d_DDMD[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle='solid', color=colors[n_states+i])
# #     plt.plot(d_NN[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle='dashdot', color=colors[n_states + i])
# #     # plt.plot(d_HAM[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle='dashdot', color=colors[n_states+i])
# #     pl_max = np.max([pl_max, np.max(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
# #     pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
# # # l1 = plt.legend(loc='lower right',fontsize = 14)
# # # plt.gca().add_artist(l1)
# # a1, = plt.plot([],'.',markersize = TRUTH_MARKER_SIZE,label='Truth',color = 'grey')
# # a2, = plt.plot([], linestyle = 'dashed',linewidth = 1,label='Seq ocdDMD',color = 'grey')
# # a3, = plt.plot([], linestyle ='solid',linewidth = 1,label='Dir ocdDMD',color = 'grey')
# # a4, = plt.plot([], linestyle ='dashdot',linewidth = 1,label='nn-model',color = 'grey')
# # l1 = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =2)
# # plt.gca().add_artist(l1)
# # # l2 = plt.legend((a1,a2,a3),('Truth','Sequential oc-deepDMD','direct oc-deepDMD','Hammerstein model'),loc = "upper right",fontsize = FONT_SIZE)
# # plt.xlabel('$k$ (time index)',fontsize = FONT_SIZE)
# # plt.text(20,-1,'[n-step]',fontsize = FONT_SIZE)
# # # plt.ylabel('x,y [n -step]',fontsize = FONT_SIZE)
# # plt.text(-6,0,'States and Outputs',rotation = 90,fontsize = FONT_SIZE)
# # # plt.title('(b)',fontsize = FONT_SIZE)
# # plt.ylim([pl_min-0.1,pl_max+0.1])
# # plt.xticks(fontsize = TICK_FONT_SIZE)
# # plt.yticks(fontsize = TICK_FONT_SIZE)
# # plt.xlim([-0.5,29.5])
#
#
#
# plt.subplot2grid((ROW_SIZE,COL_SIZE), (0,6), colspan=5, rowspan=3)
# # plt.bar(df_r2_SEQ.index,df_r2_SEQ.mean(axis=1),color = colors[1],label='Seq ocdDMD')
# # plt.plot(df_r2_DEEPDMD.index,df_r2_DEEPDMD.mean(axis=1),color = colors[0],label='dir ocdDMD', linewidth = LINE_WIDTH_c_d )
# plt.bar(df_r2_SEQ.columns.to_numpy(),df_r2_SEQ.to_numpy().reshape(-1),color = colors[1],label='Seq ocdDMD')
# plt.plot(df_r2_DEEPDMD.columns.to_numpy(),df_r2_DEEPDMD.to_numpy().reshape(-1),color = colors[0],label='dir ocdDMD', linewidth = LINE_WIDTH_c_d )
# plt.plot(df_r2_NN.columns.to_numpy(),df_r2_NN.to_numpy().reshape(-1),color = colors[2],label='nn-model',linewidth = LINE_WIDTH_c_d )
# # plt.plot(df_r2_HAM.columns.to_numpy(),df_r2_HAM.to_numpy().reshape(-1),color = colors[2],label='Hamm nn-model',linewidth = LINE_WIDTH_c_d )
# plt.xlim([0.5,14.5])
# plt.ylim([85,101])
# STEPS = 2
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =2)
# plt.xticks(fontsize = TICK_FONT_SIZE)
# plt.yticks(fontsize = TICK_FONT_SIZE)
# plt.xticks(ticks=np.arange(2, 15, step=STEPS),labels=range(2,15,STEPS))
# plt.xlabel('# Prediction Steps',fontsize = FONT_SIZE)
# plt.ylabel('$r^2$(in %)',fontsize = FONT_SIZE)
# plt.title('(c)',fontsize = HEADER_SIZE,loc='left')
#
#
#
# plt.subplot2grid((ROW_SIZE,COL_SIZE), (0,12), colspan=4, rowspan=3)
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
# plt.xlim([-0.5,20.5])
# plt.title('(d)',fontsize = HEADER_SIZE,loc='left')
# plt.xticks(fontsize = TICK_FONT_SIZE)
# plt.yticks(fontsize = TICK_FONT_SIZE)
#
# plt.show()
#
# ##
#
# p=0
# for i in range(PHI_SEQ.shape[2]):
#     title = ''
#     if p == 0:
#         f = plt.subplot2grid((10, 16), (5, 6-1), colspan=2, rowspan=2)
#     elif p == 1:
#         f = plt.subplot2grid((10, 16), (5, 10-1), colspan=3, rowspan=2)
#         break
#         # title = title + '(e)\n'
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
# plt.show()


##
NORMALIZE = True
title = ''
FONT_SIZE = 14
max_eigs = 6
# max_eigs = np.max([PHI_DEEP_X.shape[-1] - len(comp_modes_DEEP_X)])
# max_eigs = np.max([PHI_DEEP_X.shape[-1] - len(comp_modes_DEEP_X),PHI_SEQ.shape[-1] - len(comp_modes_SEQ),PHI_DEEPDMD.shape[-1] - len(comp_modes_conj_DEEPDMD)])
# max_eigs = np.max([PHI_DEEP_X.shape[-1],PHI_SEQ.shape[-1],PHI_DEEPDMD.shape[-1]])
# plt.figure(figsize=(30,5))
# ls_eig_order = np.diag(E_theo3)
# ls_eig_order = np.array(list(set(ls_eig_order).union(set(E_theo))))
# ls_eig_order = np.concatenate([ls_eig_order,np.array([1])],axis=0)
epsilon = 0.001
f,ax = plt.subplots(6,max_eigs,sharex=True,sharey=True,figsize=(3*max_eigs,18))
for row_i in range(6):
    if row_i ==0:
        # x DMD modes
        comp_modes_conj = comp_modes_conj_theo3
        comp_modes = comp_modes_theo3
        PHI = PHI_theo3
        # PHI = PSI_theo
        X1 = X1_theo3
        X2 = X2_theo3
        E = np.diag(E_theo3)
    if row_i == 3:
        # x DMD modes
        comp_modes_conj = comp_modes_conj_theo
        comp_modes = comp_modes_theo
        PHI = PHI_theo
        # PHI = PSI_theo
        X1 = X1_theo
        X2 = X2_theo
        E = np.diag(E_theo)
    elif row_i ==1:
        # x DMD modes
        comp_modes_conj = comp_modes_conj_DEEP_X
        comp_modes = comp_modes_DEEP_X
        PHI = PHI_DEEP_X
        X1 = X1_DEEP_X
        X2 = X2_DEEP_X
        E = E_DEEP_X
    elif row_i ==2:
        # Dir ocdDMD modes
        comp_modes_conj = comp_modes_conj_DEEPDMD_SUBOPT
        comp_modes = comp_modes_DEEPDMD_SUBOPT
        PHI = PHI_DEEPDMD_SUBOPT
        X1 = X1_DEEPDMD_SUBOPT
        X2 = X2_DEEPDMD_SUBOPT
        E = E_DEEPDMD_SUBOPT
    elif row_i ==4:
        # Dir ocdDMD modes
        comp_modes_conj = comp_modes_conj_DEEPDMD
        comp_modes = comp_modes_DEEPDMD
        PHI = PHI_DEEPDMD
        X1 = X1_DEEPDMD
        X2 = X2_DEEPDMD
        E = E_DEEPDMD
    elif row_i ==5:
        # Seq ocdDMD modes
        comp_modes_conj = comp_modes_conj_SEQ
        comp_modes = comp_modes_SEQ
        PHI = PHI_SEQ
        X1 = X1_SEQ
        X2 = X2_SEQ
        E = E_SEQ
    p = 0
    for i in range(PHI.shape[-1]):
        if i in comp_modes_conj:
            continue
        elif i in comp_modes:
            if NORMALIZE:
                c = ax[row_i, p].pcolor(X1, X2, PHI[:, :, i] / np.max(np.abs(PHI[:, :, i])), cmap='rainbow',vmin=-1, vmax=1)
            else:
                c = ax[row_i,p].pcolor(X1, X2, PHI[:, :, i], cmap='rainbow', vmin=np.min(PHI[:, :, i]),vmax=np.max(PHI[:, :, i]))
            f.colorbar(c, ax=ax[row_i,p])
            ax[row_i,p].set_xlabel('$x_1$ \n' + '$\lambda=$' + str(round(np.real(E[i]), 2)) + r'$\pm$' + 'j' + str(round(np.imag(E[i]), 2)), fontsize=FONT_SIZE)
            ax[row_i,p].set_title(title + '$\phi_{{{},{}}}(x)$'.format(i + 1, comp_modes_conj[comp_modes.index(i)] + 1),fontsize=FONT_SIZE)
            # plt.text(-3.5,3.5,'$\lambda=$' + str(round(np.real(E_SEQ[i]),2)) + r'$\pm$' + str(round(np.imag(E_SEQ[i]),2)), fontsize=FONT_SIZE)
        else:
            if NORMALIZE:
                c = ax[row_i, p].pcolor(X1, X2, PHI[:, :, i]/ np.max(np.abs(PHI[:, :, i])), cmap='rainbow', vmin=-1,vmax=1)
            else:
                c = ax[row_i,p].pcolor(X1, X2, PHI[:, :, i], cmap='rainbow', vmin=np.min(PHI[:, :, i]),vmax=np.max(PHI[:, :, i]))
            f.colorbar(c, ax=ax[row_i,p])
            ax[row_i,p].set_xlabel('$x_1$\n' + '$\lambda=$' + str(round(np.real(E[i]), 2)), fontsize=FONT_SIZE)
            ax[row_i,p].set_title(title + '$\phi_{{{}}}(x)$'.format(i + 1), fontsize=FONT_SIZE)
            # plt.text(-3.5,3.5,'$\lambda=$' + str(round(np.real(E_SEQ[i]),2)), fontsize=FONT_SIZE)
        ax[row_i,p].set_ylabel('$x_2$', fontsize=FONT_SIZE)
        ax[row_i, p].set_xticks([-8, 0, 8])
        ax[row_i, p].set_yticks(np.arange(-140,10,20))
        p = p+1

f.show()













# plt.show()

##

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
#
#
# # plt.savefig('Plots/eg1_TheoreticalExample.svg')
# # plt.savefig('Plots/eg1_TheoreticalExample_pycharm.png')
# plt.show()

##






# ##
# for i in range(nPC):
#     f = plt.subplot2grid((7,18), (4, 3*(i+1)), colspan=3, rowspan=2)
#     c = f.pcolor(X1,X2,PHI[:,:,i],cmap='rainbow', vmin=np.min(PHI[:,:,i]), vmax=np.max(PHI[:,:,i]))
#     if i!=0:
#         plt.yticks([])
#     else:
#         plt.ylabel('$x_2$')
#     if i ==2:
#         plt.title('(e) \n $\lambda_{} =$'.format(i+1) + str(round(eval[i],3)))
#     else:
#         plt.title('$\lambda_{} =$'.format(i+1) + str(round(eval[i],3)))
#     plt.colorbar(c,ax = f)
#     plt.xlabel('$x_1$')
# plt.savefig('Plots/eg1_TheoreticalExample.svg')
# plt.show()
#
# f,ax = plt.subplots(1,5,figsize = (10,1.5))
# for i in range(5):
#     c = ax[i].pcolor(X1,X2,PSI[:,:,i],cmap='rainbow', vmin=np.min(PSI_theo[:,:,i]), vmax=np.max(PSI_theo[:,:,i]))
#     f.colorbar(c,ax = ax[i])
# f.show()
# f,ax = plt.subplots(1,5,figsize = (7,1.5))
# for i in range(5):
#     c = ax[i].pcolor(X1_SEQ,X2_SEQ,PHI_theo[:,:,i],cmap='rainbow', vmin=np.min(PHI_theo[:,:,i]), vmax=np.max(PHI_theo[:,:,i]))
#     f.colorbar(c,ax = ax[i])
#     ax[i].set_title('$\phi_{{{}}}(x)$'.format(i+1))
#     ax[i].set_xlabel('$x_1$')
#     ax[i].set_ylabel('$x_2$')
# f.show()
#
#
#
# ##
# P = PHI_theo[:,:,0].reshape((-1,1))
# for i in range(1,PHI_theo.shape[2]):
#     P = np.concatenate([P,PHI_theo[:,:,i].reshape((-1,1))],axis=1)
# ut,st,vt = np.linalg.svd(P)
# print(st)
# print(np.cumsum(st**2)/np.sum(st**2))
# import copy
# Q = copy.deepcopy(P)
#
# for i in range(0,PHI.shape[2]):
#     Q = np.concatenate([Q,PHI[:,:,i].reshape((-1,1))],axis=1)
# u,s,v = np.linalg.svd(Q)
# print(s)
# print(np.cumsum(s**2)/np.sum(s**2))
#

## Theoretical with scaling

for i in range(5):
    print(corr(PHI_SEQ[:,:,i].reshape(-1),PHI_theo[:,:,i].reshape(-1))[0])
print('-----')
for i in range(5):
    print(corr(PHI_DEEPDMD[:,:,i].reshape(-1),PHI_theo[:,:,i].reshape(-1))[0])