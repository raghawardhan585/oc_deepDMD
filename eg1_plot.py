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

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette


SYS_NO = 10
RUN_NO = 78
RUN_NO_HAMMERSTEIN_X = 10
RUN_NO_HAMMERSTEIN_Y = 23
RUN_NO_DEEPDMD = 52 #1

sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYS_NO)
run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)
run_folder_name_DEEPDMD = sys_folder_name + '/deepDMD/RUN_' + str(RUN_NO_DEEPDMD)
run_folder_name_HAM_X = sys_folder_name + '/Hammerstein/RUN_' + str(RUN_NO_HAMMERSTEIN_X)
run_folder_name_HAM_Y = sys_folder_name + '/Hammerstein/RUN_' + str(RUN_NO_HAMMERSTEIN_Y)


with open(sys_folder_name + '/System_' + str(SYS_NO) + '_SimulatedData.pickle', 'rb') as handle:
    dict_data = pickle.load(handle)
with open(sys_folder_name + '/System_' + str(SYS_NO) + '_ocDeepDMDdata.pickle', 'rb') as handle:
    dict_oc_data = pickle.load(handle)
Ntrain = round(len(dict_oc_data['Xp'])/2)
for items in dict_oc_data:
    dict_oc_data[items] = dict_oc_data[items][0:Ntrain]
with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'rb') as handle:
    d_SEQ = pickle.load(handle)[RUN_NO]
with open(sys_folder_name + '/dict_predictions_Hammerstein.pickle', 'rb') as handle:
    d_HAM = pickle.load(handle)[RUN_NO_HAMMERSTEIN_Y]
with open(sys_folder_name + '/dict_predictions_deepDMD.pickle', 'rb') as handle:
    d_DDMD = pickle.load(handle)[RUN_NO_DEEPDMD]

##
ls_steps = list(range(1,15,1))
ls_curves = list(range(160, 240)) # test curves
Senergy_THRESHOLD = 99.99
REDUCED_MODES = False
RIGHT_EIGEN_VECTORS = True
CURVE_NO = 221 # random.choice(ls_curves)
print(CURVE_NO)
def phase_portrait_data():
    # System Parameters
    A = np.array([[0.86,0.],[0.8,0.4]])
    gamma = -0.9
    # Simulation Parameters
    N_data_points = 30
    sys_params = {'A':A , 'gamma': gamma, 'N_data_points': N_data_points}
    # Phase Space Data
    dict_phase_data = {}
    X0 = np.empty(shape=(0, 2))
    i=0
    for x1,x2 in itertools.product(list(np.arange(-10,11,2)), list(np.arange(-125,16,20))):
        sys_params['x0'] = np.array([[x1,x2]])
        X0 = np.concatenate([X0, sys_params['x0']], axis=0)
        dict_phase_data[i] = oc.sim_sys_1_2(sys_params)
        i = i+1
    # Theoretical results
    a11 = 0.86
    a21 = 0.8
    a22 = 0.4
    gamma = -0.9
    K_t = np.array(
        [[a11, 0, 0, 0, 0], [a21, a22, gamma, 0, 0], [0, 0, a11 ** 2, 0, 0], [0, 0, a11 * a21, a11 * a22, a11 * gamma],
         [0, 0, 0, 0, a11 ** 3]])
    eval_t, W_t = np.linalg.eig(K_t)
    Wi_t = np.linalg.inv(W_t)
    sampling_resolution = 0.1
    x1 = np.arange(-5, 5 + sampling_resolution, sampling_resolution)
    x2 = np.arange(-5, 5 + sampling_resolution, sampling_resolution)
    X1, X2 = np.meshgrid(x1, x2)
    PHI_theo = np.zeros(shape=(X1.shape[0], X1.shape[1], 5))
    PSI_theo = np.zeros(shape=(X1.shape[0], X1.shape[1], 5))
    for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
        x1_i = X1[i, j]
        x2_i = X2[i, j]
        psiXT_i = np.array(([[x1_i, x2_i, x1_i ** 2, x1_i * x2_i, x1_i ** 3]]))
        PHI_theo[i, j, :] = np.matmul(Wi_t, psiXT_i.T).reshape((1, 1, -1))
        PSI_theo[i, j, :] = psiXT_i.reshape((1, 1, -1))
    return dict_phase_data, PHI_theo, PSI_theo
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
    x2 = np.arange(-5, 5 + sampling_resolution, sampling_resolution)
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
def r2_n_step_prediction_accuracy_ham(ls_steps,ls_curves,dict_data):
    sess3 = tf.InteractiveSession()
    saver = tf.compat.v1.train.import_meta_graph(run_folder_name_HAM_X + '/System_' + str(SYS_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess3, tf.train.latest_checkpoint(run_folder_name_HAM_X))
    dict_params_x = {}
    dict_params_x['psix'] = tf.get_collection('psix')[0]
    dict_params_x['x_feed'] = tf.get_collection('x_feed')[0]
    dict_params_x['AT'] = tf.get_collection('AT')[0]
    dict_params_x['AT_num'] = sess3.run(dict_params_x['AT'])
    # Initialization
    n_states = len(dict_data[list(dict_data.keys())[0]]['X'][0])
    n_outputs = len(dict_data[list(dict_data.keys())[0]]['Y'][0])
    dict_X = {}
    dict_X_pred = {}
    dict_Y = {}
    dict_Y_pred = {}
    for step in ls_steps:
        dict_X[step] = np.empty(shape=(0,n_states))
        dict_X_pred[step] = np.empty(shape=(0, n_states))
        dict_Y[step] = np.empty(shape=(0, n_outputs))
        dict_Y_pred[step] = np.empty(shape=(0, n_outputs))
    # Getting/Sorting the x component
    for CURVE_NO in ls_curves:
        dict_DATA_i = oc.scale_data_using_existing_scaler_folder(dict_data[CURVE_NO], SYS_NO)
        X_scaled = dict_DATA_i['X']
        Y_scaled = dict_DATA_i['Y']
        for i in range(len(X_scaled) - np.max(ls_steps) - 1):
            xi = X_scaled[i:i+1]
            for step in range(1,1+np.max(ls_steps)):
                xi = np.matmul(xi,dict_params_x['AT_num']) + dict_params_x['psix'].eval(feed_dict={dict_params_x['x_feed']: xi})
                if step in ls_steps:
                    dict_X_pred[step] = np.concatenate([dict_X_pred[step],xi],axis=0)
                    dict_X[step] = np.concatenate([dict_X[step], X_scaled[i+step:i+step+1] ], axis=0)
                    dict_Y[step] = np.concatenate([dict_Y[step], Y_scaled[i + step:i + step + 1]],axis=0)
    tf.reset_default_graph()
    sess3.close()

    sess4 = tf.InteractiveSession()
    saver = tf.compat.v1.train.import_meta_graph(run_folder_name_HAM_Y + '/System_' + str(SYS_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess4, tf.train.latest_checkpoint(run_folder_name_HAM_Y))
    dict_params_y = {}
    dict_params_y['psix'] = tf.get_collection('psix')[0]
    dict_params_y['x_feed'] = tf.get_collection('x_feed')[0]
    dict_params_y['CT'] = tf.get_collection('AT')[0]
    dict_params_y['CT_num'] = sess4.run(dict_params_y['CT'])
    dict_r2 = {}
    for step in ls_steps:
        dict_Y_pred[step] = np.matmul(dict_X_pred[step],dict_params_y['CT_num']) + dict_params_y['psix'].eval(feed_dict={dict_params_y['x_feed']: dict_X_pred[step]})
        # Compute the r^2
        X = oc.inverse_transform_X(dict_X[step], SYS_NO)
        Y = oc.inverse_transform_Y(dict_Y[step], SYS_NO)
        Xhat = oc.inverse_transform_X(dict_X_pred[step], SYS_NO)
        Yhat = oc.inverse_transform_Y(dict_Y_pred[step], SYS_NO)
        SSE = np.sum(np.square(X - Xhat)) + np.sum(np.square(Y - Yhat))
        SST = np.sum(np.square(X)) + np.sum(np.square(Y))
        dict_r2[step] = [np.max([0, 1- (SSE/SST)])*100]
    tf.reset_default_graph()
    sess4.close()
    df_r2 = pd.DataFrame(dict_r2)
    print(df_r2)
    return df_r2
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
dict_phase_data, PHI_theo, PSI_theo = phase_portrait_data()


dict_params = {}
sess1 = tf.InteractiveSession()
dict_params['Seq'] = get_dict_param(run_folder_name,SYS_NO,sess1)
df_r2_SEQ = r2_n_step_prediction_accuracy2(ls_steps,ls_curves,dict_data,dict_params['Seq'])
# _, CURVE_NO = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Seq'])
PHI_SEQ,PSI_SEQ, Phi_t_SEQ,koop_modes_SEQ, comp_modes_SEQ, comp_modes_conj_SEQ,X1_SEQ,X2_SEQ, E_SEQ = modal_analysis(dict_oc_data,dict_data[CURVE_NO],dict_params['Seq'],REDUCED_MODES = True,Senergy_THRESHOLD = 99.9,RIGHT_EIGEN_VECTORS=True,SHOW_PCA_X = False)
tf.reset_default_graph()
sess1.close()

sess2 = tf.InteractiveSession()
dict_params['Deep'] = get_dict_param(run_folder_name_DEEPDMD,SYS_NO,sess2)
df_r2_DEEPDMD = r2_n_step_prediction_accuracy2(ls_steps,ls_curves,dict_data,dict_params['Deep'])
# df_r2_DEEPDMD, _ = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Deep'])
PHI_DEEPDMD,PSI_DEEPDMD,Phi_t_DEEPDMD,koop_modes_DEEPDMD, comp_modes_DEEPDMD, comp_modes_conj_DEEPDMD,X1_DEEPDMD, X2_DEEPDMD, E_DEEPDMD = modal_analysis(dict_oc_data,dict_data[CURVE_NO],dict_params['Deep'],REDUCED_MODES = False,Senergy_THRESHOLD = 99.99,RIGHT_EIGEN_VECTORS=True,SHOW_PCA_X = False)
tf.reset_default_graph()
sess2.close()

df_r2_HAM = r2_n_step_prediction_accuracy_ham(ls_steps,ls_curves,dict_data)

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

FONT_SIZE = 14
DOWNSAMPLE = 4
LINE_WIDTH_c_d = 3
TRUTH_MARKER_SIZE = 15
TICK_FONT_SIZE = 9
HEADER_SIZE = 21
plt.figure(figsize=(15,10))
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = 'cm'

plt.subplot2grid((10,16), (0,0), colspan=5, rowspan=4)
alpha = 1.0
epsilon = alpha - 0.01
arrow_length = 0.3
ls_pts = list(range(0,1))
for i in list(dict_phase_data.keys())[0:]:
    for j in ls_pts:
        if np.abs(dict_phase_data[i]['X'][j, 0]) > 1 or j==0:
            plt.plot(dict_phase_data[i]['X'][j, 0], dict_phase_data[i]['X'][j, 1], 'o',color='salmon',fillstyle='none',markersize=5)
    plt.plot(dict_phase_data[i]['X'][:, 0], dict_phase_data[i]['X'][:, 1], color='tab:blue',linewidth=0.3)
    if np.mod(i,1)==0:
        for j in ls_pts:
            dist = np.sqrt((dict_phase_data[i]['X'][j, 0] - dict_phase_data[i]['X'][j + 1, 0]) ** 2 + (dict_phase_data[i]['X'][j, 1] - dict_phase_data[i]['X'][j + 1, 1]) ** 2)
            x = dict_phase_data[i]['X'][j, 0]
            y = dict_phase_data[i]['X'][j, 1]
            dx = (dict_phase_data[i]['X'][j + 1, 0] - dict_phase_data[i]['X'][j, 0]) * arrow_length
            dy = (dict_phase_data[i]['X'][j + 1, 1] - dict_phase_data[i]['X'][j, 1]) * arrow_length
            # print(x,' ',y,' ',dist)
            if dist<2:
                plt.arrow(x,y,dx,dy,head_width = 0.1,head_length=0.5,alpha=1,color='tab:green')
            else:
                plt.arrow(x, y, dx, dy, head_width=0.3, head_length=3, alpha=1, color='tab:green')
plt.xlabel('$x_1$',fontsize = FONT_SIZE)
plt.ylabel('$x_2$',fontsize = FONT_SIZE)
plt.plot([0],[0],'o',color='tab:red',markersize=10)
plt.xlim([-10,10])
plt.ylim([-126,16])
plt.xticks(fontsize = TICK_FONT_SIZE)
plt.yticks(fontsize = TICK_FONT_SIZE)
plt.title('(a)',fontsize = HEADER_SIZE,loc='left')


plt.subplot2grid((10,16), (5,0), colspan=4, rowspan=2)
n_states = d_SEQ[CURVE_NO]['X'].shape[1]
n_outputs = d_SEQ[CURVE_NO]['Y'].shape[1]
pl_max = 0
pl_min = 0
for i in range(n_states):
    x_scale = 10**np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['X'][:,i]))))
    l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1) + (r'$[\times 10^{{{}}}]$').format(np.int(np.log10(x_scale))))
    plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['X']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['X'][0::DOWNSAMPLE,i]/x_scale,'.',color = colors[i],markersize = TRUTH_MARKER_SIZE)
    plt.plot(d_SEQ[CURVE_NO]['X_est_one_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
    plt.plot(d_DDMD[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='solid', color=colors[i])
    plt.plot(d_HAM[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
    pl_max = np.max([pl_max,np.max(d_SEQ[CURVE_NO]['X'][:,i]/x_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['X'][:, i] / x_scale)])
for i in range(n_outputs):
    y_scale = 10 ** np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['Y'][:, i]))))
    plt.plot([], color=colors[n_states+i], label=('$y_{}$').format(i + 1) + (r'$[\times 10^{{{}}}]$').format(np.int(np.log10(y_scale))))
    plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['Y']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['Y'][0::DOWNSAMPLE,i]/y_scale, '.',color = colors[n_states+i],markersize = TRUTH_MARKER_SIZE)
    plt.plot(d_SEQ[CURVE_NO]['Y_est_one_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
    plt.plot(d_DDMD[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='solid', color=colors[n_states+i])
    plt.plot(d_HAM[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='dashdot', color=colors[n_states+i])
    pl_max = np.max([pl_max, np.max(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
l1 = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.075),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =3)
plt.gca().add_artist(l1)
# l1 = plt.legend(loc='upper center',fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =3)
# plt.xlabel('Time Index(k)',fontsize = FONT_SIZE)
# plt.ylabel('x,y [1 -step]',fontsize = FONT_SIZE)
plt.text(20,-1,'[1-step]',fontsize = FONT_SIZE)
plt.ylim([pl_min-0.1,pl_max+0.1])
plt.xticks([])
plt.yticks(fontsize = TICK_FONT_SIZE)
plt.xlim([-0.5,29.5])
plt.title('(b)',fontsize = HEADER_SIZE,loc='left')
plt.subplot2grid((10,16), (8,0), colspan=4, rowspan=2)
pl_max = 0
pl_min = 0
for i in range(n_states):
    x_scale = 10**np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['X'][:,i]))))
    # l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(x_scale))))
    plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['X']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['X'][0::DOWNSAMPLE,i]/x_scale,'.',color = colors[i],markersize = TRUTH_MARKER_SIZE)
    plt.plot(d_SEQ[CURVE_NO]['X_est_n_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
    plt.plot(d_DDMD[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle='solid', color=colors[i])
    plt.plot(d_HAM[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
    pl_max = np.max([pl_max,np.max(d_SEQ[CURVE_NO]['X'][:,i]/x_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['X'][:, i] / x_scale)])
for i in range(n_outputs):
    y_scale = 10 ** np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['Y'][:, i]))))
    # plt.plot([], color=colors[n_states+i], label=('$y_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(y_scale))))
    plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['Y']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['Y'][0::DOWNSAMPLE,i]/y_scale, '.',color = colors[n_states+i],markersize = TRUTH_MARKER_SIZE)
    plt.plot(d_SEQ[CURVE_NO]['Y_est_n_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
    plt.plot(d_DDMD[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle='solid', color=colors[n_states+i])
    plt.plot(d_HAM[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle='dashdot', color=colors[n_states+i])
    pl_max = np.max([pl_max, np.max(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
# l1 = plt.legend(loc='lower right',fontsize = 14)
# plt.gca().add_artist(l1)
a1, = plt.plot([],'.',markersize = TRUTH_MARKER_SIZE,label='Truth',color = 'grey')
a2, = plt.plot([], linestyle = 'dashed',linewidth = 1,label='Seq ocdDMD',color = 'grey')
a3, = plt.plot([], linestyle ='solid',linewidth = 1,label='Dir ocdDMD',color = 'grey')
a4, = plt.plot([], linestyle ='dashdot',linewidth = 1,label='Hamm nn-model',color = 'grey')
l1 = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =2)
plt.gca().add_artist(l1)
# l2 = plt.legend((a1,a2,a3),('Truth','Sequential oc-deepDMD','direct oc-deepDMD','Hammerstein model'),loc = "upper right",fontsize = FONT_SIZE)
plt.xlabel('$k$ (time index)',fontsize = FONT_SIZE)
plt.text(20,-1,'[n-step]',fontsize = FONT_SIZE)
# plt.ylabel('x,y [n -step]',fontsize = FONT_SIZE)
plt.text(-6,0,'States and Outputs',rotation = 90,fontsize = FONT_SIZE)
# plt.title('(b)',fontsize = FONT_SIZE)
plt.ylim([pl_min-0.1,pl_max+0.1])
plt.xticks(fontsize = TICK_FONT_SIZE)
plt.yticks(fontsize = TICK_FONT_SIZE)
plt.xlim([-0.5,29.5])



plt.subplot2grid((10,16), (0,6), colspan=5, rowspan=3)
# plt.bar(df_r2_SEQ.index,df_r2_SEQ.mean(axis=1),color = colors[1],label='Seq ocdDMD')
# plt.plot(df_r2_DEEPDMD.index,df_r2_DEEPDMD.mean(axis=1),color = colors[0],label='dir ocdDMD', linewidth = LINE_WIDTH_c_d )
plt.bar(df_r2_SEQ.columns.to_numpy(),df_r2_SEQ.to_numpy().reshape(-1),color = colors[1],label='Seq ocdDMD')
plt.plot(df_r2_DEEPDMD.columns.to_numpy(),df_r2_DEEPDMD.to_numpy().reshape(-1),color = colors[0],label='dir ocdDMD', linewidth = LINE_WIDTH_c_d )
plt.plot(df_r2_HAM.columns.to_numpy(),df_r2_HAM.to_numpy().reshape(-1),color = colors[2],label='Hamm nn-model',linewidth = LINE_WIDTH_c_d )
plt.xlim([0.5,14.5])
plt.ylim([85,101])
STEPS = 2
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =2)
plt.xticks(fontsize = TICK_FONT_SIZE)
plt.yticks(fontsize = TICK_FONT_SIZE)
plt.xticks(ticks=np.arange(2, 15, step=STEPS),labels=range(2,15,STEPS))
plt.xlabel('# Prediction Steps',fontsize = FONT_SIZE)
plt.ylabel('$r^2$(in %)',fontsize = FONT_SIZE)
plt.title('(c)',fontsize = HEADER_SIZE,loc='left')


plt.subplot2grid((10,16), (0,12), colspan=4, rowspan=3)
p=0
for i in range(Phi_t_SEQ.shape[0]):
    if i in comp_modes_conj_SEQ:
        continue
    elif i in comp_modes_SEQ:
        # plt.plot(Phi[i, :],label = 'lala')
        plt.plot(Phi_t_SEQ[i,:],label='$\phi_{{{},{}}}(x)$'.format(i+1,comp_modes_conj_SEQ[comp_modes_SEQ.index(i)]+1), linewidth = LINE_WIDTH_c_d )
        p = p+1
    else:
        plt.plot(Phi_t_SEQ[i, :], label='$\phi_{{{}}}(x)$'.format(i + 1), linewidth = LINE_WIDTH_c_d )
        p = p+1
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =np.int(np.ceil(p/2)))
plt.xlabel('$k$ (time index)',fontsize = FONT_SIZE)
plt.ylabel('$\phi[k]$',fontsize = FONT_SIZE)
plt.xlim([-0.5,20.5])
plt.title('(d)',fontsize = HEADER_SIZE,loc='left')
plt.xticks(fontsize = TICK_FONT_SIZE)
plt.yticks(fontsize = TICK_FONT_SIZE)




p=0
for i in range(PHI_SEQ.shape[2]):
    title = ''
    if p == 0:
        f = plt.subplot2grid((10, 16), (5, 6-1), colspan=3, rowspan=2)
    elif p == 1:
        f = plt.subplot2grid((10, 16), (5, 10-1), colspan=3, rowspan=2)
        # title = title + '(e)\n'
    elif p == 2:
        f = plt.subplot2grid((10, 16), (5, 14-1), colspan=3, rowspan=2)
    elif p == 3:
        f = plt.subplot2grid((10, 16), (8, 8-1), colspan=3, rowspan=2)
    elif p == 4:
        f = plt.subplot2grid((10, 16), (8, 12-1), colspan=3, rowspan=2)
    elif p==5:
        break
    if i in comp_modes_conj_SEQ:
        continue
    elif i in comp_modes_SEQ:
        c = f.pcolor(X1_SEQ,X2_SEQ,PHI_SEQ[:,:,i],cmap='rainbow', vmin=np.min(PHI_SEQ[:,:,i]), vmax=np.max(PHI_SEQ[:,:,i]))
        plt.colorbar(c,ax = f)
        plt.xlabel('$x_1$ \n' + '$\lambda=$' + str(round(np.real(E_SEQ[i]),2)) + r'$\pm$' + str(round(np.imag(E_SEQ[i]),2)), fontsize=FONT_SIZE)
        plt.ylabel('$x_2$', fontsize=FONT_SIZE)
        plt.xticks([-4, 0, 4])
        plt.yticks([-4, 0, 4])
        plt.title(title + '$\phi_{{{},{}}}(x)$'.format(i+1,comp_modes_conj_SEQ[comp_modes_SEQ.index(i)]+1), fontsize=FONT_SIZE)
        # plt.text(-3.5,3.5,'$\lambda=$' + str(round(np.real(E_SEQ[i]),2)) + r'$\pm$' + str(round(np.imag(E_SEQ[i]),2)), fontsize=FONT_SIZE)
        p = p+1
    else:
        c = f.pcolor(X1_SEQ, X2_SEQ, PHI_SEQ[:, :, i], cmap='rainbow', vmin=np.min(PHI_SEQ[:, :, i]),vmax=np.max(PHI_SEQ[:, :, i]))
        plt.colorbar(c, ax=f)
        plt.xlabel('$x_1$\n' + '$\lambda=$' + str(round(np.real(E_SEQ[i]),2)), fontsize=FONT_SIZE)
        plt.ylabel('$x_2$', fontsize=FONT_SIZE)
        plt.xticks([-4, 0, 4])
        plt.yticks([-4, 0, 4])
        plt.title(title + '$\phi_{{{}}}(x)$'.format(i + 1), fontsize=FONT_SIZE )
        # plt.text(-3.5,3.5,'$\lambda=$' + str(round(np.real(E_SEQ[i]),2)), fontsize=FONT_SIZE)
        p = p+1
    if p ==1:
        f.text(-5,5.5,'(e)',fontsize = HEADER_SIZE)



# plt.savefig('Plots/eg1_TheoreticalExample.svg')
# plt.savefig('Plots/eg1_TheoreticalExample_pycharm.png')
plt.show()

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
f,ax = plt.subplots(1,5,figsize = (7,1.5))
for i in range(5):
    c = ax[i].pcolor(X1_SEQ,X2_SEQ,PHI_theo[:,:,i],cmap='rainbow', vmin=np.min(PHI_theo[:,:,i]), vmax=np.max(PHI_theo[:,:,i]))
    f.colorbar(c,ax = ax[i])
    ax[i].set_title('$\phi_{{{}}}(x)$'.format(i+1))
    ax[i].set_xlabel('$x_1$')
    ax[i].set_ylabel('$x_2$')
f.show()
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

##

