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
import seaborn as sb

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



# SYS_NO = 30
# RUN_NO = 47
SYS_NO = 53
RUN_NO = 344
RUN_NO_HAMMERSTEIN = 0
RUN_NO_DEEPDMD = 22

sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYS_NO)
run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)
run_folder_name_DEEPDMD = sys_folder_name + '/deepDMD/RUN_' + str(RUN_NO_DEEPDMD)

with open(sys_folder_name + '/System_' + str(SYS_NO) + '_SimulatedData.pickle', 'rb') as handle:
    dict_data = pickle.load(handle)
with open(sys_folder_name + '/System_' + str(SYS_NO) + '_ocDeepDMDdata.pickle', 'rb') as handle:
    dict_oc_data = pickle.load(handle)
Ntrain = round(len(dict_oc_data['Xp'])/2)
for items in dict_oc_data:
    dict_oc_data[items] = dict_oc_data[items][0:Ntrain]
with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'rb') as handle:
    d_SEQ = pickle.load(handle)[RUN_NO]
# with open(sys_folder_name + '/dict_predictions_Hammerstein.pickle', 'rb') as handle:
#     d_HAM = pickle.load(handle)[RUN_NO_HAMMERSTEIN]
with open(sys_folder_name + '/dict_predictions_deepDMD.pickle', 'rb') as handle:
    d_DDMD = pickle.load(handle)[RUN_NO_DEEPDMD]

##
ls_steps = list(range(1,50,1))
ls_curves = list(range(200, 300)) # test curves
Senergy_THRESHOLD = 99.99
REDUCED_MODES = False
RIGHT_EIGEN_VECTORS = True


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
    dict_rmse = {}
    dict_r2 = {}
    for CURVE_NO in ls_curves:
        dict_rmse[CURVE_NO] = {}
        dict_r2[CURVE_NO] = {}
        dict_DATA_i = oc.scale_data_using_existing_scaler_folder(dict_data[CURVE_NO], SYS_NO)
        X_scaled = dict_DATA_i['X']
        Y_scaled = dict_DATA_i['Y']
        psiX = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: X_scaled})
        for i in ls_steps:  # iterating through each step prediction
            np_psiX_true = psiX[i:, :]
            np_psiX_pred = np.matmul(psiX[:-i, :],
                                     np.linalg.matrix_power(dict_params_curr['KxT_num'], i))  # i step prediction at each datapoint
            Y_pred = np.matmul(np_psiX_pred, dict_params_curr['WhT_num'])
            Y_true = Y_scaled[i:, :]
            dict_rmse[CURVE_NO][i] = np.sqrt(np.mean(np.square(np_psiX_true - np_psiX_pred)))
            dict_r2[CURVE_NO][i] = np.max([0, (
                        1 - (np.sum(np.square(np_psiX_true - np_psiX_pred)) + np.sum(np.square(Y_true - Y_pred))) / (
                            np.sum(np.square(np_psiX_true)) + np.sum(np.square(Y_true)))) * 100])
    df_r2 = pd.DataFrame(dict_r2)
    print(df_r2)
    CHECK_VAL = df_r2.iloc[-1, :].max()
    OPT_CURVE_NO = 0
    for i in range(200, 300):
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
def eig_func_through_time(dict_oc_data,dict_data_curr,dict_params_curr,REDUCED_MODES,Senergy_THRESHOLD,RIGHT_EIGEN_VECTORS = True,SHOW_PCA_X=True):
    psiX = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: dict_oc_data['Xp']}).T
    # Phi0 = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: dict_data_curr['X'][0:1]})
    Phi0 = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: dict_data_curr['X']})
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
    #
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
            Phi = np.matmul(E, np.matmul(np.matmul(Winv, Ur.T), Phi0.T))
            # for i in range(1,len(dict_data_curr['X'])):
            #     Phi = np.concatenate([Phi,np.matmul(E,Phi[:,-1:])],axis=1)
            koop_modes = W
        else:
            #TODO - Do what happens when left eigenvectors are inserted here
            print('Meh')
    else:
        if RIGHT_EIGEN_VECTORS:
            eval, W = np.linalg.eig(K)
            E = np.diag(eval)
            E, W, comp_modes, comp_modes_conj = resolve_complex_right_eigenvalues(E, W)
            Winv = np.linalg.inv(W)
            Phi = np.matmul(E, np.matmul(Winv,Phi0.T))
            # for i in range(1, len(dict_data_curr['X'])):
            #     Phi = np.concatenate([Phi, np.matmul(E, Phi[:, -1:])], axis=1)
            koop_modes = W
        else:
            #TODO - Do what happens when left eigenvectors are inserted here
            print('Meh')
    return Phi, koop_modes, comp_modes, comp_modes_conj


dict_params = {}
sess1 = tf.InteractiveSession()
dict_params['Seq'] = get_dict_param(run_folder_name,SYS_NO,sess1)
df_r2_SEQ, CURVE_NO = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Seq'])
Phi_SEQ,koop_modes_SEQ, comp_modes_SEQ, comp_modes_conj_SEQ = eig_func_through_time(dict_oc_data,dict_data[CURVE_NO],dict_params['Seq'],REDUCED_MODES = True,Senergy_THRESHOLD = 99.99,RIGHT_EIGEN_VECTORS=True,SHOW_PCA_X = False)
tf.reset_default_graph()
sess1.close()



#
sess2 = tf.InteractiveSession()
dict_params['Deep'] = get_dict_param(run_folder_name_DEEPDMD,SYS_NO,sess2)
df_r2_DEEPDMD, _ = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Deep'])
Phi_DEEPDMD,koop_modes_DEEPDMD, comp_modes_DEEPDMD, comp_modes_conj_DEEPDMD = eig_func_through_time(dict_oc_data,dict_data[CURVE_NO],dict_params['Deep'],REDUCED_MODES = False,Senergy_THRESHOLD = 99.99,RIGHT_EIGEN_VECTORS=True,SHOW_PCA_X = False)
tf.reset_default_graph()
sess2.close()




## Dynamic Modes


# plt.figure()
# plt.plot(Phi.T)
# plt.legend(np.arange(nPC))
# plt.show()
sb.heatmap(koop_modes_SEQ,cmap = "YlOrBr",vmin=0.0)
plt.show()
sb.heatmap(koop_modes_DEEPDMD,cmap = "YlOrBr",vmin=0.0)
plt.show()


## Figure 1 - 1 step prediction comparisons


plt.figure(figsize=(15,10))
ax1 = plt.subplot2grid((2,2), (0,0), colspan=1, rowspan=1)


# CURVE_NO = 0
n_states = d_SEQ[CURVE_NO]['X'].shape[1]
n_outputs = d_SEQ[CURVE_NO]['Y'].shape[1]
pl_max = 0
pl_min = 0
for i in range(n_states):
    x_scale = 10**np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['X'][:,i]))))
    l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(x_scale))))
    ax1.plot(d_SEQ[CURVE_NO]['X'][:,i]/x_scale,'.',color = colors[i],linewidth = 5)
    ax1.plot(d_SEQ[CURVE_NO]['X_est_one_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
    ax1.plot(d_DDMD[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='solid', color=colors[i])
    # plt.plot(d_HAM[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
    pl_max = np.max([pl_max,np.max(d_SEQ[CURVE_NO]['X'][:,i]/x_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['X'][:, i] / x_scale)])
for i in range(n_outputs):
    y_scale = 10 ** np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['Y'][:, i]))))
    ax1.plot([], color=colors[n_states+i], label=('$y_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(y_scale))))
    ax1.plot(d_SEQ[CURVE_NO]['Y'][:,i]/y_scale, '.',color = colors[n_states+i],linewidth = 5)
    ax1.plot(d_SEQ[CURVE_NO]['Y_est_one_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
    ax1.plot(d_DDMD[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='solid', color=colors[n_states+i])
    # plt.plot(d_HAM[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='solid', color=colors[i])
    pl_max = np.max([pl_max, np.max(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
# Shrink current axis by 20%
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
l1 = ax1.legend(loc=(1.04,0), borderaxespad=0,fontsize = 14)
plt.gca().add_artist(l1)
a1, = ax1.plot([],'.',linewidth = 5,label='Truth',color = 'grey')
a2, = ax1.plot([], linestyle = 'dashed',linewidth = 1,label='Sequential oc-deepDMD',color = 'grey')
a3, = ax1.plot([], linestyle ='solid',linewidth = 1,label='direct oc-deepDMD',color = 'grey')
a4, = ax1.plot([], linestyle ='dashdot',linewidth = 1,label='Hammerstein model',color = 'grey')
l2 = ax1.legend((a1,a2,a3),('Truth','Sequential oc-deepDMD','direct oc-deepDMD','Hammerstein model'),loc = "upper right",fontsize = 14)
ax1.set_xlabel('Time Index(k)')
ax1.set_ylabel('States and Outputs[1 -step prediction]')
ax1.set_title('(a)')
ax1.set_ylim([pl_min,pl_max])
ax1.set_xlim([0,100])

# plt.show()


# # Figure 2 - n -step prediction comparisons
# plt.subplot2grid((2,2), (1,0), colspan=1, rowspan=1)
# n_states = d_SEQ[CURVE_NO]['X'].shape[1]
# n_outputs = d_SEQ[CURVE_NO]['Y'].shape[1]
# pl_max = 0
# pl_min = 0
# for i in range(n_states):
#     x_scale = 10**np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['X'][:,i]))))
#     l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(x_scale))))
#     plt.plot(d_SEQ[CURVE_NO]['X'][:,i]/x_scale,'.',color = colors[i],linewidth = 5)
#     plt.plot(d_SEQ[CURVE_NO]['X_est_n_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
#     plt.plot(d_DDMD[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle='solid', color=colors[i])
#     # plt.plot(d_HAM[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
#     pl_max = np.max([pl_max,np.max(d_SEQ[CURVE_NO]['X'][:,i]/x_scale)])
#     pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['X'][:, i] / x_scale)])
# for i in range(n_outputs):
#     y_scale = 10 ** np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['Y'][:, i]))))
#     plt.plot([], color=colors[n_states+i], label=('$y_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(y_scale))))
#     plt.plot(d_SEQ[CURVE_NO]['Y'][:,i]/y_scale, '.',color = colors[n_states+i],linewidth = 5)
#     plt.plot(d_SEQ[CURVE_NO]['Y_est_n_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
#     plt.plot(d_DDMD[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle='solid', color=colors[n_states+i])
#     # plt.plot(d_HAM[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='solid', color=colors[i])
#     pl_max = np.max([pl_max, np.max(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
#     pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
# l1 = plt.legend(loc='lower right',fontsize = 14)
# plt.gca().add_artist(l1)
# a1, = plt.plot([],'.',linewidth = 5,label='Truth',color = 'grey')
# a2, = plt.plot([], linestyle = 'dashed',linewidth = 1,label='Sequential oc-deepDMD',color = 'grey')
# a3, = plt.plot([], linestyle ='solid',linewidth = 1,label='direct oc-deepDMD',color = 'grey')
# a4, = plt.plot([], linestyle ='dashdot',linewidth = 1,label='Hammerstein model',color = 'grey')
# l2 = plt.legend((a1,a2,a3),('Truth','Sequential oc-deepDMD','direct oc-deepDMD','Hammerstein model'),loc = "upper right",fontsize = 14)
# plt.xlabel('Time Index(k)')
# plt.ylabel('States and Outputs[n -step prediction]')
# plt.title('(b)')
# plt.ylim([pl_min,pl_max])
# plt.xlim([0,125])



plt.subplot2grid((2,2), (0,1), colspan=1, rowspan=1)
plt.bar(df_r2_SEQ.index,df_r2_SEQ.mean(axis=1),color = colors[1],label='Sequential oc-deepDMD')
plt.plot(df_r2_DEEPDMD.index,df_r2_DEEPDMD.mean(axis=1),color = colors[0],label='direct oc-deepDMD')
# plt.plot(df_r2_HAM.index,df_r2_HAM.mean(axis=1),color = colors[2],label='Hammerstein model')
plt.xlim([0.5,50.5])
plt.ylim([80,100])
STEPS = 10
plt.legend()
plt.xticks(ticks=np.arange(10, 51, step=STEPS),labels=range(10,51,STEPS))
plt.xlabel('# Prediction Steps')
plt.ylabel('$r^2$(in %)')
plt.title('(c)')
plt.show()
##

plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
for i in range(Phi_SEQ.shape[0]):
    if i in comp_modes_conj_SEQ:
        continue
    elif i in comp_modes_SEQ:
        # plt.plot(Phi[i, :],label = 'lala')
        plt.plot(Phi_SEQ[i,:],label='$\phi_{{{},{}}}(x)$'.format(i+1,comp_modes_conj_SEQ[comp_modes_SEQ.index(i)]+1))
    else:
        plt.plot(Phi_SEQ[i, :], label='$\phi_{}(x)$'.format(i + 1))
plt.legend()
plt.xlabel('Time Index(k)')
plt.ylabel('Evolution of eigenfunctions')
plt.title('(d)')

# plt.subplot2grid((2,2), (1,1), colspan=1, rowspan=1)
# for i in range(Phi_DEEPDMD.shape[0]):
#     if i in comp_modes_conj_DEEPDMD:
#         continue
#     elif i in comp_modes_DEEPDMD:
#         # plt.plot(Phi[i, :],label = 'lala')
#         plt.plot(Phi_DEEPDMD[i,:],label='$\phi_{{{},{}}}(x)$'.format(i+1,comp_modes_conj_DEEPDMD[comp_modes_DEEPDMD.index(i)]+1))
#     else:
#         plt.plot(Phi_DEEPDMD[i, :], label='$\phi_{}(x)$'.format(i + 1))
# plt.legend()
# plt.xlabel('Time Index(k)')
# plt.ylabel('Evolution of eigenfunctions of deeepDMD')
# plt.title('(d)')



plt.savefig('Plots/eg4_GlycolyticOscillator.svg')
plt.show()


