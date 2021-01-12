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
REDUCE_MODES = False

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
with open(sys_folder_name + '/dict_predictions_Hammerstein.pickle', 'rb') as handle:
    d_HAM = pickle.load(handle)[RUN_NO_HAMMERSTEIN]
with open(sys_folder_name + '/dict_predictions_deepDMD.pickle', 'rb') as handle:
    d_DDMD = pickle.load(handle)[RUN_NO_DEEPDMD]
#

##
ls_steps = list(range(1,50,1))
ls_curves = list(range(200, 300)) # test curves

##

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

dict_params = {}


sess1 = tf.InteractiveSession()
dict_params['Seq'] = get_dict_param(run_folder_name,SYS_NO,sess1)
df_r2_SEQ, CURVE_NO = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Seq'])
tf.reset_default_graph()
sess1.close()




sess2 = tf.InteractiveSession()
dict_params['Deep'] = get_dict_param(run_folder_name_DEEPDMD,SYS_NO,sess2)
df_r2_DEEPDMD, _ = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Deep'])
tf.reset_default_graph()
sess2.close()




## Dynamic Modes

Senergy_THRESHOLD = 99.9

def eig_func_through_time(dict_oc_data,dict_params_curr,REDUCED_MODES):
    psiX = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: dict_oc_data['Xp']}).T
    K = dict_params_curr['KxT_num'].T
    if REDUCED_MODES:
        # Minimal POD modes of psiX
        U, S, VT = np.linalg.svd(psiXp_data)
        Senergy = np.cumsum(S ** 2) / np.sum(S ** 2) * 100
        for i in range(len(S)):
            if Senergy[i] > Senergy_THRESHOLD:
                nPC = i + 1
                break
        print('Optimal POD modes chosen : ', nPC)
        Ur = U[:, 0:nPC]
        Kred = np.matmul(np.matmul(Ur.T, K), Ur)

    return

# Required Variables - K, psiX,
CURVE_NO = 0
psiX = d_SEQ[CURVE_NO]['psiX'].T
K = dict_params['KxT_num'].T
psiXp_data = psiX[:,0:-1]
psiXf_data = psiX[:,1:]
if REDUCE_MODES:
    # Minimal POD modes of psiX
    U,S,VT = np.linalg.svd(psiXp_data)
    Senergy = np.cumsum(S**2)/np.sum(S**2)*100
    for i in range(len(S)):
        if Senergy[i] > Senergy_THRESHOLD:
            nPC = i+1
            break
    print('Optimal POD modes chosen : ', nPC)
    Ur = U[:,0:nPC]
    plt.figure()
    _,s,_T = np.linalg.svd(d_SEQ[CURVE_NO]['X'])
    plt.stem(np.arange(len(s)),(np.cumsum(s**2)/np.sum(s**2))*100)
    plt.plot([0,len(s)-1],[100,100])
    plt.title('just X')
    plt.show()
    plt.figure()
    _,s,_T = np.linalg.svd(d_SEQ[CURVE_NO]['psiX'])
    plt.stem(np.arange(len(s)),(np.cumsum(s**2)/np.sum(s**2))*100)
    plt.plot([0,len(s)-1],[100,100])
    plt.title('psiX')
    plt.show()
    # Reduced K - Kred
    Kred = np.matmul(np.matmul(Ur.T,K),Ur)
else:
    Kred = K
    nPC = K.shape[0]
# Eigendecomposition of Kred - Right eigenvectors
# eval,W = np.linalg.eig(Kred.T)
eval,W = np.linalg.eig(Kred)
E = np.diag(eval)
comp_modes =[]
comp_modes_conj =[]
for i1 in range(len(eval)):
    if np.imag(E[i1,i1]) != 0:
        print(i1)
        # Find the complex conjugate
        for i2 in range(i1+1,len(eval)):
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
E = np.real(E)
W = np.real(W)
Winv = np.linalg.inv(W)
# Koopman eigenfunctions
if REDUCE_MODES:
    Phi = np.matmul(E,np.matmul(np.matmul(Winv,Ur.T),psiXp_data))
else:
    Phi = np.matmul(E, np.matmul(Winv, psiXp_data))
plt.figure()
plt.plot(Phi.T)
plt.legend(np.arange(nPC))
plt.show()

# Koopman modes - UW
# Dynamic modes - Lambda*inv(W)*U.T*psiX


# sb.heatmap(W,cmap = "YlOrBr",vmin=0.0)
# plt.show()
# sb.color_palette("YlOrBr")
# sb.color_palette("YlOrBr", as_cmap=True)
# plt.show()

## [R2 function of prediction steps] Calculate the accuracy as a function of the number of steps predicted

# dict_p['psixpT'] = psixpT
# dict_p['psixfT'] = psixfT
# dict_p['xpT_feed'] = xpT_feed
# dict_p['xfT_feed'] = xfT_feed
# dict_p['KxT_num'] = KxT_num
# dict_p['ypT_feed'] = ypT_feed
# dict_p['yfT_feed'] = yfT_feed
# dict_p['WhT_num'] = WhT_num


## Figure 1 - 1 step prediction comparisons


plt.figure(figsize=(20,6))
plt.subplot2grid((6,20), (0,0), colspan=6, rowspan=6)


# CURVE_NO = 0
n_states = d_SEQ[CURVE_NO]['X'].shape[1]
n_outputs = d_SEQ[CURVE_NO]['Y'].shape[1]
pl_max = 0
pl_min = 0
for i in range(n_states):
    x_scale = 10**np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['X'][:,i]))))
    l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(x_scale))))
    plt.plot(d_SEQ[CURVE_NO]['X'][:,i]/x_scale,'.',color = colors[i],linewidth = 5)
    plt.plot(d_SEQ[CURVE_NO]['X_est_one_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
    plt.plot(d_DDMD[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='solid', color=colors[i])
    # plt.plot(d_HAM[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
    pl_max = np.max([pl_max,np.max(d_SEQ[CURVE_NO]['X'][:,i]/x_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['X'][:, i] / x_scale)])
for i in range(n_outputs):
    y_scale = 10 ** np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['Y'][:, i]))))
    plt.plot([], color=colors[n_states+i], label=('$y_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(y_scale))))
    plt.plot(d_SEQ[CURVE_NO]['Y'][:,i]/y_scale, '.',color = colors[n_states+i],linewidth = 5)
    plt.plot(d_SEQ[CURVE_NO]['Y_est_one_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
    plt.plot(d_DDMD[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='solid', color=colors[i])
    # plt.plot(d_HAM[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='solid', color=colors[i])
    pl_max = np.max([pl_max, np.max(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
l1 = plt.legend(loc="upper right")
plt.gca().add_artist(l1)
a1, = plt.plot([],'.',linewidth = 5,label='Truth',color = 'grey')
a2, = plt.plot([], linestyle = 'dashed',linewidth = 1,label='Sequential oc-deepDMD',color = 'grey')
a3, = plt.plot([], linestyle ='solid',linewidth = 1,label='direct oc-deepDMD',color = 'grey')
a4, = plt.plot([], linestyle ='dashdot',linewidth = 1,label='Hammerstein model',color = 'grey')
l2 = plt.legend((a1,a2,a3),('Truth','Sequential oc-deepDMD','direct oc-deepDMD','Hammerstein model'),loc = "lower right")
plt.xlabel('Time Index(k)')
plt.ylabel('States and Outputs')
plt.title('(a)')
plt.ylim([pl_min,pl_max])
# plt.show()

# Figure 2 - n -step prediction comparisons

plt.subplot2grid((6,20), (0,7), colspan=6, rowspan=6)


# CURVE_NO = 0
n_states = d_SEQ[CURVE_NO]['X'].shape[1]
n_outputs = d_SEQ[CURVE_NO]['Y'].shape[1]
pl_max = 0
pl_min = 0
for i in range(n_states):
    x_scale = 10**np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['X'][:,i]))))
    l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(x_scale))))
    plt.plot(d_SEQ[CURVE_NO]['X'][:,i]/x_scale,'.',color = colors[i],linewidth = 5)
    plt.plot(d_SEQ[CURVE_NO]['X_est_n_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
    plt.plot(d_DDMD[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle='solid', color=colors[i])
    # plt.plot(d_HAM[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
    pl_max = np.max([pl_max,np.max(d_SEQ[CURVE_NO]['X'][:,i]/x_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['X'][:, i] / x_scale)])
for i in range(n_outputs):
    y_scale = 10 ** np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['Y'][:, i]))))
    plt.plot([], color=colors[n_states+i], label=('$y_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(y_scale))))
    plt.plot(d_SEQ[CURVE_NO]['Y'][:,i]/y_scale, '.',color = colors[n_states+i],linewidth = 5)
    plt.plot(d_SEQ[CURVE_NO]['Y_est_n_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
    plt.plot(d_DDMD[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle='solid', color=colors[i])
    # plt.plot(d_HAM[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='solid', color=colors[i])
    pl_max = np.max([pl_max, np.max(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
l1 = plt.legend(loc="upper right")
plt.gca().add_artist(l1)
a1, = plt.plot([],'.',linewidth = 5,label='Truth',color = 'grey')
a2, = plt.plot([], linestyle = 'dashed',linewidth = 1,label='Sequential oc-deepDMD',color = 'grey')
a3, = plt.plot([], linestyle ='solid',linewidth = 1,label='direct oc-deepDMD',color = 'grey')
a4, = plt.plot([], linestyle ='dashdot',linewidth = 1,label='Hammerstein model',color = 'grey')
l2 = plt.legend((a1,a2,a3),('Truth','Sequential oc-deepDMD','direct oc-deepDMD','Hammerstein model'),loc = "lower right")
plt.xlabel('Time Index(k)')
plt.ylabel('States and Outputs')
plt.title('(a)')
plt.ylim([pl_min,pl_max])
plt.show()


##
plt.subplot2grid((6,20), (0,7), colspan=6, rowspan=6)
plt.bar(df_r2.index,df_r2.mean(axis=1))
plt.xlim([0.5,50.5])
plt.ylim([80,100])
STEPS = 10
plt.xticks(ticks=np.arange(10, 51, step=STEPS),labels=range(10,51,STEPS))
plt.xlabel('# Prediction Steps')
plt.ylabel('$r^2$(in %)')
plt.title('(b)')



##
plt.subplot2grid((6,20), (0,14), colspan=6, rowspan=6)
for i in range(nPC):
    if i in comp_modes_conj:
        continue
    elif i in comp_modes:
        # plt.plot(Phi[i, :],label = 'lala')
        plt.plot(Phi[i,:],label='$\phi_{{{},{}}}(x)$'.format(i+1,comp_modes_conj[comp_modes.index(i)]+1))
    else:
        plt.plot(Phi[i, :], label='$\phi_{}(x)$'.format(i + 1))
plt.legend()
plt.xlabel('Time Index(k)')
plt.ylabel('Evolution of eigenfunctions')
plt.title('(c)')



plt.savefig('Plots/eg4_GlycolyticOscillator.svg')
plt.show()


