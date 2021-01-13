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
RUN_NO_HAMMERSTEIN_X = 8
RUN_NO_HAMMERSTEIN_Y = 25
RUN_NO_DEEPDMD = 22

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
            np_psiX_pred = np.matmul(psiX[:-i, :],
                                     np.linalg.matrix_power(dict_params_curr['KxT_num'], i))  # i step prediction at each datapoint
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
            for step in range(np.max(ls_steps)):
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


dict_params = {}
sess1 = tf.InteractiveSession()
dict_params['Seq'] = get_dict_param(run_folder_name,SYS_NO,sess1)
df_r2_SEQ, CURVE_NO = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Seq'])
Phi_SEQ,koop_modes_SEQ, comp_modes_SEQ, comp_modes_conj_SEQ = eig_func_through_time(dict_oc_data,dict_data[CURVE_NO],dict_params['Seq'],REDUCED_MODES = True,Senergy_THRESHOLD = 99.9,RIGHT_EIGEN_VECTORS=True,SHOW_PCA_X = False)
tf.reset_default_graph()
sess1.close()

sess2 = tf.InteractiveSession()
dict_params['Deep'] = get_dict_param(run_folder_name_DEEPDMD,SYS_NO,sess2)
df_r2_DEEPDMD, _ = r2_n_step_prediction_accuracy(ls_steps,ls_curves,dict_data,dict_params['Deep'])
Phi_DEEPDMD,koop_modes_DEEPDMD, comp_modes_DEEPDMD, comp_modes_conj_DEEPDMD = eig_func_through_time(dict_oc_data,dict_data[CURVE_NO],dict_params['Deep'],REDUCED_MODES = False,Senergy_THRESHOLD = 99.99,RIGHT_EIGEN_VECTORS=True,SHOW_PCA_X = False)
tf.reset_default_graph()
sess2.close()

df_r2_HAM = r2_n_step_prediction_accuracy_ham(ls_steps,ls_curves,dict_data)



## Saving Requied Stuff

dict_dump = {}
dict_dump['Seq'] = {'d_SEQ':d_SEQ,'df_r2_SEQ':df_r2_SEQ}
dict_dump['Ham'] = {'d_HAM':d_HAM,'df_r2_HAM':df_r2_HAM}
dict_dump['Deep'] = {'d_DDMD':d_DDMD,'df_r2_DEEPDMD':df_r2_DEEPDMD}
dict_dump['CURVE_NO'] = CURVE_NO
with open(sys_folder_name + '/FinalPlotData.pickle','wb') as handle:
    pickle.dump(dict_dump,handle)


# ## Unpack Required Stuff
# with open(sys_folder_name + '/FinalPlotData.pickle','rb') as handle:
#     d = pickle.load(handle)
# d_SEQ = d['Seq']['d_SEQ']
# d_HAM = d['Ham']['d_HAM']
# d_DDMD = d['Deep']['d_DDMD']
# df_r2_SEQ = d['Seq']['df_r2_SEQ']
# df_r2_HAM = d['Ham']['df_r2_HAM']
# df_r2_DEEPDMD = d['Deep']['df_r2_DEEPDMD']
# CURVE_NO = d['CURVE_NO']

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
# CURVE_NO = 265
FONT_SIZE = 14
DOWNSAMPLE = 4
LINE_WIDTH_c_d = 3
TRUTH_MARKER_SIZE = 15
TICK_FONT_SIZE = 10
plt.figure(figsize=(15,10))
ax1 = plt.subplot2grid((13,2), (0,0), colspan=1, rowspan=5)
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1

# CURVE_NO = 0
n_states = d_SEQ[CURVE_NO]['X'].shape[1]
n_outputs = d_SEQ[CURVE_NO]['Y'].shape[1]
pl_max = 0
pl_min = 0
for i in range(n_states):
    x_scale = 10**np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['X'][:,i]))))
    l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(x_scale))))
    ax1.plot(np.arange(0,len(d_SEQ[CURVE_NO]['X']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['X'][0::DOWNSAMPLE,i]/x_scale,'.',color = colors[i],markersize = TRUTH_MARKER_SIZE)
    ax1.plot(d_SEQ[CURVE_NO]['X_est_one_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
    ax1.plot(d_DDMD[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='solid', color=colors[i])
    plt.plot(d_HAM[CURVE_NO]['X_one_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
    pl_max = np.max([pl_max,np.max(d_SEQ[CURVE_NO]['X'][:,i]/x_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['X'][:, i] / x_scale)])
for i in range(n_outputs):
    y_scale = 10 ** np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['Y'][:, i]))))
    ax1.plot([], color=colors[n_states+i], label=('$y_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(y_scale))))
    ax1.plot(np.arange(0,len(d_SEQ[CURVE_NO]['Y']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['Y'][0::DOWNSAMPLE,i]/y_scale, '.',color = colors[n_states+i],markersize = TRUTH_MARKER_SIZE)
    ax1.plot(d_SEQ[CURVE_NO]['Y_est_one_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
    ax1.plot(d_DDMD[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='solid', color=colors[n_states+i])
    plt.plot(d_HAM[CURVE_NO]['Y_one_step'][:, i] / y_scale, linestyle='dashdot', color=colors[n_states+i])
    pl_max = np.max([pl_max, np.max(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
# Shrink current axis by 20%
# box = ax1.get_position()
# ax1.set_position([box.x0, box.y0, box.width * 0.5, box.height*0.8])
# l1 = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),fancybox=True, shadow=True,fontsize = FONT_SIZE,ncol =4)
# plt.gca().add_artist(l1)
# a1, = ax1.plot([],'.',markersize = TRUTH_MARKER_SIZE,label='Truth',color = 'grey')
# a2, = ax1.plot([], linestyle = 'dashed',linewidth = 2,label='Sequential oc-deepDMD',color = 'grey')
# a3, = ax1.plot([], linestyle ='solid',linewidth = 2,label='direct oc-deepDMD',color = 'grey')
# a4, = ax1.plot([], linestyle ='dashdot',linewidth = 2,label='Hammerstein model',color = 'grey')

# l2 = ax1.legend((a1,a2,a3),('Truth','Sequential oc-deepDMD','direct oc-deepDMD','Hammerstein model'),loc='upper center', bbox_to_anchor=(0.5, -0.55),fancybox=True, shadow=True,fontsize = FONT_SIZE,ncol =4)
ax1.set_xlabel('Time Index(k)',fontsize = FONT_SIZE)
ax1.set_ylabel('States and Outputs\n[1 -step prediction]',fontsize = FONT_SIZE)
ax1.set_title('(a)',fontsize = FONT_SIZE)
ax1.set_ylim([pl_min-0.1,pl_max+0.1])
ax1.set_xlim([0,100])
ax1.tick_params(axis ='x', labelsize = TICK_FONT_SIZE)
ax1.tick_params(axis ='y', labelsize = TICK_FONT_SIZE)

# Figure 2 - n -step prediction comparisons
plt.subplot2grid((13,2), (7,0), colspan=1, rowspan=5)
n_states = d_SEQ[CURVE_NO]['X'].shape[1]
n_outputs = d_SEQ[CURVE_NO]['Y'].shape[1]
pl_max = 0
pl_min = 0
for i in range(n_states):
    x_scale = 10**np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['X'][:,i]))))
    l1_i, = plt.plot([], color=colors[i],label=('$x_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(x_scale))))
    plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['X']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['X'][0::DOWNSAMPLE,i]/x_scale,'.',color = colors[i],markersize = TRUTH_MARKER_SIZE)
    plt.plot(d_SEQ[CURVE_NO]['X_est_n_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
    plt.plot(d_DDMD[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle='solid', color=colors[i])
    plt.plot(d_HAM[CURVE_NO]['X_n_step'][:, i] / x_scale, linestyle='dashdot', color=colors[i])
    pl_max = np.max([pl_max,np.max(d_SEQ[CURVE_NO]['X'][:,i]/x_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['X'][:, i] / x_scale)])
for i in range(n_outputs):
    y_scale = 10 ** np.round(np.log10(np.max(np.abs(d_SEQ[CURVE_NO]['Y'][:, i]))))
    plt.plot([], color=colors[n_states+i], label=('$y_{}$').format(i + 1) + ('$[x10^{{{}}}]$').format(np.int(np.log10(y_scale))))
    plt.plot(np.arange(0,len(d_SEQ[CURVE_NO]['Y']))[0::DOWNSAMPLE],d_SEQ[CURVE_NO]['Y'][0::DOWNSAMPLE,i]/y_scale, '.',color = colors[n_states+i],markersize = TRUTH_MARKER_SIZE)
    plt.plot(d_SEQ[CURVE_NO]['Y_est_n_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
    plt.plot(d_DDMD[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle='solid', color=colors[n_states+i])
    plt.plot(d_HAM[CURVE_NO]['Y_n_step'][:, i] / y_scale, linestyle='dashdot', color=colors[n_states+i])
    pl_max = np.max([pl_max, np.max(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
    pl_min = np.min([pl_min, np.min(d_SEQ[CURVE_NO]['Y'][:, i] / y_scale)])
# l1 = plt.legend(loc='lower right',fontsize = 14)
# plt.gca().add_artist(l1)
a1, = plt.plot([],'.',markersize = TRUTH_MARKER_SIZE,label='Truth',color = 'grey')
a2, = plt.plot([], linestyle = 'dashed',linewidth = 1,label='Sequential oc-deepDMD',color = 'grey')
a3, = plt.plot([], linestyle ='solid',linewidth = 1,label='direct oc-deepDMD',color = 'grey')
a4, = plt.plot([], linestyle ='dashdot',linewidth = 1,label='Hammerstein nn-model',color = 'grey')
l1 = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =4)
plt.gca().add_artist(l1)
# l2 = plt.legend((a1,a2,a3),('Truth','Sequential oc-deepDMD','direct oc-deepDMD','Hammerstein model'),loc = "upper right",fontsize = FONT_SIZE)
plt.xlabel('Time Index(k)',fontsize = FONT_SIZE)
plt.ylabel('States and Outputs\n[n -step prediction]',fontsize = FONT_SIZE)
plt.title('(b)',fontsize = FONT_SIZE)
plt.ylim([pl_min-0.1,pl_max+0.1])
plt.xticks(fontsize = TICK_FONT_SIZE)
plt.yticks(fontsize = TICK_FONT_SIZE)
plt.xlim([0,100])


plt.subplot2grid((13,2), (0,1), colspan=1, rowspan=5)
plt.bar(df_r2_SEQ.index,df_r2_SEQ.mean(axis=1),color = colors[1],label='Sequential oc-deepDMD')
plt.plot(df_r2_DEEPDMD.index,df_r2_DEEPDMD.mean(axis=1),color = colors[0],label='direct oc-deepDMD', linewidth = LINE_WIDTH_c_d )
plt.plot(df_r2_HAM.T,color = colors[2],label='Hammerstein nn-model',linewidth = LINE_WIDTH_c_d )
plt.xlim([0.5,50.5])
plt.ylim([85,100])
STEPS = 10
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =2)
plt.xticks(fontsize = TICK_FONT_SIZE)
plt.yticks(fontsize = TICK_FONT_SIZE)
plt.xticks(ticks=np.arange(10, 51, step=STEPS),labels=range(10,51,STEPS))
plt.xlabel('# Prediction Steps',fontsize = FONT_SIZE)
plt.ylabel('$r^2$(in %)',fontsize = FONT_SIZE)
plt.title('(c)',fontsize = FONT_SIZE)



plt.subplot2grid((13,2), (7,1), colspan=1, rowspan=5)
p=0
for i in range(Phi_SEQ.shape[0]):
    if i in comp_modes_conj_SEQ:
        continue
    elif i in comp_modes_SEQ:
        # plt.plot(Phi[i, :],label = 'lala')
        plt.plot(Phi_SEQ[i,:],label='$\phi_{{{},{}}}(x)$'.format(i+1,comp_modes_conj_SEQ[comp_modes_SEQ.index(i)]+1), linewidth = LINE_WIDTH_c_d )
        p = p+1
    else:
        plt.plot(Phi_SEQ[i, :], label='$\phi_{{{}}}(x)$'.format(i + 1), linewidth = LINE_WIDTH_c_d )
        p = p+1
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =np.int(np.ceil(p/2)))
plt.xlabel('Time Index(k)',fontsize = FONT_SIZE)
plt.ylabel('Evolution of eigenfunctions',fontsize = FONT_SIZE)
plt.title('(d)',fontsize = FONT_SIZE)
plt.xticks(fontsize = TICK_FONT_SIZE)
plt.yticks(fontsize = TICK_FONT_SIZE)

# plt.subplot2grid((13,2), (7,1), colspan=1, rowspan=5)
# for i in range(Phi_DEEPDMD.shape[0]):
#     if i in comp_modes_conj_DEEPDMD:
#         continue
#     elif i in comp_modes_DEEPDMD:
#         # plt.plot(Phi[i, :],label = 'lala')
#         plt.plot(Phi_DEEPDMD[i,:],label='$\phi_{{{},{}}}(x)$'.format(i+1,comp_modes_conj_DEEPDMD[comp_modes_DEEPDMD.index(i)]+1), linewidth = 2)
#     else:
#         plt.plot(Phi_DEEPDMD[i, :], label='$\phi_{{{}}}(x)$'.format(i + 1), linewidth = 2)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True,fontsize = TICK_FONT_SIZE,ncol =5)
# plt.xlabel('Time Index(k)',fontsize = FONT_SIZE)
# plt.ylabel('Evolution of eigenfunctions',fontsize = FONT_SIZE)
# plt.title('(d)',fontsize = FONT_SIZE)
# plt.xticks(fontsize = TICK_FONT_SIZE)
# plt.yticks(fontsize = TICK_FONT_SIZE)


plt.savefig('Plots/eg4_GlycolyticOscillator.svg')
plt.show()



##

