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

SYS_NO = 10
RUN_NO = 78
# SYS_NO = 30
# RUN_NO = 47
# SYS_NO = 53
# RUN_NO = 234
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYS_NO)
run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)

with open(run_folder_name + '/constrainedNN-Model.pickle', 'rb') as handle:
    K = pickle.load(handle)

with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'rb') as handle:
    d = pickle.load(handle)[RUN_NO]

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette

# u,s,vT = np.linalg.svd(d[0]['X'])
# plt.stem(np.arange(len(s)),(np.cumsum(s**2)/np.sum(s**2))*100)
# plt.plot([0,len(s)-1],[100,100])
# plt.show()
# u,s,vT = np.linalg.svd(d[0]['psiX'])
# plt.stem(np.arange(len(s)),(np.cumsum(s**2)/np.sum(s**2))*100)
# plt.plot([0,len(s)-1],[100,100])
# plt.show()


## Getting the Koopman Modes

sess = tf.InteractiveSession()
saver = tf.compat.v1.train.import_meta_graph(run_folder_name + '/System_' + str(SYS_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
dict_params = {}
try:
    psixpT = tf.get_collection('psixpT')[0]
    psixfT = tf.get_collection('psixfT')[0]
    xpT_feed = tf.get_collection('xpT_feed')[0]
    xfT_feed = tf.get_collection('xfT_feed')[0]
    KxT = tf.get_collection('KxT')[0]
    KxT_num = sess.run(KxT)
    dict_params['psixpT'] = psixpT
    dict_params['psixfT'] = psixfT
    dict_params['xpT_feed'] = xpT_feed
    dict_params['xfT_feed'] = xfT_feed
    dict_params['KxT_num'] = KxT_num
except:
    print('State info not found')
try:
    ypT_feed = tf.get_collection('ypT_feed')[0]
    yfT_feed = tf.get_collection('yfT_feed')[0]
    dict_params['ypT_feed'] = ypT_feed
    dict_params['yfT_feed'] = yfT_feed
    WhT = tf.get_collection('WhT')[0];
    WhT_num = sess.run(WhT)
    dict_params['WhT_num'] = WhT_num
except:
    print('No output info found')


## Dynamic Modes

Senergy_THRESHOLD = 99.99
# Required Variables - K, psiX,
CURVE_NO = 0
psiX = d[CURVE_NO]['psiX'].T
K = dict_params['KxT_num'].T
psiXp_data = psiX[:,0:-1]
psiXf_data = psiX[:,1:]
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
_,s,_T = np.linalg.svd(d[CURVE_NO]['X'])
plt.stem(np.arange(len(s)),(np.cumsum(s**2)/np.sum(s**2))*100)
plt.plot([0,len(s)-1],[100,100])
plt.title('just X')
plt.show()
plt.figure()
_,s,_T = np.linalg.svd(d[CURVE_NO]['psiX'])
plt.stem(np.arange(len(s)),(np.cumsum(s**2)/np.sum(s**2))*100)
plt.plot([0,len(s)-1],[100,100])
plt.title('psiX')
plt.show()
# Reduced K - Kred
Kred = np.matmul(np.matmul(Ur.T,K),Ur)
# # Eigendecomposition of Kred - Right eigenvectors
# eval,W = np.linalg.eig(Kred)
# E = np.diag(eval)
# Winv = np.linalg.inv(W)
# # Koopman eigenfunctions
# Phi = np.matmul(np.matmul(Winv,Ur.T),psiXp_data)
# plt.figure()
# plt.plot(Phi.T)
# plt.show()

# Eigendecomposition of Kred - Left eigenvectors
# eval,W = np.linalg.eig(Kred.T)
eval,W = np.linalg.eig(Kred)
E = np.diag(eval)
Winv = np.linalg.inv(W)
# Koopman eigenfunctions
Phi = np.matmul(np.matmul(Winv,Ur.T),psiXp_data)
## Eigenfunctions and Observables
n_observables = psiXp_data.shape[0]
sampling_resolution = 0.1
x1 = np.arange(-5, 5 + sampling_resolution, sampling_resolution)
x2 = np.arange(-5, 5 + sampling_resolution, sampling_resolution)
X1, X2 = np.meshgrid(x1, x2)
PHI = np.zeros(shape=(X1.shape[0], X1.shape[1],nPC))
PSI = np.zeros(shape=(X1.shape[0], X1.shape[1],n_observables))
for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
    x1_i = X1[i, j]
    x2_i = X2[i, j]
    psiXT_i = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: np.array([[x1_i, x2_i]])})
    PHI[i, j, :] = np.matmul(np.matmul(Winv,Ur.T),psiXT_i.T).reshape((1,1,-1))
    PSI[i, j, :] = psiXT_i.reshape((1, 1, -1))
## Observables
f,ax = plt.subplots(1,n_observables,figsize = (2*n_observables,1.5))
for i in range(n_observables):
    c = ax[i].pcolor(X1,X2,PSI[:,:,i],cmap='rainbow', vmin=np.min(PSI[:,:,i]), vmax=np.max(PSI[:,:,i]))
    f.colorbar(c,ax = ax[i])
f.show()
## Eigenfunctions
f,ax = plt.subplots(1,nPC,figsize = (2*nPC,1.5))
for i in range(nPC):
    c = ax[i].pcolor(X1,X2,PHI[:,:,i],cmap='rainbow', vmin=np.min(PHI[:,:,i]), vmax=np.max(PHI[:,:,i]))
    f.colorbar(c,ax = ax[i])
f.show()
# Koopman modes - UW


# Dynamic modes - Lambda*inv(W)*U.T*psiX

## [Phase Portrait]

# System Parameters
A = np.array([[0.86,0.],[0.8,0.4]])
gamma = -0.9
# Simulation Parameters
N_data_points = 30
N_CURVES = 240
sys_params = {'A':A , 'gamma': gamma, 'N_data_points': N_data_points}
# Phase Space Data
dict_data = {}
X0 = np.empty(shape=(0, 2))
i=0
for x1,x2 in itertools.product(list(np.arange(-10,11,2)), list(np.arange(-125,16,20))):
    sys_params['x0'] = np.array([[x1,x2]])
    X0 = np.concatenate([X0, sys_params['x0']], axis=0)
    dict_data[i] = oc.sim_sys_1_2(sys_params)
    i = i+1
## [R2 function of prediction steps] Calculate the accuracy as a funcation of the number of steps predicted
CURVE_NO = 0
ls_steps = list(range(1,20,1))
dict_rmse = {}
dict_r2 = {}
for CURVE_NO in range(160,240):
    dict_rmse[CURVE_NO] = {}
    dict_r2[CURVE_NO] = {}
    dict_DATA_i = oc.scale_data_using_existing_scaler_folder(d[CURVE_NO], SYS_NO)
    X_scaled = dict_DATA_i['X']
    Y_scaled = dict_DATA_i['Y']
    psiX = psixfT.eval(feed_dict={xfT_feed: X_scaled})
    for i in ls_steps:  # iterating through each step prediction
        np_psiX_true = psiX[i:, :]
        np_psiX_pred = np.matmul(psiX[:-i, :], np.linalg.matrix_power(KxT_num, i))  # i step prediction at each datapoint
        Y_pred = np.matmul(np_psiX_pred, WhT_num)
        Y_true = Y_scaled[i:, :]
        dict_rmse[CURVE_NO][i] = np.sqrt(np.mean(np.square(np_psiX_true - np_psiX_pred)))
        dict_r2[CURVE_NO][i] = np.max([0, (1 - (np.sum(np.square(np_psiX_true - np_psiX_pred))+np.sum(np.square(Y_true - Y_pred))) / (np.sum(np.square(np_psiX_true))+np.sum(np.square(Y_true)))) * 100])
df_r2 = pd.DataFrame(dict_r2)
print(df_r2)
CHECK_VAL =df_r2.iloc[-1,:].min()
for i in range(160,240):
    if df_r2.loc[df_r2.index[-1],i] == CHECK_VAL:
        CURVE_NO = i
        break
## Figure 1
plt.figure(figsize=(18,7))
plt.subplot2grid((7,18), (0,0), colspan=6, rowspan=4)
alpha = 1.0
epsilon = alpha - 0.01
arrow_length = 0.3
ls_pts = list(range(0,1))
for i in list(dict_data.keys())[0:]:
    for j in ls_pts:
        if np.abs(dict_data[i]['X'][j, 0]) > 1 or j==0:
            plt.plot(dict_data[i]['X'][j, 0], dict_data[i]['X'][j, 1], 'o',color='salmon',fillstyle='none',markersize=5)
    plt.plot(dict_data[i]['X'][:, 0], dict_data[i]['X'][:, 1], color='tab:blue',linewidth=0.3)
    if np.mod(i,1)==0:
        for j in ls_pts:
            dist = np.sqrt((dict_data[i]['X'][j, 0] - dict_data[i]['X'][j + 1, 0]) ** 2 + (dict_data[i]['X'][j, 1] - dict_data[i]['X'][j + 1, 1]) ** 2)
            x = dict_data[i]['X'][j, 0]
            y = dict_data[i]['X'][j, 1]
            dx = (dict_data[i]['X'][j + 1, 0] - dict_data[i]['X'][j, 0]) * arrow_length
            dy = (dict_data[i]['X'][j + 1, 1] - dict_data[i]['X'][j, 1]) * arrow_length
            # print(x,' ',y,' ',dist)
            if dist<2:
                plt.arrow(x,y,dx,dy,head_width = 0.1,head_length=0.5,alpha=1,color='tab:green')
            else:
                plt.arrow(x, y, dx, dy, head_width=0.3, head_length=3, alpha=1, color='tab:green')
plt.xlabel('state $x_1$')
plt.ylabel('state $x_2$')
plt.plot([0],[0],'o',color='tab:red',markersize=10)
plt.xlim([-10,10])
plt.ylim([-126,16])
plt.title('(a)')


# CURVE_NO = 0
plt.subplot2grid((7,18), (0,6), colspan=6, rowspan=4)
n_states = d[CURVE_NO]['X'].shape[1]
n_outputs = d[CURVE_NO]['Y'].shape[1]
for i in range(n_states):
    x_scale = 10**np.round(np.log10(np.max(np.abs(d[CURVE_NO]['X'][:,i]))))
    l1_i, = plt.plot(0, color=colors[i],label=('$x_{}$').format(i + 1) + ('$[x10^{}]$').format(np.int(np.log10(x_scale))))
    plt.plot(d[CURVE_NO]['X'][:,i]/x_scale,'.',color = colors[i],linewidth = 5)
    plt.plot(d[CURVE_NO]['X_est_one_step'][:, i]/x_scale,linestyle = 'dotted',color=colors[i])
    plt.plot(d[CURVE_NO]['X_est_n_step'][:, i]/x_scale,linestyle =  'dashed', color=colors[i])
for i in range(n_outputs):
    y_scale = 10 ** np.round(np.log10(np.max(np.abs(d[CURVE_NO]['Y'][:, i]))))
    plt.plot(0, color=colors[i], label=('$y_{}$').format(i + 1) + ('$[x10^{}]$').format(np.int(np.log10(y_scale))))
    plt.plot(d[CURVE_NO]['Y'][:,i]/y_scale, '.',color = colors[n_states+i],linewidth = 5)
    plt.plot(d[CURVE_NO]['Y_est_one_step'][:, i]/y_scale, linestyle ='dotted',color=colors[n_states+i])
    plt.plot(d[CURVE_NO]['Y_est_n_step'][:, i]/y_scale, linestyle = 'dashed', color=colors[n_states+i])
l1 = plt.legend(loc="upper right")
plt.gca().add_artist(l1)
a1, = plt.plot(0,'.',linewidth = 5,label='Truth',color = 'grey')
a2, = plt.plot(0, linestyle ='dotted',linewidth = 1,label='1-step',color = 'grey')
a3, = plt.plot(0, linestyle = 'dashed',linewidth = 1,label='n-step',color = 'grey')
l2 = plt.legend((a1,a2,a3),('Truth','1-step','n-step'),loc = "lower right")
plt.xlabel('Time Index(k)')
plt.ylabel('States and Outputs')
plt.title('(b)')


plt.subplot2grid((7,18), (0,12), colspan=6, rowspan=4)
for i in range(nPC):
    plt.plot(Phi[i,:],label='$\phi_{}(x)$'.format(i+1))
plt.legend()
plt.xlabel('Time Index(k)')
plt.ylabel('Evolution of eigenfunctions')
plt.title('(c)')


plt.subplot2grid((7,18), (4,0), colspan=3, rowspan=2)
plt.bar(df_r2.index,df_r2.mean(axis=1))
plt.xlim([0.5,19.5])
STEPS = 3
plt.xticks(ticks=np.arange(1, 20, step=STEPS),labels=range(1,20,STEPS))
plt.xlabel('# Prediction Steps')
plt.ylabel('$r^2$(in %)')
plt.title('(d)')


for i in range(nPC):
    f = plt.subplot2grid((7,18), (4, 3*(i+1)), colspan=3, rowspan=2)
    c = f.pcolor(X1,X2,PHI[:,:,i],cmap='rainbow', vmin=np.min(PHI[:,:,i]), vmax=np.max(PHI[:,:,i]))
    if i!=0:
        plt.yticks([])
    else:
        plt.ylabel('$x_2$')
    if i ==2:
        plt.title('(e) \n $\lambda_{} =$'.format(i+1) + str(round(eval[i],3)))
    else:
        plt.title('$\lambda_{} =$'.format(i+1) + str(round(eval[i],3)))
    plt.colorbar(c,ax = f)
    plt.xlabel('$x_1$')

plt.show()

## Theoretical results
a11 = 0.86
a21 = 0.8
a22 = 0.4
gamma = -0.9

Kt = np.array([[a11,0,0,0,0],[a21,a22,gamma,0,0],[0,0,a11**2,0,0],[0,0,a11*a21,a11*a22,a11*gamma],[0,0,0,0,a11**3]])

