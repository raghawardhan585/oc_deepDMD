##
import numpy as np
import matplotlib.pyplot as plt
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

# eval,evec = np.linalg.eig(dict_params['KxT_num'])
# kmode = np.empty(shape=(len(eval),0))
# N_steps = 100
# for i in range(N_steps):
#     k
##
plt.figure()
for i in range(160,240):
    plt.plot(d[i]['X'][:,0],d[0]['X'][:,1],'.',color = 'skyblue')
    plt.plot(d[i]['X_est_n_step'][:, 0], d[0]['X_est_n_step'][:, 1], color='skyblue')
plt.show()


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
eval,W = np.linalg.eig(Kred.T)
E = np.diag(eval)
# Winv = np.linalg.inv(W)
# Koopman eigenfunctions
Phi = np.matmul(np.matmul(W,Ur.T),psiXp_data)
plt.figure()
plt.plot(Phi.T)
plt.show()
plt.plot(psiXp_data.T)
plt.show()
##
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
    PHI[i, j, :] = np.matmul(np.matmul(W,Ur.T),psiXT_i.T).reshape((1,1,-1))
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

## Calculate the accuracy as a funcation of the number of steps predicted
ls_steps = list(range(1,20,1))
dict_rmse_run = {}
dict_r2_run = {}
dict_DATA_i = oc.scale_data_using_existing_scaler_folder(d[CURVE_NO], SYS_NO)
X_scaled = dict_DATA_i['X']
Y_scaled = dict_DATA_i['Y']
psiX = psixfT.eval(feed_dict={xfT_feed: X_scaled})
for i in ls_steps:  # iterating through each step prediction
    np_psiX_true = psiX[i:, :]
    np_psiX_pred = np.matmul(psiX[:-i, :], np.linalg.matrix_power(KxT_num, i))  # i step prediction at each datapoint
    Y_pred = np.matmul(psiX[:-i, :], np.linalg.matrix_power(KxT_num, i))
    Y_true = Y_scaled[i:, :]
    dict_rmse_run[i] = np.sqrt(np.mean(np.square(np_psiX_true - np_psiX_pred)))
    dict_r2_run[i] = np.max([0, (1 - (np.sum(np.square(np_psiX_true - np_psiX_pred))+0) / (np.sum(np.square(np_psiX_true))+)) * 100])

## Figure 1
CURVE_NO = 0
n_states = d[CURVE_NO]['X'].shape[1]
n_outputs = d[CURVE_NO]['Y'].shape[1]
plt.subplot2grid((5,1), (0,0), colspan=1, rowspan=3)
for i in range(n_states):
    x_scale = 10**np.round(np.log10(np.max(np.abs(d[CURVE_NO]['X'][:,i]))))
    plt.plot(0, color=colors[i],label=('$x_{}$').format(i + 1) + ('$[x10^{}]$').format(np.int(np.log10(x_scale))))
    plt.plot(d[CURVE_NO]['X'][:,i]/x_scale,'.',color = colors[i],linewidth = 2)
    plt.plot(d[CURVE_NO]['X_est_one_step'][:, i]/x_scale,'-.',color=colors[i])
    plt.plot(d[CURVE_NO]['X_est_n_step'][:, i]/x_scale, '--', color=colors[i])
for i in range(n_outputs):
    y_scale = 10 ** np.round(np.log10(np.max(np.abs(d[CURVE_NO]['Y'][:, i]))))
    plt.plot(0, color=colors[i], label=('$y_{}$').format(i + 1) + ('$[x10^{}]$').format(np.int(np.log10(y_scale))))
    plt.plot(d[CURVE_NO]['Y'][:,i]/y_scale,'.',color = colors[n_states+i],linewidth = 2)
    plt.plot(d[CURVE_NO]['Y_est_one_step'][:, i]/y_scale,'-.',color=colors[n_states+i])
    plt.plot(d[CURVE_NO]['Y_est_n_step'][:, i]/y_scale, '--', color=colors[n_states+i])
plt.xlabel('Time Index(k)')
plt.ylabel('States and Outputs')
plt.title('One step vs n step prediction')
plt.legend()
plt.subplot2grid((5,1), (3,0), colspan=1, rowspan=2)
plt.xlabel('# Prediction Steps')
plt.ylabel('$r^2$(in %)')
plt.title('Accuracy of prediction steps')
plt.show()


## Theoretical results
a11 = 0.86
a21 = 0.8
a22 = 0.4
gamma = -0.9

Kt = np.array([[a11,0,0,0,0],[a21,a22,gamma,0,0],[0,0,a11**2,0,0],[0,0,a11*a21,a11*a22,a11*gamma],[0,0,0,0,a11**3]])

