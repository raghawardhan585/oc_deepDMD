##
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle
import tensorflow as tf

# SYS_NO = 10
# RUN_NO = 0
# SYS_NO = 30
# RUN_NO = 47
SYS_NO = 53
RUN_NO = 234
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYS_NO)
run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)

with open(run_folder_name + '/constrainedNN-Model.pickle', 'rb') as handle:
    K = pickle.load(handle)

with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'rb') as handle:
    d = pickle.load(handle)[RUN_NO]

# f = plt.figure()
# ax = f.add_subplot(111, projection='3d')
# for i in range(d['observables'].shape[2]):
#     ax.plot_surface(d['X1'],d['X2'],d['observables'][:,:,i])
# plt.show()
#
# f = plt.figure()
# ax = f.add_subplot(111, projection='3d')
# for i in range(d['eigenfunctions'].shape[2]):
#     ax.plot_surface(d['X1'],d['X2'],d['eigenfunctions'][:,:,i])
# plt.show()

u,s,vT = np.linalg.svd(d[0]['X'])
plt.stem(np.arange(len(s)),(np.cumsum(s**2)/np.sum(s**2))*100)
plt.plot([0,len(s)-1],[100,100])
plt.show()
u,s,vT = np.linalg.svd(d[0]['psiX'])
plt.stem(np.arange(len(s)),(np.cumsum(s**2)/np.sum(s**2))*100)
plt.plot([0,len(s)-1],[100,100])
plt.show()


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