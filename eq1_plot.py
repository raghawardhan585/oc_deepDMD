import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle


SYS_NO = 10
RUN_NO = 0
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYS_NO)
run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)

with open(run_folder_name + '/constrainedNN-Model.pickle', 'rb') as handle:
    K = pickle.load(handle)

with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'rb') as handle:
    d = pickle.load(handle)[RUN_NO]

f = plt.figure()
ax = f.add_subplot(111, projection='3d')
for i in range(d['observables'].shape[2]):
    ax.plot_surface(d['X1'],d['X2'],d['observables'][:,:,i])
plt.show()

f = plt.figure()
ax = f.add_subplot(111, projection='3d')
for i in range(d['eigenfunctions'].shape[2]):
    ax.plot_surface(d['X1'],d['X2'],d['eigenfunctions'][:,:,i])
plt.show()

## Getting the Koopman Modes

