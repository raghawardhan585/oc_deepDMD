##
import RNAseq_data_preprocessing_functions as rnaf
import ocdeepdmd_simulation_examples_helper_functions as oc
from scipy.signal import savgol_filter as sf
from sklearn.preprocessing import MinMaxScaler
import pickle
import random
import numpy as np
import pandas as pd
import os
import shutil
import random
import matplotlib.pyplot as plt
import re
import copy
import itertools
import seaborn as sb
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22
ls_colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# def plot_eigen_functions(): # TODO Yet to be written properly
#     root_run_file = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
#     sess = tf.InteractiveSession()
#     # run_folder_name = root_run_file + '/Sequential/RUN_' + str(run)
#     run_folder_name = root_run_file + '/deepDMD/RUN_' + str(run)
#     saver = tf.compat.v1.train.import_meta_graph(
#         run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
#     saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
#     dict_params = {}
#     dict_params['psixpT'] = tf.get_collection('psixpT')[0]
#     dict_params['psixfT'] = tf.get_collection('psixfT')[0]
#     dict_params['xpT_feed'] = tf.get_collection('xpT_feed')[0]
#     dict_params['xfT_feed'] = tf.get_collection('xfT_feed')[0]
#     dict_params['KxT_num'] = sess.run(tf.get_collection('KxT')[0])
#     # Eig func evolution
#     dict_data_curr = copy.deepcopy(dict_scaled_data['MX'][0])
#     psiX = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: dict_data_curr['XpT']}).T
#     # Phi0 = dict_params_curr['psixpT'].eval(feed_dict={dict_params_curr['xpT_feed']: dict_data_curr['X'][0:1]})
#     # Phi0 = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: dict_data_curr['X']})
#     # For Eigenfunctions and Observables
#     n_observables = len(psiX)
#     sampling_resolution = 0.5
#     x1 = np.arange(-10, 10 + sampling_resolution, sampling_resolution)
#     x2 = np.arange(-10, 10 + sampling_resolution, sampling_resolution)
#     X1, X2 = np.meshgrid(x1, x2)
#
#     eval, W_in = np.linalg.eig(dict_params['KxT_num'].T)
#     E_in = np.diag(eval)
#     E, W, comp_modes, comp_modes_conj = rnaf.resolve_complex_right_eigenvalues(E_in, W_in)
#     Winv = np.linalg.inv(W)
#     koop_modes = W
#     PHI = np.zeros(shape=(X1.shape[0], X1.shape[1], n_observables))
#     PSI = np.zeros(shape=(X1.shape[0], X1.shape[1], n_observables))
#     for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
#         x1_i = X1[i, j]
#         x2_i = X2[i, j]
#         psiXT_i = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: np.array([[x1_i, x2_i]])})
#         PHI[i, j, :] = np.matmul(Winv, psiXT_i.T).reshape((1, 1, -1))
#         PSI[i, j, :] = psiXT_i.reshape((1, 1, -1))
#     title = ''
#     f, ax = plt.subplots(1, 4, figsize=(10, 4))
#     for i in range(4):
#         c = ax[i].pcolor(X1, X2, PHI[:, :, i] / np.max(np.abs(PHI[:, :, i])), cmap='rainbow', vmin=-1, vmax=1)
#
#         # ax[i].set_xlabel('$x_1$\n' + '$\lambda=$' + str(round(np.real(E[i]), 2)))
#         ax[i].set_title(title + '$\phi_{{{}}}(x)$'.format(i + 1))
#     f.colorbar(c, ax=ax[-1])
#     f.show()
#     title = ''
#     f, ax = plt.subplots(1, 4, figsize=(10, 4))
#     for i in range(4):
#         c = ax[i].pcolor(X1, X2, PSI[:, :, i] / np.max(np.abs(PSI[:, :, i])), cmap='rainbow', vmin=-1, vmax=1)
#         # ax[i].set_xlabel('$x_1$\n' + '$\lambda=$' + str(round(np.real(E[i]), 2)))
#         ax[i].set_title(title + '$\psi_{{{}}}(x)$'.format(i + 1))
#     f.colorbar(c, ax=ax[-1])
#     f.show()
#     tf.reset_default_graph()
#     sess.close()
#     return

def plot_output_functions(system_no, method_name, run_no): # TODO Yet to be written properly
    # get the data
    FOLDER_NAME = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/h_OCdeepDMD/System_' + str(
        system_no)
    with open(FOLDER_NAME + '/System_' + str(system_no) + '_ocDeepDMDdata.pickle', 'rb') as handle:
        dict_data = pickle.load(handle)
    YT_scaler = dict_data['Y_scaler']
    XT_scaler = dict_data['X_scaler']
    # ensure that there are only two states and return with a warning
    if not(dict_data['scaled'][0]['XT'].shape[1] == 2):
        print('[WARNING]: h(x) was not plotted because dim(x) != 2')
    else:
        # sample the X - The scaled values are between 0 and 1
        sampling_resolution = 0.01
        x1 = np.arange(0, 1 + sampling_resolution, sampling_resolution)
        x2 = np.arange(0, 1 + sampling_resolution, sampling_resolution)
        X1_s, X2_s = np.meshgrid(x1, x2)
        # get the output function h
        run_folder_name = FOLDER_NAME + '/' + method_name + '/RUN_' + str(run_no)
        sess = tf.InteractiveSession()
        saver = tf.compat.v1.train.import_meta_graph(
            run_folder_name + '/System_' + str(system_no) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
        saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
        dict_params = {}
        dict_params['h'] = tf.get_collection('h_of_x')[0]
        dict_params['xT_feed'] = tf.get_collection('x_feed_h')[0]
        # Compute all h(x) and the unscaled variables of x1,x2 and h
        n_funcs = dict_params['h'].get_shape().as_list()[1]
        H_s = np.zeros(shape=(X1_s.shape[0], X1_s.shape[1], n_funcs))
        H_us = np.zeros(shape=(X1_s.shape[0], X1_s.shape[1], n_funcs))
        X1_us = np.zeros(shape=(X1_s.shape[0], X1_s.shape[1]))
        X2_us = np.zeros(shape=(X1_s.shape[0], X1_s.shape[1]))
        for i, j in itertools.product(range(X1_s.shape[0]), range(X1_s.shape[1])):
            x1_i = X1_s[i, j]
            x2_i = X2_s[i, j]
            h_i = dict_params['h'].eval(feed_dict={dict_params['xT_feed']: np.array([[x1_i, x2_i]])})
            H_s[i, j, :] = h_i.reshape((1, 1, -1))
            H_us[i, j, :] = YT_scaler.inverse_transform(h_i).reshape((1, 1, -1))
            x_us = XT_scaler.inverse_transform(np.array([[x1_i, x2_i]])).reshape(-1)
            X1_us[i, j] = x_us[0]
            X2_us[i, j] = x_us[1]
        h_min = np.min(H_us)
        h_max = np.max(H_us)
        # TODO Plot h(x)
        title = ''
        n_rows = np.int(np.floor(0.25*(n_funcs-1)))
        n_cols = np.int(np.ceil(n_funcs/n_rows))
        f, ax_all = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        ax = ax_all.reshape(-1)
        for i in range(n_funcs):
            c = ax[i].pcolor(X1_us, X2_us, H_us[:, :, i], cmap='rainbow', vmin=h_min, vmax=h_max)
            for j in dict_data['unscaled']:
                for ji in range(len(dict_data['unscaled'][j]['XT'])):
                    ax[i].plot(dict_data['unscaled'][j]['XT'][ji, 0], dict_data['unscaled'][j]['XT'][ji, 1], '.',
                         color=ls_colors[6],markersize = (ji+1)*2)
            # ax[i].set_xlabel('$x_1$\n' + '$\lambda=$' + str(round(np.real(E[i]), 2)))
            ax[i].set_title(title + '$h_{{{}}}(x)$'.format(i + 1))
        f.colorbar(c, ax=ax[-1])
        f.show()
        tf.reset_default_graph()
        sess.close()
    return

def phase_portrait(system_no):
    # get the data
    FOLDER_NAME = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/h_OCdeepDMD/System_' + str(
        system_no)
    with open(FOLDER_NAME + '/System_' + str(system_no) + '_ocDeepDMDdata.pickle', 'rb') as handle:
        dict_data = pickle.load(handle)
    plt.figure(figsize = (3,3))
    for j in dict_data['unscaled']:
        for ji in range(len(dict_data['unscaled'][j]['XT'])):
            # plt.plot(dict_data['unscaled'][j]['XT'][ji,0],dict_data['unscaled'][j]['XT'][ji,1],'.',color = ls_colors[0],linewidth = ji+1)
            plt.plot(dict_data['unscaled'][j]['XT'][ji, 0], dict_data['unscaled'][j]['XT'][ji, 1], '.',
                     color=ls_colors[ji],markersize = (ji+1)*2)
    plt.show()
    return
phase_portrait(system_no = 0)

plot_output_functions(system_no = 0, method_name = 'h_OCdeepDMD', run_no = 1)

