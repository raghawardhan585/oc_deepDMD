
# Required Packages
import pickle  # for data I/O
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import math
import random
import tensorflow as tf
import os
import shutil
import pandas as pd
import copy
from sklearn.linear_model import LinearRegression

# Default Parameters
DEVICE_NAME = '/cpu:0'
RUN_NUMBER = 5
SYSTEM_NO = 0

# Neural network parameters
activation_flag = 2  # sets the activation function type to RELU[0], ELU[1], SELU[2] (initialized a certain way,dropout has to be done differently) , or tanh()
x_deep_dict_size = 0
n_x_nn_layers = 4  # x_max_layers 3 works well
n_x_nn_nodes = 10  # max width_limit -4 works well

#  Deep Learning Optimization Parameters
keep_prob = 1.0  # keep_prob = 1-dropout probability
res_net = 0  # Boolean condition on whether to use a resnet connection.
regularization_lambda = 0
DISPLAY_SAMPLE_RATE_EPOCH = 500

# Learning Parameters
batch_size = 72#40#24#36
ls_dict_training_params = []
dict_training_params = {'step_size_val': 0.5, 'train_error_threshold': float(1e-20),'valid_error_threshold': float(1e-6), 'max_epochs': 2000, 'batch_size': batch_size} #80000
ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 00.3, 'train_error_threshold': float(1e-10),'valid_error_threshold': float(1e-6), 'max_epochs': 10000, 'batch_size': batch_size}
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.1, 'train_error_threshold': float(1e-10), 'valid_error_threshold': float(1e-7), 'max_epochs': 10000, 'batch_size': batch_size}
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.08, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 20000, 'batch_size': batch_size}
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.05, 'train_error_threshold': float(1e-10), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': batch_size}
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.01, 'train_error_threshold': float(1e-10), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': batch_size}
# ls_dict_training_params.append(dict_training_params)

ls_dict_training_params_h = copy.deepcopy(ls_dict_training_params)
ls_dict_training_params_psi = copy.deepcopy(ls_dict_training_params)

sess = tf.InteractiveSession()

def estimate_K_stability(Kx, sess, print_Kx=False):
    Kx_num = sess.run(Kx)
    np.linalg.eigvals(Kx_num)
    Kx_num_eigval_mod = np.abs(np.linalg.eigvals(Kx_num))
    if print_Kx:
        print(Kx_num)
    print('Eigen values: ')
    print(Kx_num_eigval_mod)
    unstable = True
    if np.max(Kx_num_eigval_mod) > 1:
        print('[COMP] The identified Koopman operator is UNSTABLE with ', np.sum(np.abs(Kx_num_eigval_mod) > 1),
              'eigenvalues greater than 1')
    else:
        print('[COMP] The identified Koopman operator is STABLE')
        unstable = False
    return unstable

def load_pickle_data_find_h(file_path):
    with open(file_path,'rb') as handle:
        dict_data = pickle.load(handle)
    # dict_data.keys() = ['scaled' - [0,1,...15], 'unscaled' - [0,1,...16], 'index' - ['train','valid','test'], 'X_scaler', 'Y_scaler']
    # Assume data is input in the transposed form
    num_states = dict_data['scaled'][0]['XT'].shape[1]
    num_outputs = dict_data['scaled'][0]['YT'].shape[1]
    dict_train = {'X': np.empty(shape=(0, num_states)), 'Y': np.empty(shape=(0, num_outputs))}
    for i in dict_data['index']['train']:
        dict_train['X'] = np.concatenate([dict_train['X'], dict_data['scaled'][i]['XT']], axis=0)
        dict_train['Y'] = np.concatenate([dict_train['Y'], dict_data['scaled'][i]['YT']], axis=0)
    dict_valid = {'X': np.empty(shape=(0, num_states)), 'Y': np.empty(shape=(0, num_outputs))}
    for i in dict_data['index']['valid']:
        dict_valid['X'] = np.concatenate([dict_valid['X'], dict_data['scaled'][i]['XT']], axis=0)
        dict_valid['Y'] = np.concatenate([dict_valid['Y'], dict_data['scaled'][i]['YT']], axis=0)
    # Hidden layer list creation for state dynamics
    x_hidden_vars_list = np.asarray([n_x_nn_nodes] * n_x_nn_layers)
    x_hidden_vars_list[-1] = dict_train['Y'].shape[1]
    print('============================================')
    print('For training h(x)')
    print('============================================')
    print('[INFO] Train X.shape :', dict_train['X'].shape)
    print('[INFO] Train Y.shape :', dict_train['Y'].shape)
    print('[INFO] Valid X.shape :', dict_valid['X'].shape)
    print('[INFO] Valid Y.shape :', dict_valid['Y'].shape)
    print("[INFO] STATE - hidden_vars_list: " + repr(x_hidden_vars_list))
    print('============================================')
    return dict_train, dict_valid, x_hidden_vars_list

def load_pickle_data_deepDMD(file_path):
    with open(file_path,'rb') as handle:
        dict_data = pickle.load(handle)
    # dict_data.keys() = ['scaled' - [0,1,...15], 'unscaled' - [0,1,...16], 'index' - ['train','valid','test'], 'X_scaler', 'Y_scaler']
    # Assume data is input in the transposed form
    num_states = dict_data['scaled'][0]['XT'].shape[1]
    dict_train = {'Xp': np.empty(shape=(0, num_states)), 'Xf': np.empty(shape=(0, num_states))}
    for i in dict_data['index']['train']:
        dict_train['Xp'] = np.concatenate([dict_train['Xp'], dict_data['scaled'][i]['XT'][0:-1,:]], axis=0)
        dict_train['Xf'] = np.concatenate([dict_train['Xf'], dict_data['scaled'][i]['XT'][1:,:]], axis=0)
    dict_valid = {'Xp': np.empty(shape=(0, num_states)), 'Xf': np.empty(shape=(0, num_states))}
    for i in dict_data['index']['valid']:
        dict_valid['Xp'] = np.concatenate([dict_valid['Xp'], dict_data['scaled'][i]['XT'][0:-1, :]], axis=0)
        dict_valid['Xf'] = np.concatenate([dict_valid['Xf'], dict_data['scaled'][i]['XT'][1:, :]], axis=0)
    # Hidden layer list creation for state dynamics
    x_hidden_vars_list = np.asarray([n_x_nn_nodes] * n_x_nn_layers)
    x_hidden_vars_list[-1] = x_deep_dict_size
    print('============================================')
    print('deepDMD information')
    print('============================================')
    print('[INFO] Train Xp.shape :', dict_train['Xp'].shape)
    print('[INFO] Train Xf.shape :', dict_train['Xf'].shape)
    print('[INFO] Valid Xp.shape :', dict_valid['Xp'].shape)
    print('[INFO] Valid Xf.shape :', dict_valid['Xf'].shape)
    print("[INFO] STATE - hidden_vars_list: " + repr(x_hidden_vars_list))
    print('============================================')
    return dict_train, dict_valid, x_hidden_vars_list

def weight_variable(shape):
    std_dev = math.sqrt(3.0 / (shape[0] + shape[1]))
    return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=std_dev, dtype=tf.float32))
def bias_variable(shape):
    std_dev = math.sqrt(3.0 / shape[0])
    return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=std_dev, dtype=tf.float32))
def initialize_Wblist(n_u, hv_list):
    # INITIALIZATION - going from input to first layer
    W_list = [weight_variable([n_u, hv_list[0]])]
    b_list = [bias_variable([hv_list[0]])]
    # PROPAGATION - consecutive layers
    for k in range(1,len(hv_list)):
        W_list.append(weight_variable([hv_list[k - 1], hv_list[k]]))
        b_list.append(bias_variable([hv_list[k]]))
    return W_list, b_list

def initialize_tensorflow_graph(param_list,u, state_inclusive=False,add_bias=False,constant_graph = False, h = None):
    # res_net = param_list['res_net'] --- This variable is not used!
    # TODO - remove the above variable if not required at all
    # u is the input of the neural network
    z_list = []
    n_depth = len(param_list['hidden_var_list'])
    # Initialization
    if constant_graph:
        first_layer_W = tf.constant(param_list['W_list'][0],dtype=tf.dtypes.float32)
        first_layer_b = tf.constant(param_list['b_list'][0], dtype=tf.dtypes.float32)
    else:
        first_layer_W = param_list['W_list'][0]
        first_layer_b = param_list['b_list'][0]
    if param_list['activation flag'] == 1:  # RELU
        z_list.append(tf.nn.dropout(tf.nn.relu(tf.matmul(u, first_layer_W) + first_layer_b), param_list['keep_prob']))
    if param_list['activation flag']== 2:  # ELU
        z_list.append(tf.nn.dropout(tf.nn.elu(tf.matmul(u, first_layer_W) + first_layer_b), param_list['keep_prob']))
    if param_list['activation flag'] == 3:  # tanh
        z_list.append(tf.nn.dropout(tf.nn.tanh(tf.matmul(u, first_layer_W) + first_layer_b), param_list['keep_prob']))
    # Propagation
    for k in range(1, n_depth-1):
        if constant_graph:
            prev_layer_output = tf.matmul(z_list[k - 1],tf.constant(param_list['W_list'][k], dtype=tf.dtypes.float32)) + tf.constant(param_list['b_list'][k], dtype=tf.dtypes.float32)
        else:
            prev_layer_output = tf.matmul(z_list[k - 1], param_list['W_list'][k]) + param_list['b_list'][k]
        if param_list['activation flag'] == 1: # RELU
            z_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output), param_list['keep_prob']));
        if param_list['activation flag'] == 2: # ELU
            z_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output), param_list['keep_prob']));
        if param_list['activation flag'] == 3: # tanh
            z_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output), param_list['keep_prob']));
    # Termination
    try:
        if constant_graph:
            z_list.append(tf.matmul(z_list[n_depth - 2], tf.constant(param_list['W_list'][n_depth - 1],dtype=tf.dtypes.float32)) + tf.constant(param_list['b_list'][n_depth - 1], dtype=tf.dtypes.float32))
        else:
            z_list.append(tf.matmul(z_list[n_depth-2], param_list['W_list'][n_depth-1]) + param_list['b_list'][n_depth-1])
    except:
        print('[WARNING]: There is no neural network initialized')
    # State inclusion
    if state_inclusive:
        y = tf.concat([u, z_list[-1]], axis=1)
    else:
        y = z_list[-1]
    # Adding the constraint h(x)
    if not (h == None):
        y = tf.concat([y,h],axis=1)
    # Bias addition
    if add_bias:
        y = tf.concat([y, tf.ones(shape=(tf.shape(y)[0], 1))], axis=1)
    result = sess.run(tf.global_variables_initializer())
    return z_list, y

def display_train_params(dict_run_params):
    print('======================================')
    print('CURRENT TRAINING PARAMETERS')
    print('======================================')
    print('Step Size Value            : ', dict_run_params['step_size_val'])
    print('Train Error Threshold      : ', dict_run_params['train_error_threshold'])
    print('Validation Error Threshold : ', dict_run_params['valid_error_threshold'])
    print('Maximum number of Epochs   : ', dict_run_params['max_epochs'])
    print('Batch Size   : ', dict_run_params['batch_size'])
    print('--------------------------------------')
    return

def generate_hyperparam_entry(feed_dict_train, feed_dict_valid, dict_model_metrics, n_epochs_run, dict_run_params,x_params_list):
    training_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_train)
    validation_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_valid)
    training_accuracy = dict_model_metrics['accuracy'].eval(feed_dict=feed_dict_train)
    validation_accuracy = dict_model_metrics['accuracy'].eval(feed_dict=feed_dict_valid)
    dict_hp = {}
    dict_hp['x_hidden_variable_list'] = x_params_list['hidden_var_list'] # TODO add x_hidden_vars_list to dict_run_params
    dict_hp['activation flag'] = x_params_list['activation flag'] # TODO add activation_flag = 1,2,3 to dict_run_params
    dict_hp['activation function'] = None
    if x_params_list['activation flag'] == 1:
        dict_hp['activation function'] = 'relu'
    elif x_params_list['activation flag'] == 2:
        dict_hp['activation function'] = 'elu'
    elif x_params_list['activation flag'] == 3:
        dict_hp['activation function'] = 'tanh'
    dict_hp['no of epochs'] = n_epochs_run
    dict_hp['batch size'] = dict_run_params['batch_size']
    dict_hp['step size'] = dict_run_params['step_size_val']
    dict_hp['training error'] = training_error
    dict_hp['validation error'] = validation_error
    dict_hp['r^2 training accuracy'] = training_accuracy
    dict_hp['r^2 validation accuracy'] = validation_accuracy
    return dict_hp

def objective_func_find_h(dict_feed,dict_psi): # TODO Modify this to just capture the output h(x)
    # Prior computations
    Yf_prediction_error = dict_feed['y'] - dict_psi['x']
    SST = tf.math.reduce_sum(tf.math.square(dict_feed['y'] - tf.math.reduce_mean(dict_feed['y'], axis=0)))
    SSE = tf.math.reduce_sum(tf.math.square(Yf_prediction_error))
    # The output
    dict_model_perf_metrics = {}
    dict_model_perf_metrics['loss_fn'] = tf.math.reduce_mean(tf.math.square(Yf_prediction_error))
    dict_model_perf_metrics['optimizer'] = tf.train.AdagradOptimizer(dict_feed['step_size']).minimize(dict_model_perf_metrics['loss_fn'])
    dict_model_perf_metrics['MSE'] = tf.math.reduce_mean(tf.math.square(Yf_prediction_error))
    dict_model_perf_metrics['accuracy'] = (1 - tf.divide(SSE, SST)) * 100
    sess.run(tf.global_variables_initializer())
    return dict_model_perf_metrics

def objective_func_find_psi(dict_feed, dict_psi, dict_K): # TODO Modify this to just capture the output psi(x)
    # Prior computations
    psiXf_predicted = tf.matmul(dict_psi['xp'], dict_K['Kx'])
    psiXf_prediction_error = dict_psi['xf'] - psiXf_predicted
    SST = tf.math.reduce_sum(tf.math.square(dict_psi['xf'] - tf.math.reduce_mean(dict_psi['xf'], axis=0)))
    SSE = tf.math.reduce_sum(tf.math.square(psiXf_prediction_error))
    # The output
    dict_model_perf_metrics = {}
    dict_model_perf_metrics['loss_fn'] = tf.math.reduce_mean(tf.math.square(psiXf_prediction_error))
    dict_model_perf_metrics['optimizer'] = tf.train.AdagradOptimizer(dict_feed['step_size']).minimize( dict_model_perf_metrics['loss_fn'])
    dict_model_perf_metrics['MSE'] = tf.math.reduce_mean(tf.math.square(psiXf_prediction_error))
    dict_model_perf_metrics['accuracy'] = (1 - tf.divide(SSE, SST)) * 100
    sess.run(tf.global_variables_initializer())
    return dict_model_perf_metrics

def static_train_net(dict_train, dict_valid, dict_feed, ls_dict_training_params, dict_model_metrics, all_histories = {'train error': [], 'validation error': []}, dict_run_info = {}):
    feed_dict_train = {dict_feed['xpT']: dict_train['Xp'],dict_feed['xfT']: dict_train['Xf'],dict_feed['ypT']: dict_train['Yp'],dict_feed['yfT']: dict_train['Yf']}
    feed_dict_valid = {dict_feed['xpT']: dict_valid['Xp'],dict_feed['xfT']: dict_valid['Xf'],dict_feed['ypT']: dict_valid['Yp'],dict_feed['yfT']: dict_valid['Yf']}
    # --------
    try :
        run_info_index = list(dict_run_info.keys())[-1]
    except:
        run_info_index = 0
    for dict_train_params_i in ls_dict_training_params:
        display_train_params(dict_train_params_i)
        all_histories, n_epochs_run = train_net_v2(dict_train,feed_dict_train, feed_dict_valid, dict_feed, dict_model_metrics, dict_train_params_i, all_histories)
        dict_run_info[run_info_index] = generate_hyperparam_entry(feed_dict_train, feed_dict_valid,dict_model_metrics,n_epochs_run, dict_train_params_i)
        print('Current Training Error  :', dict_run_info[run_info_index]['training error'])
        print('Current Validation Error      :', dict_run_info[run_info_index]['validation error'])
        run_info_index += 1
        try:
            estimate_K_stability(KxT)
        except:
            continue
    return all_histories, dict_run_info

def train_net_v2(dict_train, feed_dict_train, feed_dict_valid, dict_feed, dict_model_metrics, dict_run_params, all_histories):
    # -----------------------------
    # Initialization
    # -----------------------------
    N_train_samples = len(dict_train['Xp'])
    runs_per_epoch = int(np.ceil(N_train_samples / dict_run_params['batch_size']))
    epoch_i = 0
    training_error = 100
    validation_error = 100
    # -----------------------------
    # Actual training
    # -----------------------------
    while ((epoch_i < dict_run_params['max_epochs']) and (training_error > dict_run_params['train_error_threshold']) and (validation_error > dict_run_params['valid_error_threshold'])):
        epoch_i += 1
        # Re initializing the training indices
        all_train_indices = list(range(N_train_samples))
        # Random sort of the training indices
        random.shuffle(all_train_indices)
        for run_i in range(runs_per_epoch):
            if run_i != runs_per_epoch - 1:
                train_indices = all_train_indices[run_i * dict_run_params['batch_size']:(run_i + 1) * dict_run_params['batch_size']]
            else:
                # Last run with the residual data
                train_indices = all_train_indices[run_i * dict_run_params['batch_size']: N_train_samples]
            feed_dict_train_curr = {dict_feed['xpT']: dict_train['Xp'][train_indices], dict_feed['xfT']: dict_train['Xf'][train_indices],dict_feed['ypT']: dict_train['Yp'][train_indices], dict_feed['yfT']: dict_train['Yf'][train_indices], dict_feed['step_size']: dict_run_params['step_size_val']}
            dict_model_metrics['optimizer'].run(feed_dict=feed_dict_train_curr)
        # After training 1 epoch
        training_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_train)
        validation_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_valid)
        all_histories['train error'].append(training_error)
        all_histories['validation error'].append(validation_error)
        if np.mod(epoch_i, DISPLAY_SAMPLE_RATE_EPOCH) == 0:
            print('Epoch No: ', epoch_i, ' |   Training error: ', training_error)
            print('Validation error: '.rjust(len('Epoch No: ' + str(epoch_i) + ' |   Validation error: ')),validation_error)
            # estimate_K_stability(Kx)
            print('---------------------------------------------------------------------------------------------------')
    return all_histories, epoch_i


def get_best_K_DMD(Xp_train,Xf_train,Xp_valid,Xf_valid):
    Xp_train = Xp_train.T
    Xf_train = Xf_train.T
    Xp_valid = Xp_valid.T
    Xf_valid = Xf_valid.T
    U,S,Vh = np.linalg.svd(Xp_train)
    V = Vh.T.conj()
    Uh = U.T.conj()
    A_hat = np.zeros(shape = U.shape)
    ls_error_train = []
    ls_error_valid = []
    for i in range(len(S)):
        A_hat = A_hat + (1/S[i])*np.matmul(np.matmul(Xf_train,V[:,i:i+1]),Uh[i:i+1,:])
        ls_error_train.append(np.mean(np.square((Xf_train - np.matmul(A_hat,Xp_train)))))
        ls_error_valid.append(np.mean(np.square((Xf_valid - np.matmul(A_hat, Xp_valid)))))
    ls_error = np.array(ls_error_train) + np.array(ls_error_valid)
    nPC_opt = np.where(ls_error==np.min(ls_error))[0][0] + 1
    A_hat_opt = np.zeros(shape = U.shape)
    for i in range(nPC_opt):
        A_hat_opt = A_hat_opt + (1/S[i])*np.matmul(np.matmul(Xf_train,V[:,i:i+1]),Uh[i:i+1,:])
    print('Optimal Linear model Error: ',np.mean(np.square((Xf_train - np.matmul(A_hat_opt, Xp_train)))))
    return  A_hat_opt.T
def get_best_K_DMD2(X, Y, fit_intercept = True):
    if fit_intercept:
        # used for initializing K
        lin_model = LinearRegression().fit(X, Y)
        AT_opt = lin_model.coef_.T
        bT_opt = lin_model.intercept_#.reshape(1,-1)
        return AT_opt,bT_opt
    else:
        # used for initializing Wh
        lin_model = LinearRegression(fit_intercept= False).fit(X, Y)
        AT_opt = lin_model.coef_.T
        return AT_opt
##


# Main Block
data_directory = 'h_OCdeepDMD_data/'
data_suffix = 'System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'

# CMD Line Argument (Override) Inputs:
# TODO - Rearrange this section
import sys
if len(sys.argv)>1:
    DEVICE_NAME = sys.argv[1]
    if DEVICE_NAME not in ['/cpu:0','/gpu:0','/gpu:1','/gpu:2','/gpu:3']:
        DEVICE_NAME = '/cpu:0'
if len(sys.argv)>2:
    SYSTEM_NO = sys.argv[2]
    data_suffix = 'System_'+ str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
if len(sys.argv) > 3:
    RUN_NUMBER = np.int(sys.argv[3])
if len(sys.argv) > 4:
    x_deep_dict_size = np.int(sys.argv[4])
if len(sys.argv)>5:
    n_x_nn_layers = np.int(sys.argv[5])
if len(sys.argv)>6:
    n_x_nn_nodes = np.int(sys.argv[6])


## Finding h(x)
data_file = data_directory + data_suffix
# Dataset to find h(x) - we just need x and y
dict_train, dict_valid, h_x_hidden_vars_list = load_pickle_data_find_h(data_file)
num_states = dict_train['X'].shape[1]
num_outputs = dict_train['Y'].shape[1]
with tf.device(DEVICE_NAME):
    xp_feed = tf.placeholder(tf.float32, shape=[None, num_states])
    yp_feed = tf.placeholder(tf.float32, shape=[None, num_outputs])
    step_size_feed = tf.placeholder(tf.float32, shape=[])
    # Tensorflow graph
    h_Wx_list, h_bx_list = initialize_Wblist(num_states, h_x_hidden_vars_list)
    h_x_params_list = {'n_base_states': num_states, 'hidden_var_list': h_x_hidden_vars_list, 'W_list': h_Wx_list,
                     'b_list': h_bx_list, 'keep_prob': keep_prob, 'activation flag': activation_flag, 'res_net': res_net}
    sess.run(tf.global_variables_initializer())
    # Psi variables
    _, h_of_x = initialize_tensorflow_graph(h_x_params_list, xp_feed, add_bias=False, state_inclusive=False, constant_graph = False, h = None)
    dict_feed_h = {'x': xp_feed, 'y': yp_feed, 'step_size': step_size_feed}
    h_func = {'x': h_of_x}
    dict_model_metrics_h = objective_func_find_h(dict_feed_h, h_func)
    all_histories_h = {'train error': [], 'validation error': [], 'train MSE': [], 'valid MSE': []}
    dict_run_info_h = {}
    feed_dict_train = {dict_feed_h['x']: dict_train['X'], dict_feed_h['y']: dict_train['Y']}
    feed_dict_valid = {dict_feed_h['x']: dict_valid['X'], dict_feed_h['y']: dict_valid['Y']}
    # ================================================================================================
    # -------------------------------         Training code             ------------------------------
    # ================================================================================================
    run_info_index = 0
    for dict_train_params_i in ls_dict_training_params_h:
        display_train_params(dict_train_params_i)
        # -----------------------------
        # Initialization
        # -----------------------------
        N_train_samples = len(dict_train['X'])
        runs_per_epoch = int(np.ceil(N_train_samples / dict_train_params_i['batch_size']))
        epoch_i = 0
        training_error = 100
        validation_error = 100
        # -----------------------------
        # Actual training
        # -----------------------------
        while ((epoch_i < dict_train_params_i['max_epochs']) and (training_error > dict_train_params_i['train_error_threshold']) and (validation_error > dict_train_params_i['valid_error_threshold'])):
            epoch_i += 1
            # Re initializing the training indices
            all_train_indices = list(range(N_train_samples))
            # Random sort of the training indices
            random.shuffle(all_train_indices)
            for run_i in range(runs_per_epoch):
                try:
                    train_indices = all_train_indices[run_i * dict_train_params_i['batch_size']:(run_i + 1) * dict_train_params_i['batch_size']]
                except:
                    # Last run with the residual data
                    train_indices = all_train_indices[run_i * dict_train_params_i['batch_size']: N_train_samples]
                feed_dict_train_curr = {dict_feed_h['x']: dict_train['X'][train_indices], dict_feed_h['y']: dict_train['Y'][train_indices], dict_feed_h['step_size']: dict_train_params_i['step_size_val']}
                dict_model_metrics_h['optimizer'].run(feed_dict=feed_dict_train_curr)
            # After training 1 epoch
            training_error = dict_model_metrics_h['loss_fn'].eval(feed_dict=feed_dict_train)
            validation_error = dict_model_metrics_h['loss_fn'].eval(feed_dict=feed_dict_valid)
            all_histories_h['train error'].append(training_error)
            all_histories_h['validation error'].append(validation_error)
            if np.mod(epoch_i, DISPLAY_SAMPLE_RATE_EPOCH) == 0:
                print('Epoch No: ', epoch_i, ' |   Training error: ', training_error)
                print('Validation error: '.rjust(len('Epoch No: ' + str(epoch_i) + ' |   Validation error: ')), validation_error)
                print('-----------------------------------------------------------------------------------------------')
        dict_run_info_h[run_info_index] = generate_hyperparam_entry(feed_dict_train, feed_dict_valid, dict_model_metrics_h, epoch_i, dict_train_params_i,h_x_params_list)
        print('Current Training Error  :', dict_run_info_h[run_info_index]['training error'])
        print('Current Validation Error      :', dict_run_info_h[run_info_index]['validation error'])
        run_info_index += 1
    print('---   TRAINING h(x) is COMPLETE   ---')
    # Getting the constant h(x) graph from the above training
    h_Wx_list_num = sess.run(h_Wx_list)
    h_bx_list_num = sess.run(h_bx_list)
    h_x_params_list = {'n_base_states': num_states, 'hidden_var_list': h_x_hidden_vars_list, 'W_list': h_Wx_list_num,
                     'b_list': h_bx_list_num, 'keep_prob': keep_prob, 'activation flag': activation_flag, 'res_net': res_net}
    _, h_of_x_const = initialize_tensorflow_graph(h_x_params_list, xp_feed, add_bias=False, state_inclusive=False,constant_graph=True, h=None)
# ******************************************     Training ends      ******************************************


## Finding psi(x) using deepDMD constrained on h(x)
# Import the dataset as a dictionary with various indices indicating the traces of different curves
dict_train, dict_valid, psi_x_hidden_vars_list = load_pickle_data_deepDMD(data_file)
num_bas_obs = dict_train['Xp'].shape[1]
num_x_observables_total = x_deep_dict_size + num_bas_obs + num_outputs
with tf.device(DEVICE_NAME):
    xf_feed = tf.placeholder(tf.float32, shape=[None, num_states])
    # step_size_feed is already declared previously
    # Tensorflow graph
    psi_Wx_list, psi_bx_list = initialize_Wblist(num_states, psi_x_hidden_vars_list)
    psi_x_params_list = {'n_base_states': num_states, 'hidden_var_list': psi_x_hidden_vars_list, 'W_list': psi_Wx_list,
                       'b_list': psi_bx_list, 'keep_prob': keep_prob, 'activation flag': activation_flag, 'res_net': res_net}
    sess.run(tf.global_variables_initializer())
    # K matrix [everything in the deepDMD code is transposed of what it is supposed to be]
    Kx = weight_variable([num_x_observables_total + 1, num_x_observables_total])
    A_hat_opt, b_hat_opt = get_best_K_DMD2(dict_train['Xp'], dict_train['Xf'])
    sess.run(tf.global_variables_initializer())
    Kx = tf.Variable(sess.run(Kx[0:num_bas_obs, 0:num_bas_obs].assign(A_hat_opt)))
    sess.run(tf.global_variables_initializer())
    Kx = tf.Variable(sess.run(Kx[-1, 0:num_bas_obs].assign(b_hat_opt)))
    last_col = tf.constant(np.zeros(shape=(num_x_observables_total, 1)), dtype=tf.dtypes.float32)
    last_col = tf.concat([last_col, [[1.]]], axis=0)
    Kx = tf.concat([Kx, last_col], axis=1)
    # Getting unconstrained Psi variables
    _, psi_xp = initialize_tensorflow_graph(psi_x_params_list, xp_feed, add_bias=True, state_inclusive=True, constant_graph = False, h = h_of_x_const)
    _, psi_xf = initialize_tensorflow_graph(psi_x_params_list, xf_feed, add_bias=True, state_inclusive=True, constant_graph = False, h = h_of_x_const)
    # Organizing the various variables into feed, psi and K
    dict_feed_psi = {'xp': xp_feed, 'xf': xf_feed, 'step_size': step_size_feed}
    dict_psi = {'xp': psi_xp, 'xf': psi_xf}
    dict_K = {'Kx': Kx}
    dict_model_metrics_psi = objective_func_find_psi(dict_feed_psi, dict_psi, dict_K)
    all_histories_psi = {'train error': [], 'validation error': [], 'train MSE': [], 'valid MSE': []}
    dict_run_info_psi = {}
    feed_dict_train = {dict_feed_psi['xp']: dict_train['Xp'], dict_feed_psi['xf']: dict_train['Xf']}
    feed_dict_valid = {dict_feed_psi['xp']: dict_valid['Xp'], dict_feed_psi['xf']: dict_valid['Xf']}

    # ================================================================================================
    # -------------------------------         Training code             ------------------------------
    # ================================================================================================
    run_info_index = 0
    for dict_train_params_i in ls_dict_training_params_psi:
        display_train_params(dict_train_params_i)
        # -----------------------------
        # Initialization
        # -----------------------------
        N_train_samples = len(dict_train['Xp'])
        runs_per_epoch = int(np.ceil(N_train_samples / dict_train_params_i['batch_size']))
        epoch_i = 0
        training_error = 100
        validation_error = 100
        # -----------------------------
        # Actual training
        # -----------------------------
        while ((epoch_i < dict_train_params_i['max_epochs']) and (
                training_error > dict_train_params_i['train_error_threshold']) and (
                       validation_error > dict_train_params_i['valid_error_threshold'])):
            epoch_i += 1
            # Re initializing the training indices
            all_train_indices = list(range(N_train_samples))
            # Random sort of the training indices
            random.shuffle(all_train_indices)
            for run_i in range(runs_per_epoch):
                try:
                    train_indices = all_train_indices[run_i * dict_train_params_i['batch_size']:(run_i + 1) * dict_train_params_i['batch_size']]
                except:
                    # Last run with the residual data
                    train_indices = all_train_indices[run_i * dict_train_params_i['batch_size']: N_train_samples]
                feed_dict_train_curr = {dict_feed_psi['xp']: dict_train['Xp'][train_indices],
                                        dict_feed_psi['xf']: dict_train['Xf'][train_indices],
                                        dict_feed_h['step_size']: dict_train_params_i['step_size_val']}
                dict_model_metrics_psi['optimizer'].run(feed_dict=feed_dict_train_curr)
            # After training 1 epoch
            training_error = dict_model_metrics_psi['loss_fn'].eval(feed_dict=feed_dict_train)
            validation_error = dict_model_metrics_psi['loss_fn'].eval(feed_dict=feed_dict_valid)
            all_histories_psi['train error'].append(training_error)
            all_histories_psi['validation error'].append(validation_error)
            if np.mod(epoch_i, DISPLAY_SAMPLE_RATE_EPOCH) == 0:
                print('Epoch No: ', epoch_i, ' |   Training error: ', training_error)
                print('Validation error: '.rjust(len('Epoch No: ' + str(epoch_i) + ' |   Validation error: ')),
                      validation_error)
                print('-----------------------------------------------------------------------------------------------')
        dict_run_info_psi[run_info_index] = generate_hyperparam_entry(feed_dict_train, feed_dict_valid,
                                                                    dict_model_metrics_psi, epoch_i, dict_train_params_i,
                                                                    psi_x_params_list)
        print('Current Training Error  :', dict_run_info_psi[run_info_index]['training error'])
        print('Current Validation Error      :', dict_run_info_psi[run_info_index]['validation error'])
        run_info_index += 1
    print('---   TRAINING psi(x) is COMPLETE   ---')




## Saving the data

# Creating a folder for saving the objects of the current run
import platform

if platform.system() == 'Darwin':
    FOLDER_NAME = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/h_OCdeepDMD/System_' + str(SYSTEM_NO) + '/h_OCdeepDMD'
    if not os.path.exists(FOLDER_NAME):
        os.mkdir(FOLDER_NAME)
        run_no = 0
    else:
        # Find the latest/missing run number
        max_run_no = -1
        ls_complete_runs = []
        for items in os.listdir(FOLDER_NAME):
            if items[0:4] == 'RUN_':
                max_run_no = np.max([max_run_no, int(items[4:])])
                ls_complete_runs.append(int(items[4:]))
        ls_missing_runs = list(set(range(max_run_no)) - set(ls_complete_runs))
        if ls_missing_runs == []:
            run_no = max_run_no + 1
        else:
            run_no = ls_missing_runs[0]
    # Append to the run
    FOLDER_NAME = FOLDER_NAME + '/RUN_' + str(run_no)
else:
    try:
        FOLDER_NAME = '_current_run_saved_files/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(RUN_NUMBER)
    except:
        FOLDER_NAME = '_current_run_saved_files/RUN_unknown'
    if os.path.exists(FOLDER_NAME):
        shutil.rmtree(FOLDER_NAME)
os.mkdir(FOLDER_NAME)

dict_dump = {}
dict_dump['hx_W_list_num'] = [sess.run(W_temp) for W_temp in h_Wx_list]
dict_dump['hx_W_list_num'] =[sess.run(b_temp) for b_temp in h_bx_list]
dict_dump['psix_W_list_num'] = [sess.run(W_temp) for W_temp in psi_Wx_list]
dict_dump['psix_b_list_num'] =[sess.run(b_temp) for b_temp in psi_bx_list]
dict_dump['Kx_num'] = sess.run(dict_K['Kx'])
# dict_dump['Wh_num'] = sess.run(dict_K['WhT'])

dict_run_info = {'h': dict_run_info_h, 'psi': dict_run_info_psi}
all_histories = {'h': all_histories_h, 'psi': all_histories_psi}
with open(FOLDER_NAME + '/constrainedNN-Model.pickle', 'wb') as file_obj_swing:
    pickle.dump(dict_dump, file_obj_swing)
with open(FOLDER_NAME + '/run_info.pickle', 'wb') as file_obj_swing:
    pickle.dump(dict_run_info, file_obj_swing)
with open(FOLDER_NAME + '/all_histories.pickle', 'wb') as file_obj_swing:
    pickle.dump(all_histories, file_obj_swing)

saver = tf.compat.v1.train.Saver()

all_tf_var_names =[]

# TODO  Add the Wh matrix to the output and then return it

tf.compat.v1.add_to_collection('h_of_x', h_func['x'])
all_tf_var_names.append('h_of_x')
# for items in dict_feed_h.keys():
#     tf.compat.v1.add_to_collection(items+'_feed_h', dict_feed_h[items])
#     all_tf_var_names.append(items+'_feed_h')
# for items in list(dict_model_metrics_h.keys()):
#     all_tf_var_names.append(items)
#     tf.compat.v1.add_to_collection(items + '_h', dict_model_metrics_h[items])
for items in dict_psi.keys():
    tf.compat.v1.add_to_collection('psi_'+items, dict_psi[items])
    all_tf_var_names.append('psi_'+items)
for items in dict_feed_psi.keys():
    tf.compat.v1.add_to_collection(items+'_feed', dict_feed_psi[items])
    all_tf_var_names.append(items+'_feed')
for items in dict_K.keys():
    tf.compat.v1.add_to_collection(items, dict_K[items])
    all_tf_var_names.append(items)
for items in list(dict_model_metrics_psi.keys()):
    all_tf_var_names.append(items)
    tf.compat.v1.add_to_collection(items, dict_model_metrics_psi[items])

saver_path_curr = saver.save(sess, FOLDER_NAME + '/' + data_suffix + '.ckpt')
with open(FOLDER_NAME + '/all_tf_var_names.pickle', 'wb') as handle:
    pickle.dump(all_tf_var_names,handle)
for items in dict_run_info:
    print('------ ------ -----')
    print('----- Run Info ', items, '----')
    print('------ ------ -----')
    print(pd.DataFrame(dict_run_info[items]))
    print('------ ------ -----')


# Saving the hyperparameters
dict_hp = {'x_obs': x_deep_dict_size, 'x_layers': n_x_nn_layers, 'x_nodes': n_x_nn_nodes, 'regularization factor': regularization_lambda}
dict_hp['r2 h(x) train'] = np.array([dict_run_info['h'][list(dict_run_info['h'].keys())[-1]]['r^2 training accuracy']])
dict_hp['r2 h(x) valid'] = np.array([dict_run_info['h'][list(dict_run_info['h'].keys())[-1]]['r^2 validation accuracy']])
dict_hp['r2 psi(x) train'] = np.array([dict_run_info['h'][list(dict_run_info['psi'].keys())[-1]]['r^2 training accuracy']])
dict_hp['r2 psi(x) valid'] = np.array([dict_run_info['h'][list(dict_run_info['psi'].keys())[-1]]['r^2 validation accuracy']])
with open(FOLDER_NAME + '/dict_hyperparameters.pickle','wb') as handle:
    pickle.dump(dict_hp,handle)







