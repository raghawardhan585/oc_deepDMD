##
# ALWAYS CONSIDERING WITH OUTPUT and STATE

# Required Packages
import pickle  # for data I/O
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from numpy.linalg import pinv # for the least squares approach
import math;
import random;
import tensorflow as tf;
import os
import shutil
import pandas as pd

# Default Parameters
DEVICE_NAME = '/cpu:0'
RUN_NUMBER = 0
max_epochs = 2000
train_error_threshold = 1e-6
valid_error_threshold = 1e-6;
test_error_threshold = 1e-6;

##  Deep Learning Optimization Parameters ##

step_size_val = 0.5  # .025;

batch_size = 400  # 30#900;
eval_size = batch_size;
add_bias = True

use_crelu = 0;
activation_flag = 2;  # sets the activation function type to RELU[0], ELU[1], SELU[2] (initialized a certain way,dropout has to be done differently) , or tanh()

DISPLAY_SAMPLE_RATE_EPOCH = 1000
TRAIN_PERCENT = 50
keep_prob = 1.0;  # keep_prob = 1-dropout probability
res_net = 0;  # Boolean condition on whether to use a resnet connection.

## Neural network parameters

# ---- STATE OBSERVABLE PARAMETERS -------
x_deep_dict_size = 5
n_x_nn_layers = 3  # x_max_layers 3 works well
n_x_nn_nodes = 10  # max width_limit -4 works well

# ---- OUTPUT CONSTRAINED OBSERVABLE PARAMETERS ----
y_deep_dict_size = 3

best_test_error = np.inf

## Learning Parameters
ls_dict_training_params = []
dict_training_params = {'step_size_val': 00.5, 'train_error_threshold': float(1e-6),'valid_error_threshold': float(1e-6), 'max_epochs': 5000, 'batch_size': 1000}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 00.3, 'train_error_threshold': float(1e-6),'valid_error_threshold': float(1e-6), 'max_epochs': 5000, 'batch_size': 1000}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.1, 'train_error_threshold': float(1e-7), 'valid_error_threshold': float(1e-7), 'max_epochs': 100000, 'batch_size': 1000 }
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.08, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': 2000 }
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.05, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': 2000 }
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.01, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': 2000 }
ls_dict_training_params.append(dict_training_params)

sess = tf.InteractiveSession()

## Required Functions

def estimate_K_stability(Kx, print_Kx=False):
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
def load_pickle_data(file_path):
    with open(file_path,'rb') as handle:
        output_vec = pickle.load(handle)
    return output_vec['Xp'], output_vec['Xf'], output_vec['Yp'], output_vec['Yf']

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
def initialize_tensorflow_graph(param_list,Wh1):
    x_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x_hidden_vars_list,
                     'x_observables': x_deep_dict_size, 'y_observables': y_deep_dict_size, 'W_list': Wx_list,
                     'b_list': bx_list,
                     'keep_prob': keep_prob, 'activation flag': activation_flag, 'res_net': res_net}
    # res_net = param_list['res_net'] --- This variable is not used!
    # TODO - remove the above variable if not required at all
    # u is the input of the neural network
    u = tf.placeholder(tf.float32, shape=[None,param_list['n_base_states']]);  # state/input node,# inputs = dim(input) , None indicates batch size can be any size
    z_list = [];
    n_depth = len(param_list['hidden_var_list']);
    # INITIALIZATION
    if param_list['activation flag'] == 1:  # RELU
        z_list.append(tf.nn.dropout(tf.nn.relu(tf.matmul(u, param_list['W_list'][0]) + param_list['b_list'][0]), param_list['keep_prob']))
    if param_list['activation flag']== 2:  # ELU
        z_list.append(tf.nn.dropout(tf.nn.elu(tf.matmul(u, param_list['W_list'][0]) + param_list['b_list'][0]), param_list['keep_prob']))
    if param_list['activation flag'] == 3:  # tanh
        z_list.append(tf.nn.dropout(tf.nn.tanh(tf.matmul(u, param_list['W_list'][0]) + param_list['b_list'][0]), param_list['keep_prob']))
    # PROPAGATION & TERMINATION
    for k in range(0, n_depth):
        prev_layer_output = tf.matmul(z_list[k - 1], param_list['W_list'][k]) + param_list['b_list'][k]
        if param_list['activation flag'] == 1: # RELU
            z_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output), param_list['keep_prob']));
        if param_list['activation flag'] == 2: # ELU
            z_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output), param_list['keep_prob']));
        if param_list['activation flag'] == 3: # tanh
            z_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output), param_list['keep_prob']));
    y = tf.concat([u, z_list[-1][0:param_list['x_observables']]], axis=1)
    y = tf.concat([y, tf.ones(shape=(tf.shape(y)[0], 1))], axis=1)
    y = tf.concat([y, z_list[-1][0:param_list['x_observables']]], axis=1)
    y = tf.concat([y, z_list[-1][param_list['x_observables']:]], axis=1)
    result = sess.run(tf.global_variables_initializer())
    return z_list, y, u

def get_variable_value(variable_name, prev_variable_value, reqd_data_type, lower_bound=0):
    # Purpose: This function is mainly to
    not_valid = True
    variable_output = prev_variable_value
    while (not_valid):
        print('Current value of ', variable_name, ' = ', prev_variable_value)
        variable_input = input('Enter new ' + variable_name + ' value [-1 or ENTER to retain previous entry]: ')
        # First check for -1
        if variable_input in ['-1', '']:
            not_valid = False
        else:
            # Second check for correct data type
            try:
                variable_input = reqd_data_type(variable_input)
                # Third check for the correct bound
                if not (variable_input > lower_bound):
                    print('Error! Value is out of bounds. Please enter a value greater than ', lower_bound)
                    not_valid = True
                else:
                    variable_output = variable_input
                    not_valid = False
            except:
                print('Error! Please enter a ', reqd_data_type, ' value, -1 or ENTER')
                not_valid = True
    return variable_output
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
def generate_hyperparam_entry(feed_dict_train, feed_dict_valid, error_func, r2_accuracy, n_epochs_run, dict_run_params):
    training_error = error_func.eval(feed_dict=feed_dict_train)
    validation_error = error_func.eval(feed_dict=feed_dict_valid)
    training_accuracy = r2_accuracy.eval(feed_dict=feed_dict_train)
    validation_accuracy = r2_accuracy.eval(feed_dict=feed_dict_valid)
    dict_hp = {}
    dict_hp['x_hidden_variable_list'] = x_hidden_vars_list
    # dict_hp['u_hidden_variable_list'] = u_hidden_vars_list
    # dict_hp['xu_hidden_variable_list'] = xu_hidden_vars_list
    dict_hp['activation flag'] = activation_flag
    dict_hp['activation function'] = None
    if activation_flag == 1:
        dict_hp['activation function'] = 'relu'
    elif activation_flag == 2:
        dict_hp['activation function'] = 'elu'
    elif activation_flag == 3:
        dict_hp['activation function'] = 'tanh'
    dict_hp['no of epochs'] = n_epochs_run
    dict_hp['batch size'] = dict_run_params['batch_size']
    dict_hp['step size'] = dict_run_params['step_size_val']
    dict_hp['training error'] = training_error
    dict_hp['validation error'] = validation_error
    # dict_hp['test error'] = test_error
    dict_hp['r^2 training accuracy'] = training_accuracy
    dict_hp['r^2 validation accuracy'] = validation_accuracy
    # dict_hp['r^2 test accuracy'] = test_accuracy
    return dict_hp

def state_dynamics_objective(dict_feed,dict_psi,dict_K):
    psiXf_predicted = tf.matmul(dict_psi['xpT'], dict_K['KxT'])
    psiXf_prediction_error = dict_psi['xfT'] - psiXf_predicted
    SST_x = tf.math.reduce_sum(tf.math.square(dict_psi['xfT']), axis=0)
    SSE_x = tf.math.reduce_sum(tf.math.square(psiXf_prediction_error), axis=0)
    psiXf_accuracy_percent = (1 - tf.math.reduce_max(tf.divide(SSE_x, SST_x))) * 100
    tf_koopman_loss = tf.math.reduce_mean(tf.math.square(psiXf_prediction_error))
    optimizer = tf.train.AdagradOptimizer(dict_feed['step_size']).minimize(tf_koopman_loss)
    return tf_koopman_loss, optimizer, psiXf_accuracy_percent

def static_train_net(dict_train, dict_valid, dict_feed, ls_dict_training_params, deep_koopman_loss, optimizer, model_accuracy_percent, all_histories = {'train error': [],'validation error': []}, dict_run_info = {}):
    feed_dict_train = {dict_feed['xpT']: dict_train['Xp'],dict_feed['xfT']: dict_train['Xf']}
    feed_dict_valid = {dict_feed['xpT']: dict_valid['Xp'],dict_valid['xfT']: dict_valid['Xf']}
    # --------
    try :
        run_info_index = list(dict_run_info.keys())[-1]
    except:
        run_info_index = 0
    for dict_train_params_i in ls_dict_training_params:
        display_train_params(dict_train_params_i)
        all_histories, n_epochs_run = train_net_v2(dict_train,feed_dict_train, feed_dict_valid, dict_feed, deep_koopman_loss, optimizer,dict_train_params_i,all_histories)
        dict_run_info[run_info_index] = generate_hyperparam_entry(feed_dict_train, feed_dict_valid,deep_koopman_loss, model_accuracy_percent,n_epochs_run, dict_train_params_i)
        print('Current Training Error  :', dict_run_info[run_info_index]['training error'])
        print('Current Validation Error      :', dict_run_info[run_info_index]['validation error'])
        estimate_K_stability(Kx)
        run_info_index += 1
    return all_histories, dict_run_info

def train_net_v2(dict_train, feed_dict_train, feed_dict_valid, dict_feed, loss_func,optimizer, dict_run_params, all_histories):
    # -----------------------------
    # Initialization
    # -----------------------------
    N_train_samples = len(dict_train['Xp'])
    runs_per_epoch = int(np.ceil(N_train_samples / batch_size))
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
                train_indices = all_train_indices[run_i * batch_size:(run_i + 1) * batch_size]
            else:
                # Last run with the residual data
                train_indices = all_train_indices[run_i * batch_size: N_train_samples]
            feed_dict_train_curr = {dict_feed['xpT']: dict_train['Xp'][train_indices], dict_feed['xfT']: dict_train['Xf'][train_indices], dict_feed['step_size']: dict_run_params['step_size_val']}
            optimizer.run(feed_dict=feed_dict_train_curr)
        # After training 1 epoch
        training_error = loss_func.eval(feed_dict=feed_dict_train)
        validation_error = loss_func.eval(feed_dict=feed_dict_valid)
        all_histories['train error'].append(training_error)
        all_histories['validation error'].append(validation_error)
        if np.mod(epoch_i, DISPLAY_SAMPLE_RATE_EPOCH) == 0:
            print('Epoch No: ', epoch_i, ' |   Training error: ', training_error)
            print('Validation error: '.rjust(len('Epoch No: ' + str(epoch_i) + ' |   Validation error: ')),validation_error)
            # estimate_K_stability(Kx)
            print('---------------------------------------------------------------------------------------------------')
    return all_histories, epoch_i



# def least_squares_soln_by_inverse(X,Y):
#     # Model: Y = XA
#     # A is being returned
#     return pinv(X.T @X)@ X.T @ Y
#
# def least_squares_soln_by_svd(Xtrain,Ytrain,Xvalid,Yvalid):
#     # Model: Y = XA
#     U,S,VT = np.linalg.svd(Xtrain)
#     V = VT.T
#     UT = U.T
#     nPC = list(range(len(S)))
#     Ahat = np.zeros(shape=(Xtrain.shape[1],Ytrain.shape[1]))
#     Ytrain_err = np.empty(shape=(0,1))
#     Yvalid_err = np.empty(shape=(0,1))
#     for i in nPC:
#         Ahat = Ahat + 1/S[i]*V[:,i:i+1] @ UT[i:i+1,:] @ Ytrain
#         Ytrain_err = np.append(Ytrain_err,[np.sum(np.abs(Ytrain - Xtrain @ Ahat))])
#         Yvalid_err = np.append(Yvalid_err, [np.sum(np.abs(Yvalid - Xvalid @ Ahat))])
#     Ytot_err = Ytrain_err +Yvalid_err
#     nPC_opt = nPC[Ytot_err == np.min(Ytot_err)]
#     Aopt = V[:,0:nPC_opt] @ pinv(np.diag(S[0:nPC_opt])) @ UT[0:nPC_opt,:] @ Ytrain
#     return Aopt



# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================


## Main Block

data_directory = 'koopman_data/'
data_suffix = 'System_4_ocDeepDMDdata.pickle'

## CMD Line Argument (Override) Inputs:
# # TODO - Rearrange this section
# import sys
# if len(sys.argv)>1:
#     DEVICE_NAME = sys.argv[1]
#     if DEVICE_NAME not in ['/cpu:0','/gpu:0','/gpu:1','/gpu:2','/gpu:3']:
#         DEVICE_NAME = '/cpu:0'
# if len(sys.argv)>2:
#     SYSTEM_NO = sys.argv[2]
#     data_suffix = 'System_'+ str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
# if len(sys.argv) > 3:
#     with_control = np.int(sys.argv[3])
# if len(sys.argv)>4:
#     with_output = np.int(sys.argv[4])
# if len(sys.argv)>5:
#     mix_state_and_control = np.int(sys.argv[5])
# if len(sys.argv)>6:
#     RUN_NUMBER = np.int(sys.argv[6])
# if len(sys.argv)>7:
#     x_deep_dict_size = np.int(sys.argv[7])
# if len(sys.argv)>8:
#     x_max_nn_layers = np.int(sys.argv[8])
# if len(sys.argv)>9:
#     x_max_nn_nodes_limit = np.int(sys.argv[9])
#     x_min_nn_nodes_limit = x_max_nn_nodes_limit

data_file = data_directory + data_suffix
Xp, Xf, Yp, Yf = load_pickle_data(data_file)
num_bas_obs = len(Xp[0])
num_all_samples = len(Xp)
num_outputs = len(Yf[0])

## Train/Test Split for Benchmarking Forecasting Later

num_trains = np.int(num_all_samples * TRAIN_PERCENT / 100)
train_indices = np.arange(0, num_trains, 1)
valid_indices = np.arange(num_trains,num_all_samples,1)
dict_train = {}
dict_valid = {}
dict_train['Xp'] = Xp[train_indices]
dict_valid['Xp'] = Xp[valid_indices]
dict_train['Yp'] = Yp[train_indices]
dict_valid['Yp'] = Yp[valid_indices]
dict_train['Xf'] = Xf[train_indices]
dict_valid['Xf'] = Xf[valid_indices]
dict_train['Yf'] = Yf[train_indices]
dict_valid['Yf'] = Yf[valid_indices]
# Hidden layer list creation for state dynamics
x_hidden_vars_list = np.asarray([n_x_nn_nodes] * n_x_nn_layers)
x_hidden_vars_list[-1] = x_deep_dict_size  + y_deep_dict_size # The last hidden layer being declared as the output


# Display info
print("[INFO] Number of total samples: " + repr(num_all_samples))
print("[INFO] Observable dimension of a sample: " + repr(num_bas_obs))
print("[INFO] Xp.shape (E-DMD): " + repr(Xp.shape));
print("[INFO] Yf.shape (E-DMD): " + repr(Xf.shape));
print("Number of training snapshots: " + repr(len(train_indices)));
print("Number of validation snapshots: " + repr(len(valid_indices)));
print("[INFO] STATE - hidden_vars_list: " + repr(x_hidden_vars_list))


# ============================
# LEARNING THE STATE DYNAMICS
# ============================
with tf.device(DEVICE_NAME):
    dict_feed = {}
    dict_psi = {}
    dict_K ={}
    # Initialize the K and Wh matrices
    # Kx definition w/ bias
    KxT_11 = weight_variable([x_deep_dict_size + num_bas_obs + 1, x_deep_dict_size + num_bas_obs])
    last_col = tf.constant(np.zeros(shape=(x_deep_dict_size + num_bas_obs, 1)), dtype=tf.dtypes.float32)
    last_col = tf.concat([last_col, [[1.]]], axis=0)
    KxT_11 = tf.concat([KxT_11, last_col], axis=1)
    KxT_12 = tf.constant(np.zeros(shape=(y_deep_dict_size + num_outputs, x_deep_dict_size + num_bas_obs+1)), dtype=tf.dtypes.float32)
    KxT_1 = tf.concat([KxT_11,KxT_12],axis=0)
    KxT_2 = weight_variable([x_deep_dict_size + num_bas_obs + y_deep_dict_size + num_outputs + 1, y_deep_dict_size + num_outputs])
    KxT = tf.concat([KxT_1, KxT_2], axis=1)
    # Wh definition
    Wh1 = weight_variable([x_deep_dict_size + num_bas_obs + 1, num_outputs])
    Wh2 = tf.concat([tf.constant(np.identity(num_outputs), dtype=tf.dtypes.float32),tf.constant(np.zeros(shape=(y_deep_dict_size,num_outputs)), dtype=tf.dtypes.float32)],axis=0)
    Wh = tf.concat([Wh1,Wh2],axis=0)
    # Initialize the hidden layers
    Wx_list, bx_list = initialize_Wblist(num_bas_obs, x_hidden_vars_list)
    x_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x_hidden_vars_list, 'x_observables':x_deep_dict_size, 'y_observables':y_deep_dict_size, 'W_list': Wx_list, 'b_list': bx_list,
                     'keep_prob': keep_prob, 'activation flag': activation_flag, 'res_net': res_net, 'include state': True, 'add bias': add_bias}
    psixpz_list, psixp, xp_feed = initialize_tensorflow_graph(x_params_list,Wh1)
    psixfz_list, psixf, xf_feed = initialize_tensorflow_graph(x_params_list)

    print('Kx initiation done!')
    dict_feed ['xpT'] = xp_feed
    dict_feed ['xfT'] = xf_feed
    dict_psi ['xpT'] = psixp
    dict_psi['xfT'] = psixf
    dict_K['KxT'] = KxT
    dict_feed['step_size'] = tf.placeholder(tf.float32, shape=[])
    sess.run(tf.global_variables_initializer())
    deep_koopman_loss, optimizer, psiXf_accuracy_percent = state_dynamics_objective(dict_feed, dict_psi, dict_K)
    print('Training begins now!')
    all_histories, dict_run_info = static_train_net(dict_train, dict_valid, dict_feed, ls_dict_training_params, deep_koopman_loss, optimizer,psiXf_accuracy_percent)
    print('---   TRAINING COMPLETE   ---')
unstable = estimate_K_stability(Kx)
if unstable:
    exit()
training_error_history_nocovar = all_histories['train error'];
validation_error_history_nocovar = all_histories['validation error'];
feed_dict_train = {xp_feed: dict_train['Xp'], xf_feed: dict_train['Xf']}
feed_dict_valid = {xp_feed: dict_valid['Xp'], xf_feed: dict_valid['Xf']}
train_error = deep_koopman_loss.eval(feed_dict=feed_dict_train)
valid_error = deep_koopman_loss.eval(feed_dict=feed_dict_valid)


# # Display the results of training the dynamics
#
# # ============================
# # LEARNING THE OUTPUT DYNAMICS [Nothing to train - Maybe tensorflow is not required]
# # ============================
# # Wh identified by least squares
# psiXp_num_train = psixp.eval(feed_dict = {dict_feed['xpT']:dict_train['Xp']})
# psiXp_num_valid = psixp.eval(feed_dict = {dict_feed['xpT']:dict_valid['Xp']})
# psiXf_num_train = psixf.eval(feed_dict = {dict_feed['xfT']:dict_train['Xf']})
# psiXf_num_valid = psixf.eval(feed_dict = {dict_feed['xfT']:dict_valid['Xf']})
# Yf_train = dict_train['Yf']
# Yf_valid = dict_valid['Yf']
# Wh1 = least_squares_soln_by_inverse(psiXf_num_train,Yf_train)
# Wh2 = least_squares_soln_by_svd(psiXf_num_train,Yf_train,psiXf_num_valid,Yf_valid)
# Wh1_err = np.mean(np.append(np.square(Yf_train - psiXf_num_train @ Wh1),np.square(Yf_valid - psiXf_num_valid @ Wh1)))
# Wh2_err = np.mean(np.append(np.square(Yf_train - psiXf_num_train @ Wh2),np.square(Yf_valid - psiXf_num_valid @ Wh2)))
# Wh = Wh1
# Wh_err = Wh1_err
# if Wh1_err > Wh2_err:
#     Wh = Wh2
#     Wh_err = Wh2_err
# print('Error in output equation = ', Wh_err)
#
# # ============================
# # LEARNING THE ERROR DYNAMICS
# # ============================
# # Getting the error data
# dict_train['ep'] = dict_train['Yp'] - psiXp_num_train @ Wh
# dict_train['ef'] = dict_train['Yf'] - psiXf_num_train @ Wh
# dict_valid['ep'] = dict_valid['Yp'] - psiXp_num_valid @ Wh
# dict_valid['ef'] = dict_valid['Yf'] - psiXf_num_valid @ Wh
# # Hidden layer list
# e_hidden_vars_list = np.asarray([n_e_nn_nodes] * n_e_nn_layers)
# e_hidden_vars_list[-1] = e_deep_dict_size # The last hidden layer being declared as the output
# print("[INFO] OUTPUT ERROR - hidden_vars_list: " + repr(e_hidden_vars_list))
# e_num_bas_obs = len(dict_train['ep'][0])
# with tf.device(DEVICE_NAME):
#     We_list, be_list = initialize_Wblist(e_num_bas_obs, e_hidden_vars_list)
#     e_params_list = {'no of base observables': e_num_bas_obs, 'hidden_var_list': e_hidden_vars_list, 'W_list': We_list, 'b_list': be_list,
#                      'keep_prob': keep_prob, 'activation flag': activation_flag, 'res_net': res_net, 'include state': True, 'add bias': False}
#     psiepz_list, psiep, ep_feed = initialize_tensorflow_graph(e_params_list)
#     psiefz_list, psief, ef_feed = initialize_tensorflow_graph(e_params_list)
#     # Kx definition
#     Kx2 = weight_variable([x_deep_dict_size + num_bas_obs, x_deep_dict_size + num_bas_obs])
#
#     print('Kx initiation done!')
#     dict_feed ['xpT'] = xp_feed;
#     dict_feed ['xfT'] = xf_feed;
#     dict_psi ['xpT'] = psixp;
#     dict_psi['xfT'] = psixf;
#     dict_K['KxT'] = Kx;
#     dict_feed['step_size'] = tf.placeholder(tf.float32, shape=[])
#     sess.run(tf.global_variables_initializer())
#     deep_koopman_loss, optimizer, psiXf_accuracy_percent = state_dynamics_objective(dict_feed, dict_psi, dict_K)
#     print('Training begins now!')
#     all_histories, dict_run_info = static_train_net(dict_train, dict_valid, dict_feed, ls_dict_training_params, deep_koopman_loss, optimizer,psiXf_accuracy_percent)
#     print('---   TRAINING COMPLETE   ---')
#





# Saving the results of the run




