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
# max_epochs = 2000
# train_error_threshold = 1e-6
# valid_error_threshold = 1e-6;
# test_error_threshold = 1e-6;

#  Deep Learning Optimization Parameters ##

activation_flag = 2;  # sets the activation function type to RELU[0], ELU[1], SELU[2] (initialized a certain way,dropout has to be done differently) , or tanh()

DISPLAY_SAMPLE_RATE_EPOCH = 1000
TRAIN_PERCENT = 50
keep_prob = 1.0;  # keep_prob = 1-dropout probability
res_net = 0;  # Boolean condition on whether to use a resnet connection.

# Neural network parameters

# ---- STATE OBSERVABLE PARAMETERS -------
x_deep_dict_size = 2
n_x_nn_layers = 3  # x_max_layers 3 works well
n_x_nn_nodes = 10  # max width_limit -4 works well

# ---- OUTPUT CONSTRAINED OBSERVABLE PARAMETERS ----
y_deep_dict_size = 1
n_y_nn_layers = 3
n_y_nn_nodes = 5

xy_deep_dict_size = 2
n_xy_nn_layers = 3
n_xy_nn_nodes = 10

best_test_error = np.inf

# Learning Parameters
ls_dict_training_params = []
dict_training_params = {'step_size_val': 00.5, 'train_error_threshold': float(1e-6),'valid_error_threshold': float(1e-6), 'max_epochs': 5000, 'batch_size': 100}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 00.3, 'train_error_threshold': float(1e-6),'valid_error_threshold': float(1e-6), 'max_epochs': 5000, 'batch_size': 100}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.1, 'train_error_threshold': float(1e-7), 'valid_error_threshold': float(1e-7), 'max_epochs': 10000, 'batch_size': 100 }
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.08, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': 200 }
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.05, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': 200 }
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.01, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': 200 }
ls_dict_training_params.append(dict_training_params)

ls_dict_training_params1 = ls_dict_training_params

# ls_dict_training_params = []
# dict_training_params = {'step_size_val': 00.5, 'train_error_threshold': float(1e-6),'valid_error_threshold': float(1e-6), 'max_epochs': 10000, 'batch_size': 100}
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 00.3, 'train_error_threshold': float(1e-6),'valid_error_threshold': float(1e-6), 'max_epochs': 20000, 'batch_size': 100}
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.1, 'train_error_threshold': float(1e-7), 'valid_error_threshold': float(1e-7), 'max_epochs': 30000, 'batch_size': 100 }
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.08, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': 200 }
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.05, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': 200 }
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.01, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 5000, 'batch_size': 200 }
# ls_dict_training_params.append(dict_training_params)


ls_dict_training_params2 = ls_dict_training_params
ls_dict_training_params3 = []

sess = tf.InteractiveSession()

# Required Functions

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
def initialize_tensorflow_graph(param_list,u, state_inclusive=False,add_bias=False):
    # res_net = param_list['res_net'] --- This variable is not used!
    # TODO - remove the above variable if not required at all
    # u is the input of the neural network
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
    for k in range(1, n_depth):
        prev_layer_output = tf.matmul(z_list[k - 1], param_list['W_list'][k]) + param_list['b_list'][k]
        if param_list['activation flag'] == 1: # RELU
            z_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output), param_list['keep_prob']));
        if param_list['activation flag'] == 2: # ELU
            z_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output), param_list['keep_prob']));
        if param_list['activation flag'] == 3: # tanh
            z_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output), param_list['keep_prob']));
    if state_inclusive:
        y = tf.concat([u, z_list[-1]], axis=1)
    else:
        y = z_list[-1]
    if add_bias:
        y = tf.concat([y, tf.ones(shape=(tf.shape(y)[0], 1))], axis=1)
    result = sess.run(tf.global_variables_initializer())
    return z_list, y

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

def generate_hyperparam_entry(feed_dict_train, feed_dict_valid, dict_model_metrics, n_epochs_run, dict_run_params,x_hidden_vars_list):
    training_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_train)
    validation_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_valid)
    training_accuracy = dict_model_metrics['accuracy'].eval(feed_dict=feed_dict_train)
    validation_accuracy = dict_model_metrics['accuracy'].eval(feed_dict=feed_dict_valid)
    training_MSE = dict_model_metrics['MSE'].eval(feed_dict=feed_dict_train)
    validation_MSE = dict_model_metrics['MSE'].eval(feed_dict=feed_dict_valid)
    dict_hp = {}
    dict_hp['x_hidden_variable_list'] = x_hidden_vars_list
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
    dict_hp['r^2 training accuracy'] = training_accuracy
    dict_hp['r^2 validation accuracy'] = validation_accuracy
    dict_hp['MSE training'] = training_MSE
    dict_hp['MSE validation'] = validation_MSE
    return dict_hp

def objective_func_output(dict_feed,dict_psi,dict_K):
    dict_model_perf_metrics ={}
    Yf_prediction_error = dict_feed['yfT'] - tf.matmul(dict_psi['xfT'], dict_K['WhT'])
    try:
        Yp_prediction_error = dict_feed['ypT'] - tf.matmul(dict_psi['xpT'], dict_K['WhT'])
        Y_prediction_error = tf.concat([Yp_prediction_error,Yf_prediction_error],axis=1)
    except:
        Y_prediction_error = Yf_prediction_error

    dict_model_perf_metrics['loss_fn'] = tf.math.reduce_mean(tf.math.square(Y_prediction_error))
    dict_model_perf_metrics['optimizer'] = tf.train.AdagradOptimizer(dict_feed['step_size']).minimize(dict_model_perf_metrics ['loss_fn'])
    # Mean Squared Error
    dict_model_perf_metrics ['MSE'] = tf.math.reduce_mean(tf.math.square(Y_prediction_error))
    # Accuracy computation
    SST = tf.math.reduce_sum(tf.math.square(dict_feed['yfT']), axis=0)
    SSE = tf.math.reduce_sum(tf.math.square(Yf_prediction_error), axis=0)
    dict_model_perf_metrics ['accuracy'] = (1 - tf.math.reduce_max(tf.divide(SSE, SST))) * 100
    sess.run(tf.global_variables_initializer())
    return dict_model_perf_metrics

def objective_func_state(dict_feed,dict_psi,dict_K):
    dict_model_perf_metrics ={}
    psiXf_predicted = tf.matmul(dict_psi['xpT'], dict_K['KxT'])
    psiXf_prediction_error = dict_psi['xfT'] - psiXf_predicted
    dict_model_perf_metrics['loss_fn'] = tf.math.reduce_mean(tf.math.square(psiXf_prediction_error))
    dict_model_perf_metrics['optimizer'] = tf.train.AdagradOptimizer(dict_feed['step_size']).minimize(dict_model_perf_metrics ['loss_fn'])
    # Mean Squared Error
    dict_model_perf_metrics['MSE'] = tf.math.reduce_mean(tf.math.square(psiXf_prediction_error))
    # Accuracy computation
    SST = tf.math.reduce_sum(tf.math.square(dict_psi['xfT']), axis=0)
    SSE = tf.math.reduce_sum(tf.math.square(psiXf_prediction_error), axis=0)
    dict_model_perf_metrics['accuracy'] = (1 - tf.math.reduce_max(tf.divide(SSE, SST))) * 100
    sess.run(tf.global_variables_initializer())
    return dict_model_perf_metrics


def get_fed_dict(dict_train,dict_valid,dict_feed):
    # The function doesn't feed step size
    feed_dict_train = {}
    feed_dict_valid = {}
    for items in dict_feed.keys():
        if items == 'xpT':
            feed_dict_train[dict_feed['xpT']] = dict_train['Xp']
            feed_dict_valid[dict_feed['xpT']] = dict_valid['Xp']
        elif items == 'xfT':
            feed_dict_train[dict_feed['xfT']] = dict_train['Xf']
            feed_dict_valid[dict_feed['xfT']] = dict_valid['Xf']
        elif items == 'ypT':
            feed_dict_train[dict_feed['ypT']] = dict_train['Yp']
            feed_dict_valid[dict_feed['ypT']] = dict_valid['Yp']
        elif items == 'yfT':
            feed_dict_train[dict_feed['yfT']] = dict_train['Yf']
            feed_dict_valid[dict_feed['yfT']] = dict_valid['Yf']
        elif items == 'psix1pT':
            feed_dict_train[dict_feed['psix1pT']] = dict_train['psiX1p']
            feed_dict_valid[dict_feed['psix1pT']] = dict_valid['psiX1p']
        elif items == 'psix1fT':
            feed_dict_train[dict_feed['psix1fT']] = dict_train['psiX1f']
            feed_dict_valid[dict_feed['psix1fT']] = dict_valid['psiX1f']
        elif items == 'psix2pT':
            feed_dict_train[dict_feed['psix2pT']] = dict_train['psiX2p']
            feed_dict_valid[dict_feed['psix2pT']] = dict_valid['psiX2p']
        elif items == 'psix2fT':
            feed_dict_train[dict_feed['psix2fT']] = dict_train['psiX2f']
            feed_dict_valid[dict_feed['psix2fT']] = dict_valid['psiX2f']
    return feed_dict_train,feed_dict_valid

def get_fed_dict_train_only(dict_train,dict_feed,train_indices):
    # The function doesn't feed step size
    feed_dict_train = {}
    for items in dict_feed.keys():
        if items == 'xpT':
            feed_dict_train[dict_feed['xpT']] = dict_train['Xp'][train_indices]
        elif items == 'xfT':
            feed_dict_train[dict_feed['xfT']] = dict_train['Xf'][train_indices]
        elif items == 'ypT':
            feed_dict_train[dict_feed['ypT']] = dict_train['Yp'][train_indices]
        elif items == 'yfT':
            feed_dict_train[dict_feed['yfT']] = dict_train['Yf'][train_indices]
        elif items == 'psix1pT':
            feed_dict_train[dict_feed['psix1pT']] = dict_train['psiX1p'][train_indices]
        elif items == 'psix1fT':
            feed_dict_train[dict_feed['psix1fT']] = dict_train['psiX1f'][train_indices]
        elif items == 'psix2pT':
            feed_dict_train[dict_feed['psix2pT']] = dict_train['psiX2p'][train_indices]
        elif items == 'psix2fT':
            feed_dict_train[dict_feed['psix2fT']] = dict_train['psiX2f'][train_indices]
    return feed_dict_train

def static_train_net(dict_train, dict_valid, dict_feed, ls_dict_training_params, dict_model_metrics, all_histories, dict_run_info,x_params_list={}):
    feed_dict_train, feed_dict_valid = get_fed_dict(dict_train,dict_valid,dict_feed)
    # --------
    try :
        run_info_index = list(dict_run_info.keys())[-1]
    except:
        run_info_index = 0
    for dict_train_params_i in ls_dict_training_params:
        display_train_params(dict_train_params_i)
        all_histories, n_epochs_run = train_net_v2(dict_train,feed_dict_train, feed_dict_valid, dict_feed, dict_model_metrics, dict_train_params_i, all_histories)
        dict_run_info[run_info_index] = generate_hyperparam_entry(feed_dict_train, feed_dict_valid,dict_model_metrics,n_epochs_run, dict_train_params_i,x_params_list['hidden_var_list'])
        print('Current Training Error  :', dict_run_info[run_info_index]['training error'])
        print('Current Validation Error      :', dict_run_info[run_info_index]['validation error'])
        # estimate_K_stability(KxT)
        run_info_index += 1
    return all_histories, dict_run_info

def train_net_v2(dict_train, feed_dict_train, feed_dict_valid, dict_feed, dict_model_metrics, dict_run_params, all_histories):

    # -----------------------------
    # Initialization
    # -----------------------------
    N_train_samples = num_trains # Taken from global reference
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
            feed_dict_train_curr = get_fed_dict_train_only(dict_train,dict_feed,train_indices)
            feed_dict_train_curr[dict_feed['step_size']] =  dict_run_params['step_size_val']
            dict_model_metrics['optimizer'].run(feed_dict=feed_dict_train_curr)
        # After training 1 epoch
        training_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_train)
        validation_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_valid)
        all_histories['train error'].append(training_error)
        all_histories['validation error'].append(validation_error)
        all_histories['train MSE'].append(dict_model_metrics['MSE'].eval(feed_dict=feed_dict_train))
        all_histories['valid MSE'].append(dict_model_metrics['MSE'].eval(feed_dict=feed_dict_valid))
        if np.mod(epoch_i, DISPLAY_SAMPLE_RATE_EPOCH) == 0:
            print('Epoch No: ', epoch_i, ' |   Training error: ', training_error)
            print('Validation error: '.rjust(len('Epoch No: ' + str(epoch_i) + ' |   Validation error: ')),validation_error)
            # estimate_K_stability(Kx)
            print('---------------------------------------------------------------------------------------------------')
    return all_histories, epoch_i


# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================
# ==============================================================================================================================


# Main Block
# data_directory = os.path.normpath(os.getcwd() + os.sep + os.pardir) +'/koopman_data/'
data_directory = 'koopman_data/'
data_suffix = 'System_5_ocDeepDMDdata.pickle'

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
if len(sys.argv) > 7:
    y_deep_dict_size = np.int(sys.argv[7])
if len(sys.argv)>8:
    n_y_nn_layers = np.int(sys.argv[8])
if len(sys.argv)>9:
    n_y_nn_nodes = np.int(sys.argv[9])
if len(sys.argv) > 10:
    xy_deep_dict_size = np.int(sys.argv[10])
if len(sys.argv)>11:
    n_xy_nn_layers = np.int(sys.argv[11])
if len(sys.argv)>12:
    n_xy_nn_nodes = np.int(sys.argv[12])

data_file = data_directory + data_suffix
Xp, Xf, Yp, Yf = load_pickle_data(data_file)
num_bas_obs = len(Xp[0])
num_all_samples = len(Xp)
num_outputs = len(Yf[0])

# Train/Test Split for Benchmarking Forecasting Later

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



# Display info
print("[INFO] Number of total samples: " + repr(num_all_samples))
print("[INFO] Observable dimension of a sample: " + repr(num_bas_obs))
print("[INFO] Xp.shape (E-DMD): " + repr(Xp.shape))
print("[INFO] Yf.shape (E-DMD): " + repr(Xf.shape))
print("Number of training snapshots: " + repr(len(train_indices)))
print("Number of validation snapshots: " + repr(len(valid_indices)))
# print("[INFO] STATE - hidden_vars_list: " + repr(x_hidden_vars_list))

##
# ============================
# LEARNING THE STATE DYNAMICS
# ============================
with tf.device(DEVICE_NAME):
    # ==============
    # RUN 1
    # ==============
    # Hidden layer creation
    x1_hidden_vars_list = np.asarray([n_x_nn_nodes] * n_x_nn_layers)
    x1_hidden_vars_list[-1] = x_deep_dict_size # The last hidden layer being declared as the output
    Wx1_list, bx1_list = initialize_Wblist(num_bas_obs, x1_hidden_vars_list)
    x1_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x1_hidden_vars_list,'W_list': Wx1_list,
                      'b_list': bx1_list,'keep_prob': keep_prob,'activation flag': activation_flag,'res_net': res_net}
    # Data Required
    # We already have xp and xf given
    dict_train1 = {'Xp': Xp[train_indices], 'Xf': Xf[train_indices]}
    dict_valid1 = {'Xp': Xp[valid_indices], 'Xf': Xf[valid_indices]}
    # K Variables  -    Kx definition w/ bias
    KxT_11 = weight_variable([x_deep_dict_size + num_bas_obs + 1, x_deep_dict_size + num_bas_obs])
    last_col = tf.constant(np.zeros(shape=(x_deep_dict_size + num_bas_obs, 1)), dtype=tf.dtypes.float32)
    last_col = tf.concat([last_col, [[1.]]], axis=0)
    KxT_11 = tf.concat([KxT_11, last_col], axis=1)
    # Feed Variable definitions
    xp_feed = tf.placeholder(tf.float32, shape=[None, num_bas_obs])
    xf_feed = tf.placeholder(tf.float32, shape=[None, num_bas_obs])
    step_size_feed = tf.placeholder(tf.float32, shape=[])
    # Psi variables
    psix1pz_list, psix1p = initialize_tensorflow_graph(x1_params_list, xp_feed, state_inclusive=True, add_bias=True)
    psix1fz_list, psix1f = initialize_tensorflow_graph(x1_params_list, xf_feed, state_inclusive=True, add_bias=True)
    # Objective Function Variables
    dict_feed1 = { 'xpT': xp_feed, 'xfT': xf_feed, 'step_size': step_size_feed}
    dict_psi1 = {'xpT': psix1p, 'xfT': psix1f}
    dict_K1 ={'KxT':KxT_11}
    # First optimization
    print('---------    TRAINING BEGINS   ---------')
    dict_model1_metrics = objective_func_state({'step_size': step_size_feed}, dict_psi1, dict_K1)
    all_histories1 = {'train error': [], 'validation error': [], 'train MSE': [], 'valid MSE': []}
    dict_run_info1 = {}
    all_histories1, dict_run_info1 = static_train_net(dict_train1, dict_valid1, dict_feed1, ls_dict_training_params1,dict_model1_metrics,all_histories1,dict_run_info1,x_params_list =x1_params_list)
    print('---   STATE TRAINING COMPLETE   ---')

    print(pd.DataFrame(dict_run_info1))
    # ==============
    # RUN 2
    # ==============
    # Hidden layer creation
    x2_hidden_vars_list = np.asarray([n_y_nn_nodes] * n_y_nn_layers)
    x2_hidden_vars_list[-1] = y_deep_dict_size  # The last hidden layer being declared as the output
    Wx2_list, bx2_list = initialize_Wblist(num_bas_obs, x2_hidden_vars_list)
    x2_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x2_hidden_vars_list, 'W_list': Wx2_list,
                      'b_list': bx2_list, 'keep_prob': keep_prob, 'activation flag': activation_flag,'res_net': res_net}
    # Data Required
    psix1p_num = psix1p.eval(feed_dict={xp_feed: Xp})
    psix1f_num = psix1f.eval(feed_dict={xf_feed: Xf})
    dict_train2 = {'Xp': Xp[train_indices],  'psiX1p': psix1p_num[train_indices],  'Yp': Yp[train_indices], 'Xf': Xf[train_indices],  'psiX1f': psix1f_num[train_indices],  'Yf': Yf[train_indices]}
    dict_valid2 = {'Xp': Xp[valid_indices], 'psiX1p': psix1p_num[valid_indices], 'Yp': Yp[valid_indices], 'Xf': Xf[valid_indices], 'psiX1f': psix1f_num[valid_indices], 'Yf': Yf[valid_indices]}
    # K Variables
    Wh1T = weight_variable([x_deep_dict_size + num_bas_obs + 1 + y_deep_dict_size, num_outputs])  # Wh definition
    # Feed Variable Definition
    yp_feed = tf.placeholder(tf.float32, shape=[None, Yp.shape[1]])
    yf_feed = tf.placeholder(tf.float32, shape=[None, Yf.shape[1]])
    psix1p_feed = tf.placeholder(tf.float32, shape=[None, psix1p_num.shape[1]])
    psix1f_feed = tf.placeholder(tf.float32, shape=[None, psix1f_num.shape[1]])
    # Psi variables
    psix2pz_list, psix2p = initialize_tensorflow_graph(x2_params_list, xp_feed)
    psix2fz_list, psix2f = initialize_tensorflow_graph(x2_params_list, xf_feed)
    psix12p_concat = tf.concat([psix1p_feed, psix2p],axis=1)
    psix12f_concat = tf.concat([psix1f_feed, psix2f], axis=1)
    # Objective Function Variables
    dict_feed2 = {'psix1pT': psix1p_feed, 'xpT': xp_feed, 'ypT': yp_feed, 'psix1fT': psix1f_feed, 'xfT': xf_feed, 'yfT': yf_feed, 'step_size': step_size_feed}
    dict_psi2 = {'xpT': psix12p_concat,'xfT': psix12f_concat}
    dict_K2 = {'WhT': Wh1T}
    # Second optimization
    dict_model2_metrics = objective_func_output(dict_feed2, dict_psi2, dict_K2)
    all_histories2 = {'train error': [], 'validation error': [], 'train MSE': [], 'valid MSE': []}
    dict_run_info2 = {}
    all_histories2, dict_run_info2 = static_train_net(dict_train2, dict_valid2, dict_feed2, ls_dict_training_params2 ,dict_model2_metrics,all_histories2,dict_run_info2,x_params_list =x2_params_list)
    print('---   OUTPUT TRAINING COMPLETE   ---')
    print(dict_run_info2)
    print(dict_run_info1)

    # # ==============
    # # RUN 3
    # # ==============
    # # Hidden layer creation
    # x3_hidden_vars_list = np.asarray([n_xy_nn_nodes] * n_xy_nn_layers)
    # x3_hidden_vars_list[-1] = xy_deep_dict_size  # The last hidden layer being declared as the output
    # Wx3_list, bx3_list = initialize_Wblist(num_bas_obs, x3_hidden_vars_list)
    # x3_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x3_hidden_vars_list, 'W_list': Wx3_list,
    #                   'b_list': bx3_list, 'keep_prob': keep_prob, 'activation flag': activation_flag,
    #                   'res_net': res_net}
    # # Data Required
    # psix2p_num = psix2p.eval(feed_dict={xp_feed: Xp})
    # psix2f_num = psix2f.eval(feed_dict={xf_feed: Xf})
    # dict_train3 = {'Xp': Xp[train_indices], 'Xf': Xf[train_indices], 'psiX1p': psix1p_num[train_indices], 'psiX2p': psix2p_num[train_indices],'psiX2f': psix2f_num[train_indices]}
    # dict_valid3 = {'Xp': Xp[valid_indices], 'Xf': Xf[valid_indices], 'psiX1p': psix1p_num[valid_indices], 'psiX2p': psix2p_num[valid_indices],'psiX2f': psix2f_num[valid_indices]}
    # # K Variables
    # KxT_2 = weight_variable([x_deep_dict_size + num_bas_obs + y_deep_dict_size + xy_deep_dict_size + 1, y_deep_dict_size + xy_deep_dict_size])
    # # Feed variables
    # psix2p_feed = tf.placeholder(tf.float32, shape=[None, psix2p_num.shape[1]])
    # psix2f_feed = tf.placeholder(tf.float32, shape=[None, psix2f_num.shape[1]])
    # # Psi variables
    # psix3pz_list, psix3p = initialize_tensorflow_graph(x3_params_list, xp_feed)
    # psix3fz_list, psix3f = initialize_tensorflow_graph(x3_params_list, xf_feed)
    #
    # psix123p_concat = tf.concat([psix1p_feed, psix2p_feed, psix3p],axis=1)
    # psix23f_concat = tf.concat([psix2f_feed, psix3f],axis=1)
    # # Objective Function variables
    # dict_feed3 = {'psix1pT': psix1p_feed, 'psix2pT': psix2p_feed, 'psix2fT': psix2f_feed, 'xpT': xp_feed, 'xfT': xf_feed, 'step_size': step_size_feed}
    # dict_psi3 = {'xpT': psix123p_concat, 'xfT': psix23f_concat}
    # dict_K3 = {'KxT': KxT_2}
    # # Third optimization
    # dict_model3_metrics = objective_func_state({'step_size': step_size_feed}, dict_psi3, dict_K3)
    # all_histories3 = {'train error': [], 'validation error': [], 'train MSE': [], 'valid MSE': []}
    # dict_run_info3 = {}
    # all_histories3, dict_run_info3 = static_train_net(dict_train3, dict_valid3, dict_feed3, ls_dict_training_params3,dict_model3_metrics,all_histories3, dict_run_info3, x_params_list =x3_params_list)
    # print('---   OUTPUT COMPENSATED STATE TRAINING COMPLETE   ---')
    #----------------------------------------------------------------------------------------------------------------------------------
    # Post RUNS

    # AFTER RUN 1
    # all_histories = all_histories1
    # dict_run_info = dict_run_info1
    # psixp = psix1p
    # psixf = psix1f
    # KxT = KxT_11
    # dict_K = {'KxT': KxT}
    # dict_feed = {'xpT': xp_feed, 'xfT': xf_feed}
    # dict_psi = {'xpT': psixp, 'xfT': psixf}

    # AFTER RUN 2
    all_histories = {1: all_histories1, 2: all_histories2}
    dict_run_info = {1: dict_run_info1, 2: dict_run_info2}
    psixf = tf.concat([psix1f, psix2f], axis=1)
    dict_K = {'WhT': Wh1T}
    dict_feed = {'xfT': xf_feed, 'yfT': yf_feed}
    dict_psi = {'xfT': psixf}

    # # Saving all the runs
    # all_histories = {1: all_histories1, 2: all_histories2, 3: all_histories3}
    # dict_run_info = {1: dict_run_info1, 2: dict_run_info2, 3: dict_run_info3}
    # # Concatenating the psi to a single variable
    # psixp = tf.concat([psix1p,psix2p,psix3p],axis=1)
    # psixf = tf.concat([psix1f, psix2f, psix3f], axis=1)
    # dict_psi = {'xpT': psixp, 'xfT': psixf}

    # # Concatenating Ks to a single variable
    # KxT_12 = tf.constant(np.zeros(shape=(y_deep_dict_size + xy_deep_dict_size, x_deep_dict_size + num_bas_obs + 1)),dtype=tf.dtypes.float32)
    # KxT_1 = tf.concat([KxT_11, KxT_12], axis=0)
    # KxT = tf.concat([KxT_1, KxT_2], axis=1)
    # Wh2T = tf.constant(np.zeros(shape=(xy_deep_dict_size, num_outputs)), dtype=tf.dtypes.float32)
    # WhT = tf.concat([Wh1T, Wh2T], axis=0)
    # dict_K = {'KxT': KxT, 'WhT': WhT}

    # dict_feed ={'xpT': xp_feed, 'xfT': xf_feed, 'ypT': yp_feed, 'yfT': yf_feed, 'step_size': step_size_feed}

# estimate_K_stability(KxT)
# feed_dict_train, feed_dict_valid = get_fed_dict(dict_train, dict_valid, dict_feed)
# train_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_train)
# valid_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_valid)

## Saving the results of the run

# Creating a folder for saving the objects of the current run
try:
    FOLDER_NAME = '_current_run_saved_files/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(RUN_NUMBER)
except:
    FOLDER_NAME = '_current_run_saved_files/RUN_unknown'
if os.path.exists(FOLDER_NAME):
    shutil.rmtree(FOLDER_NAME)
os.mkdir(FOLDER_NAME)

dict_dump = {}
# dict_dump['Wx_list_num'] = [sess.run(W_temp) for W_temp in Wx_list]
# dict_dump['bx_list_num'] =[sess.run(b_temp) for b_temp in bx_list]
# dict_dump['Kx_num'] = sess.run(dict_K['KxT'])
dict_dump['Wh_num'] = sess.run(dict_K['WhT'])

with open(FOLDER_NAME + '/constrainedNN-Model.pickle', 'wb') as file_obj_swing:
    pickle.dump(dict_dump, file_obj_swing)
with open(FOLDER_NAME + '/run_info.pickle', 'wb') as file_obj_swing:
    pickle.dump(dict_run_info, file_obj_swing)
with open(FOLDER_NAME + '/all_histories.pickle', 'wb') as file_obj_swing:
    pickle.dump(all_histories, file_obj_swing)

saver = tf.compat.v1.train.Saver()

all_tf_var_names =[]
for items in dict_psi.keys():
    tf.compat.v1.add_to_collection('psi'+items, dict_psi[items])
    all_tf_var_names.append('psi'+items)
for items in dict_feed.keys():
    tf.compat.v1.add_to_collection(items+'_feed', dict_feed[items])
    all_tf_var_names.append(items+'_feed')
for items in dict_K.keys():
    tf.compat.v1.add_to_collection(items, dict_K[items])
    all_tf_var_names.append(items)
# Only needed if we want to retrain
# for items in list(dict_model_metrics1.keys()):
#     all_tf_var_names.append(items)
#     tf.compat.v1.add_to_collection(items, dict_model_metrics1[items])

saver_path_curr = saver.save(sess, FOLDER_NAME + '/' + data_suffix + '.ckpt')
with open(FOLDER_NAME + '/all_tf_var_names.pickle', 'wb') as handle:
    pickle.dump(all_tf_var_names,handle)
print('------ ------ -----')
print('----- Run Info ----')
print('------ ------ -----')
for items in dict_run_info.keys():
    print(pd.DataFrame(dict_run_info[items]))
    print('-----     -----     -----     -----     -----     -----     -----     -----     -----     -----     -----')

# Saving the hyperparameters
dict_hp = {'x_obs': x_deep_dict_size, 'x_layers': n_x_nn_layers, 'x_nodes': n_x_nn_nodes,'y_obs': y_deep_dict_size, 'y_layers': n_y_nn_layers, 'y_nodes': n_y_nn_nodes,'xy_obs': xy_deep_dict_size, 'xy_layers': n_xy_nn_layers, 'xy_nodes': n_xy_nn_nodes}
with open(FOLDER_NAME + '/dict_hyperparameters.pickle','wb') as handle:
    pickle.dump(dict_hp,handle)

##

