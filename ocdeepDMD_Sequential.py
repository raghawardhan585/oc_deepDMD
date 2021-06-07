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
import sys # For command line inputs and for sys.exit() function
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
pd.set_option("display.max_rows", None, "display.max_columns", None)
from sklearn.metrics import r2_score


# Default Parameters
DEVICE_NAME = '/cpu:0'
RUN_NUMBER = 0
SYSTEM_NO = 402
# max_epochs = 2000
# train_error_threshold = 1e-6
# valid_error_threshold = 1e-6
# test_error_threshold = 1e-6

#  Deep Learning Optimization Parameters ##

activation_flag = 2;  # sets the activation function type to RELU[0], ELU[1], tanh[2] (initialized a certain way,dropout has to be done differently) , or tanh()

DISPLAY_SAMPLE_RATE_EPOCH = 500
TRAIN_PERCENT = 85.71429#87.5#80
keep_prob = 1.0;  # keep_prob = 1-dropout probability
res_net = 0;  # Boolean condition on whether to use a resnet connection.

regularization_lambda = 1e-10
# Neural network parameters

# ---- STATE OBSERVABLE PARAMETERS -------
x_deep_dict_size = 0
n_x_nn_layers = 1  # x_max_layers 3 works well
n_x_nn_nodes = 0  # max width_limit -4 works well

# ---- OUTPUT CONSTRAINED OBSERVABLE PARAMETERS ----
y_deep_dict_size = 2
n_y_nn_layers = 3
n_y_nn_nodes = 4

xy_deep_dict_size = 3
n_xy_nn_layers = 2
n_xy_nn_nodes = 6

best_test_error = np.inf

# RUN_1_SAVED = False
# RUN_2_SAVED = False

# 1 - Making Dynamics Linear
# 2 - Fitting the output
# 3 - Making both dynamics and output linear

# RUN_OPTIMIZATION = 1
# RUN_1_SAVED = RUN_2_SAVED = RUN_3_SAVED = False
RUN_OPTIMIZATION = 2
RUN_1_SAVED = True
RUN_2_SAVED = RUN_3_SAVED = False
# RUN_OPTIMIZATION = 3
# RUN_1_SAVED = RUN_2_SAVED = True
# RUN_3_SAVED = False



# Checking that the parameters work
if RUN_OPTIMIZATION ==2:
    if not RUN_1_SAVED:
        print('Need to save a RUN 1 before running OPTIMIZATION 2 to train on output')
        sys.exit()
if RUN_OPTIMIZATION ==3:
    if (not RUN_1_SAVED) and (not RUN_2_SAVED):
        print('Need to save a RUN 1 and RUN 2 before running OPTIMIZATION 3 to train on Koopman closure of state and output')
        sys.exit()
    elif not RUN_1_SAVED:
        print('Need to save a RUN 1 before running OPTIMIZATION 3 to train on Koopman closure of state and output')
        sys.exit()
    elif not RUN_2_SAVED:
        print('Need to save a RUN 2 before running OPTIMIZATION 3 to train on Koopman closure of state and output')
        sys.exit()

# if RUN_SEQUENTIAL_SAVED == 1:
#     RUN_1_SAVED = False
#     RUN_2_SAVED = False
# elif RUN_SEQUENTIAL_SAVED == 2:
#     RUN_1_SAVED = True
#     RUN_2_SAVED = False
# elif RUN_SEQUENTIAL_SAVED == 3:
#     RUN_1_SAVED = True
#     RUN_2_SAVED = True

# Learning Parameters
ls_dict_training_params = []
# dict_training_params = {'step_size_val': 00.5, 'train_error_threshold': float(1e-20),'valid_error_threshold': float(1e-6), 'max_epochs': 200, 'batch_size': 36} #20000
# ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 00.5, 'train_error_threshold': float(1e-20),'valid_error_threshold': float(1e-6), 'max_epochs': 80000, 'batch_size': 36} #20000
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 00.3, 'train_error_threshold': float(1e-20),'valid_error_threshold': float(1e-6), 'max_epochs': 10000, 'batch_size': 36}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.1, 'train_error_threshold': float(1e-7), 'valid_error_threshold': float(1e-7), 'max_epochs': 10000, 'batch_size': 36}
ls_dict_training_params.append(dict_training_params)
# # dict_training_params = {'step_size_val': 0.09, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 30000, 'batch_size': 2000}
# # ls_dict_training_params.append(dict_training_params)
# # dict_training_params = {'step_size_val': 0.08, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 30000, 'batch_size': 2000}
# # ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.05, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 10000, 'batch_size': 36}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.01, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 10000, 'batch_size': 36}
ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.001, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 10000, 'batch_size': 500}
# ls_dict_training_params.append(dict_training_params)

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
ls_dict_training_params3 = ls_dict_training_params

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
    if type(output_vec['Xp']) == dict:
        Xp_out = output_vec['Xp'][next(iter(output_vec['Xp']))]
        Xf_out = output_vec['Xf'][next(iter(output_vec['Xf']))]
    else:
        Xp_out = output_vec['Xp']
        Xf_out = output_vec['Xf']
    return Xp_out, Xf_out, output_vec['Yp'], output_vec['Yf']

def weight_variable(shape):
    std_dev = math.sqrt(3.0 / (shape[0] + shape[1]))
    print(std_dev)
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
    # PROPAGATION
    for k in range(1, n_depth-1):
        prev_layer_output = tf.matmul(z_list[k - 1], param_list['W_list'][k]) + param_list['b_list'][k]
        if param_list['activation flag'] == 1: # RELU
            z_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output), param_list['keep_prob']));
        if param_list['activation flag'] == 2: # ELU
            z_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output), param_list['keep_prob']));
        if param_list['activation flag'] == 3: # tanh
            z_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output), param_list['keep_prob']));
    # TERMINATION
    try:
        z_list.append(tf.matmul(z_list[n_depth-2], param_list['W_list'][n_depth-1]) + param_list['b_list'][n_depth-1])
    except:
        print('[WARNING]: There is no neural network initialized')
    if state_inclusive:
        y = tf.concat([u, z_list[-1]], axis=1)
    else:
        y = z_list[-1]
    if add_bias:
        y = tf.concat([y, tf.ones(shape=(tf.shape(y)[0], 1))], axis=1)
    result = sess.run(tf.global_variables_initializer())
    return z_list, y

def initialize_constant_tensorflow_graph(param_list,u, state_inclusive=False,add_bias=False):
    # res_net = param_list['res_net'] --- This variable is not used!
    # TODO - remove the above variable if not required at all
    # u is the input of the neural network
    z_list = [];
    n_depth = len(param_list['hidden_var_list']);
    # INITIALIZATION
    if param_list['activation flag'] == 1:  # RELU
        z_list.append(tf.nn.dropout(tf.nn.relu(tf.matmul(u, tf.constant(param_list['W_list'][0],dtype=tf.dtypes.float32)) + tf.constant(param_list['b_list'][0],dtype=tf.dtypes.float32)), param_list['keep_prob']))
    if param_list['activation flag']== 2:  # ELU
        z_list.append(tf.nn.dropout(tf.nn.elu(tf.matmul(u, tf.constant(param_list['W_list'][0],dtype=tf.dtypes.float32)) + tf.constant(param_list['b_list'][0],dtype=tf.dtypes.float32)), param_list['keep_prob']))
    if param_list['activation flag'] == 3:  # tanh
        z_list.append(tf.nn.dropout(tf.nn.tanh(tf.matmul(u, tf.constant(param_list['W_list'][0],dtype=tf.dtypes.float32)) + tf.constant(param_list['b_list'][0],dtype=tf.dtypes.float32)), param_list['keep_prob']))
    # PROPAGATION & TERMINATION
    for k in range(1, n_depth-1):
        prev_layer_output = tf.matmul(z_list[k - 1], tf.constant(param_list['W_list'][k],dtype=tf.dtypes.float32)) + tf.constant(param_list['b_list'][k],dtype=tf.dtypes.float32)
        if param_list['activation flag'] == 1: # RELU
            z_list.append(tf.nn.dropout(tf.nn.relu(prev_layer_output), param_list['keep_prob']));
        if param_list['activation flag'] == 2: # ELU
            z_list.append(tf.nn.dropout(tf.nn.elu(prev_layer_output), param_list['keep_prob']));
        if param_list['activation flag'] == 3: # tanh
            z_list.append(tf.nn.dropout(tf.nn.tanh(prev_layer_output), param_list['keep_prob']));
    # TERMINATION
    try:
        z_list.append(tf.matmul(z_list[n_depth - 2], tf.constant(param_list['W_list'][n_depth-1],dtype=tf.dtypes.float32)) + tf.constant(param_list['b_list'][n_depth-1],dtype=tf.dtypes.float32))
    except:
        print('[WARNING]: There is no neural network initialized')
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

def generate_hyperparam_entry(feed_dict_train, feed_dict_valid, dict_model_metrics, n_epochs_run, dict_run_params,x_hidden_vars_list,variance_weighted = True):
    if variance_weighted:
        weight_by_variance = dict_model_metrics['variance_weight'].eval(feed_dict=feed_dict_train)
    else:
        weight_by_variance = np.ones(shape = (num_x_observables_total+1,))
    training_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_train)
    validation_error = dict_model_metrics['loss_fn'].eval(feed_dict=feed_dict_valid)
    SSE_train = dict_model_metrics['SSE'].eval(feed_dict=feed_dict_train)
    SST_train = dict_model_metrics['SST'].eval(feed_dict=feed_dict_train)
    SSE_valid = dict_model_metrics['SSE'].eval(feed_dict=feed_dict_valid)
    SST_valid= dict_model_metrics['SST'].eval(feed_dict=feed_dict_valid)
    training_accuracy = (1 - np.sum(np.nan_to_num(SSE_train/SST_train*weight_by_variance)))*100
    validation_accuracy = (1 - np.sum(np.nan_to_num(SSE_valid/SST_valid*weight_by_variance)))*100
    # training_accuracy = dict_model_metrics['accuracy'].eval(feed_dict=feed_dict_train)
    # validation_accuracy = dict_model_metrics['accuracy'].eval(feed_dict=feed_dict_valid)
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
    dict_hp['regularization factor'] = regularization_lambda
    dict_hp['training error'] = training_error
    dict_hp['validation error'] = validation_error
    dict_hp['r^2 training accuracy'] = training_accuracy
    dict_hp['r^2 validation accuracy'] = validation_accuracy
    # dict_hp['r2 train'] = r2_score(dict_model_metrics['xf_true'].eval(feed_dict=feed_dict_train),dict_model_metrics['xf_pred'].eval(feed_dict=feed_dict_train),multioutput='variance_weighted')
    # dict_hp['r2 valid'] = r2_score(dict_model_metrics['xf_true'].eval(feed_dict=feed_dict_valid),
    #                                dict_model_metrics['xf_pred'].eval(feed_dict=feed_dict_valid),
    #                                multioutput='variance_weighted')
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

    dict_model_perf_metrics['MSE'] = tf.math.reduce_mean(tf.math.square(Y_prediction_error))
    dict_model_perf_metrics['loss_fn'] = dict_model_perf_metrics['MSE'] + regularization_lambda * tf.math.reduce_sum(tf.math.square(dict_K['WhT']))
    dict_model_perf_metrics['optimizer'] = tf.train.AdagradOptimizer(dict_feed['step_size']).minimize(dict_model_perf_metrics ['loss_fn'])
    # Accuracy computation
    # dict_model_perf_metrics['SST'] = tf.math.reduce_sum(tf.math.square(dict_feed['yfT']- tf.math.reduce_mean(dict_feed['yfT'])), axis=0)
    # dict_model_perf_metrics['SSE'] = tf.math.reduce_sum(tf.math.square(Yf_prediction_error), axis=0)
    # Specific for RNAseq
    dict_model_perf_metrics['SST'] = tf.math.reduce_sum(tf.math.square(dict_feed['yfT'] - tf.math.reduce_mean(dict_feed['yfT'])))
    dict_model_perf_metrics['SSE'] = tf.math.reduce_sum(tf.math.square(Yf_prediction_error))
    dict_model_perf_metrics ['accuracy'] = (1 - tf.divide(dict_model_perf_metrics['SSE'], dict_model_perf_metrics['SST'])) * 100
    dict_model_perf_metrics['variance_weight'] = tf.constant(1,dtype=tf.float32)
    sess.run(tf.global_variables_initializer())
    return dict_model_perf_metrics


# objective_func_state({'step_size': step_size_feed}, dict_psi1, dict_K1,objective_weight_variance= True)
def objective_func_state(dict_feed,dict_psi,dict_K):
    dict_model_perf_metrics ={}
    psiXf_predicted = tf.matmul(dict_psi['xpT'], dict_K['KxT'])
    psiXf_prediction_error = dict_psi['xfT'] - psiXf_predicted

    # TODO Delete this if this turns out to be moot which is mostly going to be the case
    # if objective_weight_variance:
    #     weight_by_variance = tf.math.reduce_variance(dict_psi['xfT'],axis=0)
    #     weight_by_variance = tf.reshape(tf.math.divide_no_nan(tf.math.cumsum(weight_by_variance), tf.math.reduce_sum(weight_by_variance)),shape =(-1,1))
    # else:
    #     weight_by_variance = tf.constant(np.ones(shape = (len(dict_K['KxT'].eval()),1)),dtype = tf.float32)
    #     # try:
    # # dict_model_perf_metrics['MSE'] = tf.math.reduce_sum(tf.matmul(tf.transpose(tf.expand_dims(tf.math.reduce_mean(tf.math.square(psiXf_prediction_error), axis=0), axis=1)),weight_by_variance))
    # dict_model_perf_metrics['MSE'] = tf.math.reduce_mean(tf.math.square(psiXf_prediction_error))
    # dict_model_perf_metrics['loss_fn'] = dict_model_perf_metrics['MSE'] +  tf.constant(regularization_lambda,dtype=tf.float32) * tf.math.reduce_sum(tf.math.square(dict_K['KxT']))
    # SST = tf.math.reduce_sum(tf.math.square(dict_psi['xfT'] - tf.math.reduce_mean(dict_psi['xfT'], axis=0)), axis=0)
    # SSE = tf.math.reduce_sum(tf.math.square(psiXf_prediction_error), axis=0)
    # dict_model_perf_metrics['accuracy'] = tf.reduce_sum((1 - tf.matmul(tf.reshape(tf.math.divide_no_nan(SSE, SST),shape=(1,-1)), weight_by_variance)) * 100)
    # except:
    #     print('No objective weight variance')
    # dict_model_perf_metrics['xf_pred'] = tf.matmul(dict_psi['xpT'], dict_K['KxT'])
    # dict_model_perf_metrics['xf_true'] = dict_psi['xfT']
    dict_model_perf_metrics['MSE'] = tf.math.reduce_mean(tf.math.square(psiXf_prediction_error))
    dict_model_perf_metrics['loss_fn'] = dict_model_perf_metrics['MSE'] + regularization_lambda * tf.math.reduce_sum(tf.math.square(dict_K['KxT']))
    dict_model_perf_metrics['SST'] = tf.math.reduce_sum(tf.math.square(dict_psi['xfT'] - tf.math.reduce_mean(dict_psi['xfT'], axis=0)), axis=0)
    dict_model_perf_metrics['SSE'] = tf.math.reduce_sum(tf.math.square(psiXf_prediction_error), axis=0)
    dict_model_perf_metrics['accuracy'] = (1 - tf.math.reduce_mean(tf.math.divide_no_nan(dict_model_perf_metrics['SSE'], dict_model_perf_metrics['SST']))) * 100
    dict_model_perf_metrics['optimizer'] = tf.train.AdagradOptimizer(dict_feed['step_size']).minimize(dict_model_perf_metrics['loss_fn'])
    # Weight Variance
    dict_model_perf_metrics['variance_weight'] = tf.math.reduce_variance(dict_psi['xfT'], axis=0)
    dict_model_perf_metrics['variance_weight'] = tf.math.divide_no_nan(dict_model_perf_metrics['variance_weight'],
        tf.math.reduce_sum(dict_model_perf_metrics['variance_weight']))
    # Initialize global variables
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

def get_best_K_DMD(XpT_train,XfT_train,XpT_valid,XfT_valid):
    Xp_train = XpT_train.T
    Xf_train = XfT_train.T
    Xp_valid = XpT_valid.T
    Xf_valid = XfT_valid.T

    # Model 1
    U,S,Vh = np.linalg.svd(Xp_train)
    V = Vh.T.conj()
    Uh = U.T.conj()
    A_hat = np.zeros(shape = U.shape)
    ls_error_train = []
    ls_error_valid = []
    for i in range(len(S)):
        A_hat = A_hat + (1/S[i])*np.matmul(np.matmul(Xf_train,V[:,i:i+1]),Uh[i:i+1,:])
        ls_error_train.append(np.mean(np.square((Xf_train - np.matmul(A_hat,Xp_train)))))
        if Xp_valid.shape[1] != 0:
            ls_error_valid.append(np.mean(np.square((Xf_valid - np.matmul(A_hat, Xp_valid)))))
    if Xp_valid.shape[1] == 0:
        ls_error = np.array(ls_error_train)
    else:
        ls_error = np.array(ls_error_train) + np.array(ls_error_valid)
    nPC_opt = np.where(ls_error==np.min(ls_error))[0][0] + 1
    A_hat_opt = np.zeros(shape = U.shape)
    for i in range(nPC_opt):
        A_hat_opt = A_hat_opt + (1/S[i])*np.matmul(np.matmul(Xf_train,V[:,i:i+1]),Uh[i:i+1,:])
    # # Prediction of model 1
    # Xf_train_hat1 = np.multiply(A_hat_opt,Xp_train)
    # Xf_valid_hat1 = np.multiply(A_hat_opt, Xp_valid)
    # MSE1 = np.mean((Xf_train - Xf_train_hat1)**2) + np.mean((Xf_valid - Xf_valid_hat1)**2)

    # lin_model = LinearRegression().fit(XpT_train,XfT_train)
    # XfT_train_hat2 = lin_model.predict(XpT_train)
    # XfT_valid_hat2 = lin_model.predict(XpT_valid)
    # MSE2 = np.mean((XfT_train - XfT_train_hat2) ** 2) + np.mean((XfT_valid - XfT_valid_hat2) ** 2)
    # # Compare the two models
    # if MSE1 >= MSE2:
    #     AT_opt = A_hat_opt.T
    # else:
    #     A =
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


def static_train_net(dict_train, dict_valid, dict_feed, ls_dict_training_params, dict_model_metrics_curr, all_histories, dict_run_info,x_params_list={}):
    feed_dict_train, feed_dict_valid = get_fed_dict(dict_train,dict_valid,dict_feed)
    print('Starting Training Error: ',dict_model_metrics_curr['MSE'].eval(feed_dict = feed_dict_train))
    print('Starting Validation Error: ', dict_model_metrics_curr['MSE'].eval(feed_dict=feed_dict_valid))
    # --------
    try :
        run_info_index = list(dict_run_info.keys())[-1]
    except:
        run_info_index = 0
    for dict_train_params_i in ls_dict_training_params:
        display_train_params(dict_train_params_i)
        all_histories, n_epochs_run = train_net_v2(dict_train,feed_dict_train, feed_dict_valid, dict_feed, dict_model_metrics_curr, dict_train_params_i, all_histories)
        dict_run_info[run_info_index] = generate_hyperparam_entry(feed_dict_train, feed_dict_valid,dict_model_metrics_curr,n_epochs_run, dict_train_params_i,x_params_list['hidden_var_list'])
        print('Current Training Error  :', dict_run_info[run_info_index]['training error'])
        print('Current Validation Error      :', dict_run_info[run_info_index]['validation error'])
        # estimate_K_stability(KxT)
        run_info_index += 1
    print(dict_model_metrics_curr['MSE'].eval(feed_dict=feed_dict_train))
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
data_suffix = 'System_'+str(SYSTEM_NO)+'_ocDeepDMDdata.pickle'

# CMD Line Argument (Override) Inputs:
# TODO - Rearrange this section

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
if len(sys.argv)>13:
    regularization_lambda = np.float(sys.argv[13])

# Sanity Check
if (RUN_OPTIMIZATION ==1) and (RUN_1_SAVED):
    if ((x_deep_dict_size >= 0) and (x_deep_dict_size >= 0) and (x_deep_dict_size >= 0)):
        print('[INFO] You have asked to continue training a saved run but you are specifying parameters for output')
        print('[INFO] Overriding the saved run parameters')
        RUN_1_SAVED = False
if (RUN_OPTIMIZATION ==2) and (RUN_2_SAVED):
    if ((y_deep_dict_size >= 0) and (y_deep_dict_size >= 0) and (y_deep_dict_size >= 0)):
        print('[INFO] You have asked to continue training a saved run but you are specifying parameters for output')
        print('[INFO] Overriding the saved run parameters')
        RUN_2_SAVED = False
if (RUN_OPTIMIZATION ==3) and (RUN_3_SAVED):
    if ((y_deep_dict_size >= 0) and (y_deep_dict_size >= 0) and (y_deep_dict_size >= 0)):
        print('[INFO] You have asked to continue training a saved run but you are specifying parameters for state-output Koopman closure')
        print('[INFO] Overriding the saved run parameters')
        RUN_3_SAVED = False

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

objective_weight_variance = (dict_train['Xp'].var(axis=0)/np.sum(dict_train['Xp'].var(axis=0)) ).reshape(-1,1)
objective_weight_variance = np.concatenate([objective_weight_variance,np.array([[1]])],axis=0)

num_x_observables_total = x_deep_dict_size + num_bas_obs


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
    # Data Required
    # We already have xp and xf given
    dict_train1 = {'Xp': Xp[train_indices], 'Xf': Xf[train_indices]}
    dict_valid1 = {'Xp': Xp[valid_indices], 'Xf': Xf[valid_indices]}
    # Feed Variable definitions
    xp_feed = tf.placeholder(tf.float32, shape=[None, num_bas_obs])
    xf_feed = tf.placeholder(tf.float32, shape=[None, num_bas_obs])
    step_size_feed = tf.placeholder(tf.float32, shape=[])

    if RUN_1_SAVED:
        # SYSTEM_NO = 23
        print(SYSTEM_NO)
        with open('System_' + str(SYSTEM_NO) + '_BestRun_1.pickle','rb') as handle:
            var_i = pickle.load(handle)
        x_deep_dict_size = var_i['x_obs']
        n_x_nn_layers = var_i['x_layers']
        n_x_nn_nodes = var_i['x_nodes']
        x1_hidden_vars_list = np.asarray([var_i['x_nodes']] * var_i['x_layers'])
        x1_hidden_vars_list[-1] = var_i['x_obs']  # The last hidden layer being declared as the output
        Wx1_list_num = var_i['Wx_list_num']
        bx1_list_num = var_i['bx_list_num']
        KxT_11_num = var_i['Kx_num']
    if RUN_OPTIMIZATION == 1:
        if RUN_1_SAVED:
            Wx1_list = [tf.Variable(items) for items in Wx1_list_num]
            bx1_list = [tf.Variable(items) for items in bx1_list_num]
            KxT_11 = tf.Variable(KxT_11_num)
        else:
            # Hidden layer creation
            x1_hidden_vars_list = np.asarray([n_x_nn_nodes] * n_x_nn_layers)
            x1_hidden_vars_list[-1] = x_deep_dict_size # The last hidden layer being declared as the output
            Wx1_list, bx1_list = initialize_Wblist(num_bas_obs, x1_hidden_vars_list)
            # K Variables  -    Kx definition w/ bias
            KxT_11 = weight_variable([num_x_observables_total + 1, num_x_observables_total])
            # Type 1 initialization - using linear dmd method
            # A_hat_opt = get_best_K_DMD(dict_train['Xp'], dict_train['Xf'],dict_valid['Xp'], dict_valid['Xf'])
            # sess.run(tf.global_variables_initializer())
            # KxT_11 = tf.Variable(sess.run(KxT_11[0:num_bas_obs, 0:num_bas_obs].assign(A_hat_opt)))
            # Type 2 initialization - using direct least squares
            A_hat_opt,b_hat_opt = get_best_K_DMD2(dict_train['Xp'], dict_train['Xf'])
            sess.run(tf.global_variables_initializer())
            KxT_11 = tf.Variable(sess.run(KxT_11[0:num_bas_obs, 0:num_bas_obs].assign(A_hat_opt)))
            sess.run(tf.global_variables_initializer())
            KxT_11 = tf.Variable(sess.run(KxT_11[-1, 0:num_bas_obs].assign(b_hat_opt)))
            # Other K variables
            last_col = tf.constant(np.zeros(shape=(num_x_observables_total, 1)), dtype=tf.dtypes.float32)
            last_col = tf.concat([last_col, [[1.]]], axis=0)
            KxT_11 = tf.concat([KxT_11, last_col], axis=1)
        sess.run(tf.global_variables_initializer())
        x1_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x1_hidden_vars_list, 'W_list': Wx1_list,
                          'b_list': bx1_list, 'keep_prob': keep_prob, 'activation flag': activation_flag, 'res_net': res_net}
        # Psi variables
        psix1pz_list, psix1p = initialize_tensorflow_graph(x1_params_list, xp_feed, state_inclusive=True, add_bias=True)
        psix1fz_list, psix1f = initialize_tensorflow_graph(x1_params_list, xf_feed, state_inclusive=True, add_bias=True)
        # Objective Function Variables
        dict_feed1 = { 'xpT': xp_feed, 'xfT': xf_feed, 'step_size': step_size_feed}
        dict_psi1 = {'xpT': psix1p, 'xfT': psix1f}
        dict_K1 ={'KxT':KxT_11}
        # First optimization
        print('---------    TRAINING BEGINS   ---------')
        # print(psix1p.eval(feed_dict={xp_feed: Xp[1:5,:]}))
        # dict_model1_metrics = objective_func_state({'step_size': step_size_feed}, dict_psi1, dict_K1,objective_weight_variance)
        dict_model1_metrics = objective_func_state({'step_size': step_size_feed}, dict_psi1, dict_K1)
        all_histories1 = {'train error': [], 'validation error': [], 'train MSE': [], 'valid MSE': []}
        dict_run_info1 = {}
        all_histories1, dict_run_info1 = static_train_net(dict_train1, dict_valid1, dict_feed1, ls_dict_training_params1,dict_model1_metrics,all_histories1,dict_run_info1,x_params_list =x1_params_list)
        print('---   STATE TRAINING COMPLETE   ---')
        # print(psix1p.eval(feed_dict={xp_feed: Xp[1:5,:]}))
        estimate_K_stability(KxT_11)
        # Post Run 1 Saves
        KxT_11_num = sess.run(KxT_11)
        Wx1_list_num = sess.run(Wx1_list)
        bx1_list_num = sess.run(bx1_list)
        print(pd.DataFrame(dict_run_info1))

    x1_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x1_hidden_vars_list, 'W_list': Wx1_list_num,
                      'b_list': bx1_list_num, 'keep_prob': keep_prob, 'activation flag': activation_flag,
                      'res_net': res_net}
    psix1pz_list_const, psix1p_const = initialize_constant_tensorflow_graph(x1_params_list, xp_feed, state_inclusive=True, add_bias=True)
    psix1fz_list_const, psix1f_const = initialize_constant_tensorflow_graph(x1_params_list, xf_feed, state_inclusive=True, add_bias=True)

# SYSTEM_NO = 6
# sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
# with open(sys_folder_name + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle', 'rb') as handle:
#     dict_indexed_data = pickle.load(handle)

# feed_dict_train1, feed_dict_valid1 = get_fed_dict(dict_train1, dict_valid1, dict_feed1)
#
# print('Training Error: ', np.mean(np.square(psix1f_const.eval(feed_dict = feed_dict_train1) - np.matmul(psix1p_const.eval(feed_dict = feed_dict_train1) ,KxT_11_num))))
# print('Validation Error: ', np.mean(np.square(psix1f_const.eval(feed_dict = feed_dict_valid1) - np.matmul(psix1p_const.eval(feed_dict = feed_dict_valid1) ,KxT_11_num))))
# print('Train Error: ',dict_model1_metrics['loss_fn'].eval(feed_dict = feed_dict_train1))
# print('Valid Error: ',dict_model1_metrics['loss_fn'].eval(feed_dict = feed_dict_valid1))
# import ocdeepdmd_simulation_examples_helper_functions as oc
# dict_params = {'psixpT':psix1p_const,'psixfT':psix1f_const,'xpT_feed':xp_feed,'xfT_feed':xf_feed, 'KxT_num':KxT_11_num}
# dict_p = oc.model_prediction_state_only(dict_indexed_data, dict_params, SYSTEM_NUMBER=6)
# i = 0
# for j in range(2):
#     plt.plot(dict_p[i]['X_scaled'][:, j], '.', color=colors[j])
#     plt.plot(dict_p[i]['X_scaled_est_one_step'][:, j], color=colors[j], label='x' + str(j + 1) + '[scaled]')
#     # plt.plot(dict_run[ls_runs[i]]['X_scaled_est_n_step'][:, j], color=colors[j],label='x' + str(j + 1) + '[scaled]')
# plt.legend()
# plt.show()

    # ==============
    # RUN 2
    # ==============

    # if (RUN_OPTIMIZATION == 2) and RUN_2_SAVED:
    #     # Getting data of existing Run 2
    #     with open('System_' + str(SYSTEM_NO) + '_BestRun_2.pickle', 'rb') as handle:
    #         var_i = pickle.load(handle)
    #     y_deep_dict_size = var_i['y_obs']
    #     n_y_nn_layers = var_i['y_layers']
    #     n_y_nn_nodes = var_i['y_nodes']
    #     Wx2_list_num = var_i['Wy_list_num']
    #     bx2_list_num = var_i['by_list_num']
    #     Wh1T_num = var_i['Wh_num']
    #     x2_hidden_vars_list = np.asarray([n_y_nn_nodes] * n_y_nn_layers)
    #     x2_hidden_vars_list[-1] = y_deep_dict_size  # The last hidden layer being declared as the output
    #     # Data Required
    #     psix1p_num = psix1p_const.eval(feed_dict={xp_feed: Xp})
    #     psix1f_num = psix1f_const.eval(feed_dict={xf_feed: Xf})
    #     dict_train2 = {'Xp': Xp[train_indices], 'psiX1p': psix1p_num[train_indices], 'Yp': Yp[train_indices],
    #                    'Xf': Xf[train_indices], 'psiX1f': psix1f_num[train_indices], 'Yf': Yf[train_indices]}
    #     dict_valid2 = {'Xp': Xp[valid_indices], 'psiX1p': psix1p_num[valid_indices], 'Yp': Yp[valid_indices],
    #                    'Xf': Xf[valid_indices], 'psiX1f': psix1f_num[valid_indices], 'Yf': Yf[valid_indices]}
    #     # Creating placeholder variables for Run 2
    #     yp_feed = tf.placeholder(tf.float32, shape=[None, Yp.shape[1]])
    #     yf_feed = tf.placeholder(tf.float32, shape=[None, Yf.shape[1]])
    #     psix1p_feed = tf.placeholder(tf.float32, shape=[None, psix1p_num.shape[1]])
    #     psix1f_feed = tf.placeholder(tf.float32, shape=[None, psix1f_num.shape[1]])
    #     # Initializing the parameters
    #     Wx2_list = [tf.Variable(items) for items in Wx2_list_num]
    #     bx2_list = [tf.Variable(items) for items in bx2_list_num]
    #     Wh1T = tf.Variable(Wh1T_num)
    #     sess.run(tf.global_variables_initializer())
    #     x2_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x2_hidden_vars_list, 'W_list': Wx2_list,
    #                       'b_list': bx2_list, 'keep_prob': keep_prob, 'activation flag': activation_flag,'res_net': res_net}
    #     # Psi variables
    #     psix2pz_list, psix2p = initialize_tensorflow_graph(x2_params_list, xp_feed)
    #     psix2fz_list, psix2f = initialize_tensorflow_graph(x2_params_list, xf_feed)
    #     psix12p_concat = tf.concat([psix1p_feed, psix2p], axis=1)
    #     psix12f_concat = tf.concat([psix1f_feed, psix2f], axis=1)
    #     # Objective Function Variables
    #     dict_feed2 = {'psix1pT': psix1p_feed, 'xpT': xp_feed, 'ypT': yp_feed, 'psix1fT': psix1f_feed,
    #                   'xfT': xf_feed, 'yfT': yf_feed, 'step_size': step_size_feed}
    #     dict_psi2 = {'xpT': psix12p_concat, 'xfT': psix12f_concat}
    #     dict_K2 = {'WhT': Wh1T}
    #     # Second optimization
    #     dict_model2_metrics = objective_func_output(dict_feed2, dict_psi2, dict_K2)
    #     all_histories2 = {'train error': [], 'validation error': [], 'train MSE': [], 'valid MSE': []}
    #     dict_run_info2 = {}
    #     all_histories2, dict_run_info2 = static_train_net(dict_train2, dict_valid2, dict_feed2,
    #                                                       ls_dict_training_params2, dict_model2_metrics,
    #                                                       all_histories2, dict_run_info2, x_params_list=x2_params_list)
    #     print('---   OUTPUT TRAINING COMPLETE   ---')
    #     print(pd.DataFrame(dict_run_info2))
    #     # Post Run 2 Saves
    #     Wh1T_num = sess.run(Wh1T)
    #     Wx2_list_num = sess.run(Wx2_list)
    #     bx2_list_num = sess.run(bx2_list)
    #     x2_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x2_hidden_vars_list, 'W_list': Wx2_list_num,
    #                       'b_list': bx2_list_num, 'keep_prob': keep_prob, 'activation flag': activation_flag,
    #                       'res_net': res_net}
    #     psix2pz_list_const, psix2p_const = initialize_constant_tensorflow_graph(x2_params_list, xp_feed)
    #     psix2fz_list_const, psix2f_const = initialize_constant_tensorflow_graph(x2_params_list, xf_feed)

    if RUN_OPTIMIZATION == 2:
        psix1p_num = psix1p_const.eval(feed_dict={xp_feed: Xp})
        psix1f_num = psix1f_const.eval(feed_dict={xf_feed: Xf})
        yp_feed = tf.placeholder(tf.float32, shape=[None, Yp.shape[1]])
        yf_feed = tf.placeholder(tf.float32, shape=[None, Yf.shape[1]])
        psix1p_feed = tf.placeholder(tf.float32, shape=[None, psix1p_num.shape[1]])
        psix1f_feed = tf.placeholder(tf.float32, shape=[None, psix1f_num.shape[1]])
        # Data Required
        dict_train2 = {'Xp': Xp[train_indices], 'psiX1p': psix1p_num[train_indices], 'Yp': Yp[train_indices],
                       'Xf': Xf[train_indices], 'psiX1f': psix1f_num[train_indices], 'Yf': Yf[train_indices]}
        dict_valid2 = {'Xp': Xp[valid_indices], 'psiX1p': psix1p_num[valid_indices], 'Yp': Yp[valid_indices],
                       'Xf': Xf[valid_indices], 'psiX1f': psix1f_num[valid_indices], 'Yf': Yf[valid_indices]}
        x2_hidden_vars_list = np.asarray([n_y_nn_nodes] * n_y_nn_layers)
        x2_hidden_vars_list[-1] = y_deep_dict_size  # The last hidden layer being declared as the output
        Wx2_list, bx2_list = initialize_Wblist(num_bas_obs, x2_hidden_vars_list)
        sess.run(tf.global_variables_initializer())
        # K Variables
        Wh1T = weight_variable([x_deep_dict_size + num_bas_obs + 1 + y_deep_dict_size, num_outputs])
        C_hat_opt = get_best_K_DMD2(dict_train2['psiX1f'], dict_train2['Yf'],fit_intercept=False)
        sess.run(tf.global_variables_initializer())
        Wh1T = tf.Variable(sess.run(Wh1T[0:x_deep_dict_size + num_bas_obs + 1, :].assign(C_hat_opt)))

        x2_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x2_hidden_vars_list, 'W_list': Wx2_list,
                          'b_list': bx2_list, 'keep_prob': keep_prob, 'activation flag': activation_flag, 'res_net': res_net}
        # Psi variables
        psix2pz_list, psix2p = initialize_tensorflow_graph(x2_params_list, xp_feed)
        psix2fz_list, psix2f = initialize_tensorflow_graph(x2_params_list, xf_feed)
        psix12p_concat = tf.concat([psix1p_feed, psix2p], axis=1)
        psix12f_concat = tf.concat([psix1f_feed, psix2f], axis=1)
        # Objective Function Variables
        dict_feed2 = {'psix1pT': psix1p_feed, 'xpT': xp_feed, 'ypT': yp_feed, 'psix1fT': psix1f_feed,
                      'xfT': xf_feed, 'yfT': yf_feed, 'step_size': step_size_feed}
        dict_psi2 = {'xpT': psix12p_concat, 'xfT': psix12f_concat}
        dict_K2 = {'WhT': Wh1T}
        # Second optimization
        dict_model2_metrics = objective_func_output(dict_feed2, dict_psi2, dict_K2)
        all_histories2 = {'train error': [], 'validation error': [], 'train MSE': [], 'valid MSE': []}
        dict_run_info2 = {}
        all_histories2, dict_run_info2 = static_train_net(dict_train2, dict_valid2, dict_feed2,
                                                          ls_dict_training_params2, dict_model2_metrics,
                                                          all_histories2, dict_run_info2, x_params_list=x2_params_list)
        print('---   OUTPUT TRAINING COMPLETE   ---')
        print(pd.DataFrame(dict_run_info2))
        # Post Run 2 Saves
        Wh1T_num = sess.run(Wh1T)
        Wx2_list_num = sess.run(Wx2_list)
        bx2_list_num = sess.run(bx2_list)
        x2_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x2_hidden_vars_list, 'W_list': Wx2_list_num,
                          'b_list': bx2_list_num, 'keep_prob': keep_prob, 'activation flag': activation_flag,
                          'res_net': res_net}
        psix2pz_list_const, psix2p_const = initialize_constant_tensorflow_graph(x2_params_list, xp_feed)
        psix2fz_list_const, psix2f_const = initialize_constant_tensorflow_graph(x2_params_list, xf_feed)


    elif RUN_2_SAVED:
        # psix1p_num = psix1p_const.eval(feed_dict={xp_feed: Xp})
        # psix1f_num = psix1f_const.eval(feed_dict={xf_feed: Xf})
        with open('System_' + str(SYSTEM_NO) + '_BestRun_2.pickle', 'rb') as handle:
            var_i = pickle.load(handle)
        y_deep_dict_size = var_i['y_obs']
        n_y_nn_layers = var_i['y_layers']
        n_y_nn_nodes = var_i['y_nodes']
        Wx2_list_num = var_i['Wy_list_num']
        bx2_list_num = var_i['by_list_num']
        Wh1T_num = var_i['Wh_num']
        x2_hidden_vars_list = np.asarray([n_y_nn_nodes] * n_y_nn_layers)
        x2_hidden_vars_list[-1] = y_deep_dict_size  # The last hidden layer being declared as the output
        x2_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x2_hidden_vars_list, 'W_list': Wx2_list_num,
                          'b_list': bx2_list_num, 'keep_prob': keep_prob, 'activation flag': activation_flag,
                          'res_net': res_net}
        psix2pz_list_const, psix2p_const = initialize_constant_tensorflow_graph(x2_params_list, xp_feed)
        psix2fz_list_const, psix2f_const = initialize_constant_tensorflow_graph(x2_params_list, xf_feed)
    else:
        print('[INFO] Run 2 was not done - No output was trained')

# feed_dict_train2, feed_dict_valid2 = get_fed_dict(dict_train2, dict_valid2, dict_feed2)
#
# psiXf = np.concatenate([psix1f_const.eval(feed_dict = feed_dict_train2),psix2f_const.eval(feed_dict = feed_dict_train2)],axis=1)
# psiXp = np.concatenate([psix1p_const.eval(feed_dict = feed_dict_train2),psix2p_const.eval(feed_dict = feed_dict_train2)],axis=1)
# print('[PHASE 2] Training Error: ', np.mean(np.square(np.concatenate([ yf_feed.eval(feed_dict = feed_dict_train2) - np.matmul(psiXf,Wh1T_num) , yp_feed.eval(feed_dict = feed_dict_train2)-np.matmul(psiXp,Wh1T_num)],axis=0))))
# psiXf = np.concatenate([psix1f_const.eval(feed_dict = feed_dict_valid2),psix2f_const.eval(feed_dict = feed_dict_valid2)],axis=1)
# psiXp = np.concatenate([psix1p_const.eval(feed_dict = feed_dict_valid2),psix2p_const.eval(feed_dict = feed_dict_valid2)],axis=1)
# print('[PHASE 2] Validation Error: ', np.mean(np.square(np.concatenate([ yf_feed.eval(feed_dict = feed_dict_valid2) - np.matmul(psiXf,Wh1T_num) , yp_feed.eval(feed_dict = feed_dict_valid2)-np.matmul(psiXp,Wh1T_num)],axis=0))))
    # ==============
    #  RUN 3
    # ==============
    if (RUN_OPTIMIZATION == 3):
        if RUN_3_SAVED:
            print('Yet to do a saved run for Run 3')
            # TODO - For Later
            # with open('System_' + str(SYSTEM_NO) + '_BestRun_3.pickle', 'rb') as handle:
            #     var_i = pickle.load(handle)
            # x_deep_dict_size = var_i['x_obs']
            # n_x_nn_layers = var_i['x_layers']
            # n_x_nn_nodes = var_i['x_nodes']
            # x1_hidden_vars_list = np.asarray([var_i['x_nodes']] * var_i['x_layers'])
            # x1_hidden_vars_list[-1] = var_i['x_obs']  # The last hidden layer being declared as the output
            # Wx1_list_num = var_i['Wx_list_num']
            # bx1_list_num = var_i['bx_list_num']
            # KxT_11_num = var_i['Kx_num']
        else:
            # Hidden layer creation
            x3_hidden_vars_list = np.asarray([n_xy_nn_nodes] * n_xy_nn_layers)
            x3_hidden_vars_list[-1] = xy_deep_dict_size  # The last hidden layer being declared as the output
            Wx3_list, bx3_list = initialize_Wblist(num_bas_obs, x3_hidden_vars_list)
        x3_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x3_hidden_vars_list, 'W_list': Wx3_list,
                          'b_list': bx3_list, 'keep_prob': keep_prob, 'activation flag': activation_flag,'res_net': res_net}
        # Data Required
        psix1p_num = psix1p_const.eval(feed_dict={xp_feed: Xp})
        psix1f_num = psix1f_const.eval(feed_dict={xf_feed: Xf})
        psix2p_num = psix2p_const.eval(feed_dict={xp_feed: Xp})
        psix2f_num = psix2f_const.eval(feed_dict={xf_feed: Xf})
        dict_train3 = {'Xp': Xp[train_indices], 'Xf': Xf[train_indices], 'psiX1p': psix1p_num[train_indices], 'psiX2p': psix2p_num[train_indices],'psiX2f': psix2f_num[train_indices]}
        dict_valid3 = {'Xp': Xp[valid_indices], 'Xf': Xf[valid_indices], 'psiX1p': psix1p_num[valid_indices], 'psiX2p': psix2p_num[valid_indices],'psiX2f': psix2f_num[valid_indices]}
        # K Variables
        KxT_2 = weight_variable([x_deep_dict_size + num_bas_obs + y_deep_dict_size + xy_deep_dict_size + 1, y_deep_dict_size + xy_deep_dict_size])
        # Feed variables
        psix1p_feed = tf.placeholder(tf.float32, shape=[None, psix1p_num.shape[1]])
        psix1f_feed = tf.placeholder(tf.float32, shape=[None, psix1f_num.shape[1]])
        psix2p_feed = tf.placeholder(tf.float32, shape=[None, psix2p_num.shape[1]])
        psix2f_feed = tf.placeholder(tf.float32, shape=[None, psix2f_num.shape[1]])
        yp_feed = tf.placeholder(tf.float32, shape=[None, Yp.shape[1]])
        yf_feed = tf.placeholder(tf.float32, shape=[None, Yf.shape[1]])
        # Psi variables
        psix3pz_list, psix3p = initialize_tensorflow_graph(x3_params_list, xp_feed)
        psix3fz_list, psix3f = initialize_tensorflow_graph(x3_params_list, xf_feed)

        psix123p_concat = tf.concat([psix1p_feed, psix2p_feed, psix3p],axis=1)
        psix23f_concat = tf.concat([psix2f_feed, psix3f],axis=1)
        # Objective Function variables
        dict_feed3 = {'psix1pT': psix1p_feed, 'psix2pT': psix2p_feed, 'psix2fT': psix2f_feed, 'xpT': xp_feed, 'xfT': xf_feed, 'step_size': step_size_feed}
        dict_psi3 = {'xpT': psix123p_concat, 'xfT': psix23f_concat}
        dict_K3 = {'KxT': KxT_2}
        # Third optimization
        dict_model3_metrics = objective_func_state({'step_size': step_size_feed}, dict_psi3, dict_K3)
        all_histories3 = {'train error': [], 'validation error': [], 'train MSE': [], 'valid MSE': []}
        dict_run_info3 = {}
        all_histories3, dict_run_info3 = static_train_net(dict_train3, dict_valid3, dict_feed3, ls_dict_training_params3,dict_model3_metrics,all_histories3, dict_run_info3, x_params_list =x3_params_list)
        print('---   OUTPUT COMPENSATED STATE TRAINING COMPLETE   ---')
        print(pd.DataFrame(dict_run_info3))
        # Post Run 3 Saves
        KxT_2_num = sess.run(KxT_2)
        Wx3_list_num = sess.run(Wx3_list)
        bx3_list_num = sess.run(bx3_list)
        x3_params_list = {'n_base_states': num_bas_obs, 'hidden_var_list': x3_hidden_vars_list, 'W_list': Wx3_list_num,
                          'b_list': bx3_list_num, 'keep_prob': keep_prob, 'activation flag': activation_flag, 'res_net': res_net}
        # feed_dict_train3, feed_dict_valid3 = get_fed_dict(dict_train3, dict_valid3, dict_feed3)
        # print('Train Error: ', dict_model3_metrics['loss_fn'].eval(feed_dict=feed_dict_train3))
        # print('Valid Error: ', dict_model3_metrics['loss_fn'].eval(feed_dict=feed_dict_valid3))
        # print('Training Error: ', np.mean(np.square(psix23f_concat.eval(feed_dict=feed_dict_train3) - np.matmul(psix123p_concat.eval(feed_dict=feed_dict_train3),KxT_2_num))))
        # print('Validation Error: ', np.mean(np.square(psix23f_concat.eval(feed_dict=feed_dict_valid3) - np.matmul(psix123p_concat.eval(feed_dict=feed_dict_valid3),KxT_2_num))))
        psix3pz_list_const, psix3p_const = initialize_constant_tensorflow_graph(x3_params_list, xp_feed)
        psix3fz_list_const, psix3f_const = initialize_constant_tensorflow_graph(x3_params_list, xf_feed)
        psi3p = tf.concat([psix1p_const, psix2p_const, psix3p_const], axis=1)
        psi3f = tf.concat([psix2f_const, psix3f_const], axis=1)
        # print('=====')
        # print('Training Error: ', np.mean(np.square(psix23f_concat.eval(feed_dict=feed_dict_train3) - np.matmul(psix123p_concat.eval(feed_dict=feed_dict_train3), KxT_2_num))))
        # print('Validation Error: ', np.mean(np.square(psix23f_concat.eval(feed_dict=feed_dict_valid3) - np.matmul(psix123p_concat.eval(feed_dict=feed_dict_valid3), KxT_2_num))))
    else:
        print('[INFO] Run 3 was not done - No state-output Koopman closure was trained')

# ---------
# ##
# feed_dict_train3, feed_dict_valid3 = get_fed_dict(dict_train3, dict_valid3, dict_feed3)
# print('Train Error: ',dict_model3_metrics['loss_fn'].eval(feed_dict = feed_dict_train3))
# print('Valid Error: ',dict_model3_metrics['loss_fn'].eval(feed_dict = feed_dict_valid3))
#
# psi3p = tf.concat([psix1p_const,psix2p_const,psix3p_const],axis=1)
# psi3f = tf.concat([psix2f_const,psix3f_const],axis=1)
#
# print('Training Error: ', np.mean(np.square(psi3f.eval(feed_dict = feed_dict_train1) - np.matmul(psi3p.eval(feed_dict = feed_dict_train1) ,KxT_2_num))))
# print('Validation Error: ', np.mean(np.square(psi3f.eval(feed_dict = feed_dict_valid1) - np.matmul(psi3p.eval(feed_dict = feed_dict_valid1) ,KxT_2_num))))
# Plot the above results
# import matplotlib.pyplot as plt
# psix3p_num = psix3p.eval(feed_dict={xp_feed: Xp})
# psix3f_num = psix3f.eval(feed_dict={xf_feed: Xf})
#
# psixp_num = np.concatenate([psix1p_num,psix2p_num,psix3p_num],axis=1)
# psixf_num = np.concatenate([psix1f_num,psix2f_num,psix3f_num],axis=1)
#
# KxT_12_num = np.zeros(shape=(y_deep_dict_size + xy_deep_dict_size, x_deep_dict_size + num_bas_obs + 1))
# KxT_1_num = np.concatenate([KxT_11_num, KxT_12_num], axis=0)
# KxT_num = np.concatenate([KxT_1_num, KxT_2_num], axis=1)
#
# psixf_num_est = np.matmul(psixp_num,KxT_num)
#
# print('All Observables Error[MSE]: ', np.mean(np.square(psixf_num_est - psixf_num)))
# print('State Error[MSE]: ', np.mean(np.square(psix1f_num - np.matmul(psix1p_num,KxT_11_num))))
# print('Extra observables Error[MSE]: ', np.mean(np.square(np.concatenate([psix2f_num,psix3f_num],axis=1) - np.matmul(psixp_num,KxT_2_num))))
#
# colors = [[0.68627453, 0.12156863, 0.16470589],
#           [0.96862745, 0.84705883, 0.40000001],
#           [0.83137256, 0.53333336, 0.6156863],
#           [0.03529412, 0.01960784, 0.14509805],
#           [0.90980393, 0.59607846, 0.78039217],
#           [0.69803923, 0.87843138, 0.72941178],
#           [0.20784314, 0.81568629, 0.89411765]];
# colors = np.asarray(colors);  # defines a color palette
#
# for j in range(len(psixf_num[0])):
#     plt.plot(psixf_num[:,j], '.', color = colors[np.mod(j,7)],linewidth = int(j/7+1))
#     plt.plot(psixf_num_est[:,j], color = colors[np.mod(j,7)],linewidth = int(j/7+1), label='psi ' + str(j + 1))
#     # plt.plot(dict_run[ls_runs[i]]['X_scaled_est_n_step'][:, j], color=colors[j],label='x' + str(j + 1) + '[scaled]')
# plt.legend()
# plt.show()
#----------------------------------------------------------------------------------------------------------------------------------
    # Post RUNS


    if RUN_OPTIMIZATION == 1:
        # AFTER RUN 1
        all_histories = all_histories1
        dict_run_info = dict_run_info1
        KxT = tf.Variable(KxT_11_num)
        dict_K = {'KxT': KxT}
        dict_feed = {'xpT': xp_feed, 'xfT': xf_feed}
        dict_psi = {'xpT': psix1p_const, 'xfT': psix1f_const}
        sess.run(tf.global_variables_initializer())
    elif RUN_OPTIMIZATION == 2:
        # AFTER RUN 2
        # all_histories = {1: all_histories1, 2: all_histories2}
        # dict_run_info = {1: dict_run_info1, 2: dict_run_info2}
        all_histories =  all_histories2
        dict_run_info = dict_run_info2
        psixf = tf.concat([psix1f_const, psix2f_const], axis=1) # TODO - Verify this output variable when unlocking
        WhT = tf.Variable(Wh1T_num)
        sess.run(tf.global_variables_initializer())
        dict_K = {'WhT': WhT}
        dict_feed = {'xfT': xf_feed, 'yfT': yf_feed}
        dict_psi = {'xfT': psixf}
    elif RUN_OPTIMIZATION == 3:
        # AFTER RUN 3
        # all_histories = {1: all_histories1, 2: all_histories2, 3: all_histories3}
        # dict_run_info = {1: dict_run_info1, 2: dict_run_info2, 3: dict_run_info3}
        all_histories =  all_histories3
        dict_run_info = dict_run_info3
        psixp = tf.concat([psix1p_const,psix2p_const,psix3p_const],axis=1)
        psixf = tf.concat([psix1f_const, psix2f_const, psix3f_const], axis=1)
        dict_psi = {'xpT': psixp, 'xfT': psixf}
        dict_feed ={'xpT': xp_feed, 'xfT': xf_feed, 'ypT': yp_feed, 'yfT': yf_feed}
        # Concatenating Ks to a single variable
        KxT_12_num = np.zeros(shape=(y_deep_dict_size + xy_deep_dict_size, x_deep_dict_size + num_bas_obs + 1))
        KxT_1_num = np.concatenate([KxT_11_num, KxT_12_num], axis=0)
        KxT_num = np.concatenate([KxT_1_num, KxT_2_num], axis=1)
        KxT = tf.Variable(KxT_num)
        Wh2T = tf.constant(np.zeros(shape=(xy_deep_dict_size, num_outputs)), dtype=tf.dtypes.float32)
        WhT_num = tf.concat([Wh1T_num, Wh2T], axis=0)
        WhT = tf.Variable(WhT_num)
        dict_K = {'KxT': KxT, 'WhT': WhT}
        sess.run(tf.global_variables_initializer())
        estimate_K_stability(KxT,True)
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

if RUN_OPTIMIZATION == 1:
    # RUN 1
    dict_dump = {}
    dict_dump['Wx_list_num'] = Wx1_list_num
    dict_dump['bx_list_num'] = bx1_list_num
    dict_dump['Kx_num'] = sess.run(dict_K['KxT'])
elif RUN_OPTIMIZATION == 2:
    # RUN 2
    dict_dump = {}
    dict_dump['Wx_list_num'] = Wx1_list_num
    dict_dump['bx_list_num'] = bx1_list_num
    dict_dump['Wy_list_num'] = Wx2_list_num
    dict_dump['by_list_num'] = bx2_list_num
    dict_dump['Wh_num'] = sess.run(dict_K['WhT'])
elif RUN_OPTIMIZATION == 3:
    # RUN 3
    dict_dump = {}
    dict_dump['Wx_list_num'] = Wx1_list_num
    dict_dump['bx_list_num'] = bx1_list_num
    dict_dump['Wy_list_num'] = Wx2_list_num
    dict_dump['by_list_num'] = bx2_list_num
    dict_dump['Wxy_list_num'] = Wx3_list_num
    dict_dump['bxy_list_num'] = bx3_list_num
    dict_dump['Kx_num'] = sess.run(dict_K['KxT'])
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

# print(all_tf_var_names)

saver_path_curr = saver.save(sess, FOLDER_NAME + '/' + data_suffix + '.ckpt')
with open(FOLDER_NAME + '/all_tf_var_names.pickle', 'wb') as handle:
    pickle.dump(all_tf_var_names,handle)
print('------ ------ -----    -----     -----     -----     -----     -----     -----     -----     -----')
print('-----     -----     -----     -----     ----  Run Info  ---    -----     -----     -----     -----')
print('------ ------ -----    -----     -----     -----     -----     -----     -----     -----     -----')
# for items in dict_run_info.keys():
print(pd.DataFrame(dict_run_info))
print('-----     -----     -----     -----     -----     -----     -----     -----     -----     -----     -----')

# Saving the hyperparameters
dict_hp = {'x_obs': x_deep_dict_size, 'x_layers': n_x_nn_layers, 'x_nodes': n_x_nn_nodes,'y_obs': y_deep_dict_size, 'y_layers': n_y_nn_layers, 'y_nodes': n_y_nn_nodes,'xy_obs': xy_deep_dict_size, 'xy_layers': n_xy_nn_layers, 'xy_nodes': n_xy_nn_nodes, 'regularization factor': regularization_lambda}
dict_hp['r2 train'] = dict_run_info[list(dict_run_info.keys())[-1]]['r^2 training accuracy']
dict_hp['r2 valid'] = dict_run_info[list(dict_run_info.keys())[-1]]['r^2 validation accuracy']
with open(FOLDER_NAME + '/dict_hyperparameters.pickle','wb') as handle:
    pickle.dump(dict_hp,handle)

##
from sklearn.metrics import r2_score
lin_model1_X = LinearRegression(fit_intercept=True).fit(dict_train['Xp'],dict_train['Xf'])
print(r2_score(dict_train['Xf'],lin_model1_X.predict(dict_train['Xp']),multioutput='variance_weighted'))
print(r2_score(dict_valid['Xf'],lin_model1_X.predict(dict_valid['Xp']),multioutput='variance_weighted'))
