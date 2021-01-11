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
PROCESS_VARIABLE = 'x'
DEVICE_NAME = '/cpu:0'
RUN_NUMBER = 0
SYSTEM_NO = 60
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
n_layers = 3  # x_max_layers 3 works well
n_nodes = 3  # max width_limit -4 works well

best_test_error = np.inf


# 1 - Making Dynamics Linear
# 2 - Fitting the output
# 3 - Making both dynamics and output linear

# Learning Parameters
ls_dict_training_params = []
dict_training_params = {'step_size_val': 00.5, 'train_error_threshold': float(1e-6),'valid_error_threshold': float(1e-6), 'max_epochs': 10000, 'batch_size': 1000}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 00.3, 'train_error_threshold': float(1e-6),'valid_error_threshold': float(1e-6), 'max_epochs': 10000, 'batch_size': 1000}
ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.1, 'train_error_threshold': float(1e-7), 'valid_error_threshold': float(1e-7), 'max_epochs': 10000, 'batch_size': 1000}
ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.09, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 10000, 'batch_size': 500}
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.08, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 10000, 'batch_size': 500}
# ls_dict_training_params.append(dict_training_params)
dict_training_params = {'step_size_val': 0.05, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 10000, 'batch_size': 1000}
ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.01, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 10000, 'batch_size': 500}
# ls_dict_training_params.append(dict_training_params)
# dict_training_params = {'step_size_val': 0.001, 'train_error_threshold': float(1e-8), 'valid_error_threshold': float(1e-8), 'max_epochs': 10000, 'batch_size': 500}
# ls_dict_training_params.append(dict_training_params)


sess = tf.InteractiveSession()

# Required Functions


def load_pickle_data(file_path,LEARN_DYNAMICS= True):
    with open(file_path,'rb') as handle:
        output_vec = pickle.load(handle)
    if LEARN_DYNAMICS:
        X = output_vec['Xp']
        Y = output_vec['Xf']
    else:
        X = output_vec['Xf']
        Y = output_vec['Yf']
    return X,Y

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
def initialize_tensorflow_graph(param_list,u):
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
    z_list.append(tf.matmul(z_list[n_depth-2], param_list['W_list'][n_depth-1]) + param_list['b_list'][n_depth-1])
    y = z_list[-1]
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


def objective_func_state(dict_feed,psix,A):
    dict_model_perf_metrics ={}
    y_predicted = tf.matmul(dict_feed['x'], A) + psix
    y_prediction_error = dict_feed['y'] - y_predicted
    dict_model_perf_metrics['loss_fn'] = tf.math.reduce_mean(tf.math.square(y_prediction_error))
    dict_model_perf_metrics['optimizer'] = tf.train.AdagradOptimizer(dict_feed['step_size']).minimize(dict_model_perf_metrics['loss_fn'])
    # Mean Squared Error
    dict_model_perf_metrics['MSE'] = tf.math.reduce_mean(tf.math.square(y_prediction_error))
    # Accuracy computation
    SST = tf.math.reduce_sum(tf.math.square(dict_feed['y']), axis=0)
    SSE = tf.math.reduce_sum(tf.math.square(y_prediction_error), axis=0)
    dict_model_perf_metrics['accuracy'] = (1 - tf.math.reduce_max(tf.divide(SSE, SST))) * 100
    sess.run(tf.global_variables_initializer())
    return dict_model_perf_metrics

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
    return  A_hat_opt.T


def static_train_net(dict_train, dict_valid, dict_feed, ls_dict_training_params, dict_model_metrics, all_histories, dict_run_info,x_params_list={}):
    fed_dict_train ={dict_feed['x']: dict_train['X'],dict_feed['y']: dict_train['Y']}
    fed_dict_valid = {dict_feed['x']: dict_valid['X'], dict_feed['y']: dict_valid['Y']}
    print(dict_model_metrics['MSE'].eval(feed_dict = fed_dict_train))
    # --------
    try :
        run_info_index = list(dict_run_info.keys())[-1]
    except:
        run_info_index = 0
    for dict_train_params_i in ls_dict_training_params:
        display_train_params(dict_train_params_i)
        all_histories, n_epochs_run = train_net_v2(dict_train,fed_dict_train, fed_dict_valid, dict_feed, dict_model_metrics, dict_train_params_i, all_histories)
        dict_run_info[run_info_index] = generate_hyperparam_entry(fed_dict_train, fed_dict_valid,dict_model_metrics,n_epochs_run, dict_train_params_i,x_params_list['hidden_var_list'])
        print('Current Training Error  :', dict_run_info[run_info_index]['training error'])
        print('Current Validation Error      :', dict_run_info[run_info_index]['validation error'])
        # estimate_K_stability(KxT)
        run_info_index += 1
    print(dict_model_metrics['MSE'].eval(feed_dict=fed_dict_train))
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
            fed_dict_train_curr = {dict_feed['x']: dict_train['X'][train_indices],dict_feed['y']: dict_train['Y'][train_indices],dict_feed['step_size']:  dict_run_params['step_size_val']}
            dict_model_metrics['optimizer'].run(feed_dict=fed_dict_train_curr)
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
import sys
if len(sys.argv)>1:
    DEVICE_NAME = sys.argv[1]
    if DEVICE_NAME not in ['/cpu:0','/gpu:0','/gpu:1','/gpu:2','/gpu:3']:
        DEVICE_NAME = '/cpu:0'
if len(sys.argv)>2:
    SYSTEM_NO = sys.argv[2]
    data_suffix = 'System_'+ str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
if len(sys.argv) > 3:
    PROCESS_VARIABLE = sys.argv[3]
if len(sys.argv) > 4:
    RUN_NUMBER = np.int(sys.argv[4])
if len(sys.argv)>5:
    n_layers = np.int(sys.argv[5])
if len(sys.argv)>6:
    n_nodes = np.int(sys.argv[6])

if PROCESS_VARIABLE == 'x':
    LEARN_DYNAMICS = True
elif PROCESS_VARIABLE == 'y':
    LEARN_DYNAMICS = False
else:
    print('Process Variable invalid!!! EXITING')
    exit()


data_file = data_directory + data_suffix
X,Y = load_pickle_data(data_file, LEARN_DYNAMICS)
num_states = len(X[0])
num_all_samples = len(X)
num_outputs = len(Y[0])

# Train/Test Split for Benchmarking Forecasting Later
num_trains = np.int(num_all_samples * TRAIN_PERCENT / 100)
train_indices = np.arange(0, num_trains, 1)
valid_indices = np.arange(num_trains,num_all_samples,1)
dict_train = {}
dict_valid = {}
dict_train['X'] = X[train_indices]
dict_valid['X'] = X[valid_indices]
dict_train['Y'] = Y[train_indices]
dict_valid['Y'] = Y[valid_indices]

# Hidden layer creation
hidden_vars_list = np.asarray([n_nodes] * n_layers)
hidden_vars_list[-1] = num_outputs  # The last hidden layer being declared as the output

# Display info
print("[INFO] Number of total samples: " + repr(num_all_samples))
print("[INFO] Observable dimension of a sample: " + repr(num_states))
print("[INFO] X shape : " + repr(X.shape))
print("[INFO] Y shape : " + repr(Y.shape))
print("Number of training snapshots: " + repr(len(train_indices)))
print("Number of validation snapshots: " + repr(len(valid_indices)))
print("[INFO] STATE - hidden_vars_list: " + repr(hidden_vars_list))

##
# ============================
# LEARNING THE STATE DYNAMICS
# ============================
with tf.device(DEVICE_NAME):
    # Feed Variable definitions
    x_feed = tf.placeholder(tf.float32, shape=[None, num_states])
    y_feed = tf.placeholder(tf.float32, shape=[None, num_outputs])
    step_size_feed = tf.placeholder(tf.float32, shape=[])
    # Tensorflow graph
    Wx1_list, bx1_list = initialize_Wblist(num_states, hidden_vars_list)
    x_params_list = {'n_base_states': num_states, 'hidden_var_list': hidden_vars_list,'W_list': Wx1_list,
                      'b_list': bx1_list,'keep_prob': keep_prob,'activation flag': activation_flag,'res_net': res_net}
    # K Variables  -    Kx definition w/ bias
    A = weight_variable([num_states, num_states])
    A_hat_opt = get_best_K_DMD(dict_train['X'], dict_train['Y'],dict_valid['X'], dict_valid['Y'])
    sess.run(tf.global_variables_initializer())
    A = tf.Variable(sess.run(A[0:num_states, 0:num_states].assign(A_hat_opt)))
    # Psi variables
    psixz_list, psix = initialize_tensorflow_graph(x_params_list, x_feed)
    # Objective Function Variables
    dict_feed = { 'x': x_feed,'y': y_feed, 'step_size': step_size_feed}
    dict_psi = {'xpT': psix}
    dict_K ={'KxT':A}
    # First optimization
    print('---------    TRAINING BEGINS   ---------')
    # print(psix1p.eval(feed_dict={xp_feed: Xp[1:5,:]}))
    dict_model_metrics = objective_func_state(dict_feed, psix, A)
    all_histories = {'train error': [], 'validation error': [], 'train MSE': [], 'valid MSE': []}
    dict_run_info = {}
    all_histories, dict_run_info = static_train_net(dict_train, dict_valid, dict_feed, ls_dict_training_params,dict_model_metrics,all_histories,dict_run_info,x_params_list =x_params_list)
    print('---   TRAINING COMPLETE   ---')
    # print(psix1p.eval(feed_dict={xp_feed: Xp[1:5,:]}))
    A_num = sess.run(A)
    Wx1_list_num = sess.run(Wx1_list)
    bx1_list_num = sess.run(bx1_list)
# AFTER RUN 1
dict_K = {'AT': A}
dict_feed = {'xT': x_feed}
dict_psi = {'xT': psix}
sess.run(tf.global_variables_initializer())
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
dict_dump['Wx_list_num'] = Wx1_list_num
dict_dump['bx_list_num'] = bx1_list_num
dict_dump['A_num'] = A_num

with open(FOLDER_NAME + '/constrainedNN-Model.pickle', 'wb') as file_obj_swing:
    pickle.dump(dict_dump, file_obj_swing)
with open(FOLDER_NAME + '/run_info.pickle', 'wb') as file_obj_swing:
    pickle.dump(dict_run_info, file_obj_swing)
with open(FOLDER_NAME + '/all_histories.pickle', 'wb') as file_obj_swing:
    pickle.dump(all_histories, file_obj_swing)

saver = tf.compat.v1.train.Saver()
tf.compat.v1.add_to_collection('psix', x_feed)
tf.compat.v1.add_to_collection('x_feed', x_feed)
tf.compat.v1.add_to_collection('AT', A)
all_tf_var_names =['psix','x_feed','AT']
saver_path_curr = saver.save(sess, FOLDER_NAME + '/' + data_suffix + '.ckpt')
with open(FOLDER_NAME + '/all_tf_var_names.pickle', 'wb') as handle:
    pickle.dump(all_tf_var_names,handle)


print('------ ------ -----')
print('----- Run Info ----')
print('------ ------ -----')
for items in dict_run_info.keys():
    print(pd.DataFrame(dict_run_info[items]).T)
print('-----     -----     -----     -----     -----     -----     -----     -----     -----     -----     -----')

# Saving the hyperparameters
dict_hp = {'n_layers': n_layers, 'n_nodes': n_nodes, 'process_variable': PROCESS_VARIABLE}
with open(FOLDER_NAME + '/dict_hyperparameters.pickle','wb') as handle:
    pickle.dump(dict_hp,handle)



