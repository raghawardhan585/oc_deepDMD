##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pickle
import random
import os
import shutil
import tensorflow as tf
import Sequential_Helper_Functions as seq

import ocdeepdmd_simulation_examples_helper_functions as oc
colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette


## Bash Script Generation


# DEVICE_TO_RUN_ON = 'microtensor'
DEVICE_TO_RUN_ON = 'optictensor'
# DEVICE_TO_RUN_ON = 'goldentensor'
DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR = 5
NO_OF_ITERATIONS_PER_GPU = 2
NO_OF_ITERATIONS_IN_CPU = 2

dict_run_conditions = {}

# MICROTENSOR CPU RUN
# dict_run_conditions[0] = {}
# dict_run_conditions[0]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':9}
# dict_run_conditions[0]['y']  = {'dict_size':1,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[0]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[1] = {}
# dict_run_conditions[1]['x']  = {'dict_size':5,'nn_layers':9,'nn_nodes':15}
# dict_run_conditions[1]['y']  = {'dict_size':5,'nn_layers':9,'nn_nodes':15}
# dict_run_conditions[1]['xy'] = {'dict_size':5,'nn_layers':9,'nn_nodes':15}
# dict_run_conditions[2] = {}
# dict_run_conditions[2]['x']  = {'dict_size':5,'nn_layers':9,'nn_nodes':15}
# dict_run_conditions[2]['y']  = {'dict_size':5,'nn_layers':9,'nn_nodes':15}
# dict_run_conditions[2]['xy'] = {'dict_size':5,'nn_layers':9,'nn_nodes':15}

# Golden tensor
# dict_run_conditions[0] = {}
# dict_run_conditions[0]['x']  = {'dict_size':1,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[0]['y']  = {'dict_size':1,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[0]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[1] = {}
# dict_run_conditions[1]['x']  = {'dict_size':1,'nn_layers':3,'nn_nodes':6}
# dict_run_conditions[1]['y']  = {'dict_size':1,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[1]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[2] = {}
# dict_run_conditions[2]['x']  = {'dict_size':1,'nn_layers':3,'nn_nodes':9}
# dict_run_conditions[2]['y']  = {'dict_size':1,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[2]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[3] = {}
# dict_run_conditions[3]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[3]['y']  = {'dict_size':1,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[3]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}

# Optic tensor
dict_run_conditions[0] = {}
dict_run_conditions[0]['x']  = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
dict_run_conditions[0]['y']  = {'dict_size':1,'nn_layers':3,'nn_nodes':3}
dict_run_conditions[0]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
dict_run_conditions[1] = {}
dict_run_conditions[1]['x']  = {'dict_size':2,'nn_layers':3,'nn_nodes':6}
dict_run_conditions[1]['y']  = {'dict_size':1,'nn_layers':3,'nn_nodes':3}
dict_run_conditions[1]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
dict_run_conditions[2] = {}
dict_run_conditions[2]['x']  = {'dict_size':2,'nn_layers':3,'nn_nodes':9}
dict_run_conditions[2]['y']  = {'dict_size':1,'nn_layers':3,'nn_nodes':3}
dict_run_conditions[2]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
dict_run_conditions[3] = {}
dict_run_conditions[3]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
dict_run_conditions[3]['y']  = {'dict_size':1,'nn_layers':3,'nn_nodes':3}
dict_run_conditions[3]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}


# dict_run_conditions[4] = {'x_dict_size':3,'x_nn_layers':4,'x_nn_nodes':18}

seq.write_bash_script(DEVICE_TO_RUN_ON, dict_run_conditions, DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR, NO_OF_ITERATIONS_PER_GPU, NO_OF_ITERATIONS_IN_CPU)


##

