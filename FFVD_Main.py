#! /usr/local/bin/ipython --

import sys
import os
import logging
import pickle
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import glob
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

import numpy as np
from vfegpssm.models import RegressionModel
import argparse
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import json

from sklearn.model_selection import train_test_split
# from synthetic_data_generation_v2 import syn_gen

from pprint import pprint
from datetime import datetime

def next_path(path_pattern):
    i = 1
    while os.path.exists(path_pattern % i):
        i = i * 2
    a, b = (i / 2, i)
    while a + 1 < b:
        c = (a + b) / 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)
    directory = path_pattern % b
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

def kink():
    T = 20
    x1 = 0.5
    process_noise_std = 0.05
    observation_noise_std = 0.2

    observations = (pd.read_csv('../RealData-test/kink_observations.csv', delimiter = ',').values)

    observation_std = np.std(observations[:int(len(observations)/2)])
    control_inputs = []

    observations = (observations-np.mean(observations[:int(len(observations)/2)]))/observation_std

    Y_train = observations[:int(len(observations)/2)]
    Y_test = observations[int(len(observations)/2):]
    return Y_train, Y_test, control_inputs, observation_std


def read_real_data(file_path):
    filess = glob.glob('../vfe/'+file_path+'data.npz')

    syndata = np.load(filess[0])
    Y_test = syndata['yy_test']
    Y_train = syndata['yy_train']

    control_inputs = syndata['control_inputs']

    Y_train_std = np.std(Y_train)
    Y_train_mean = np.mean(Y_train, axis=0)

    control_inputs_std = np.std(control_inputs)
    control_inputs_mean = np.mean(control_inputs, axis=0)

    control_inputs = (control_inputs-control_inputs_mean)/control_inputs_std
    Y_train = (Y_train-Y_train_mean)/Y_train_std
    Y_test = (Y_test-Y_train_mean)/Y_train_std

    control_inputs = tf.convert_to_tensor(control_inputs)

    return Y_train, Y_test, control_inputs, Y_train_std, Y_train_mean, control_inputs_mean, control_inputs_std


def load_synthetic_data(file_path):
    filess = glob.glob(file_path+'data.npz')

    syndata = np.load(filess[0])


    Y_train = syndata['yy_train']
    Y_test = syndata['yy_test']

    X_train = syndata['xx_train']
    X_test = syndata['xx_test']

    mean_Y_train = np.mean(Y_train)
    observation_std = np.std(Y_train)

    CC = syndata['CC']
    dd = syndata['dd']
    observation_noise_std = np.asarray([[syndata['R_chol']]])
    Q_diag_val = np.asarray([syndata['Q_chol']])


    # Y_train = (Y_train-mean_Y_train)/observation_std
    # Y_test = (Y_test-mean_Y_train)/observation_std
    # # yy_seq, xx_seq, f_seq, RR, CC, DD, ZZ, UU
    # observations = syndata['yy_seq']
    # # print('RR is: ', syndata['RR'])
    # print('CC is: ', syndata['CC'])
    # print('DD is: ', syndata['DD'])
    #
    # # xx_seq = syndata['xx_seq']
    #
    # observation_std = np.std(observations[:int(len(observations)/2)])
    control_inputs = tf.convert_to_tensor(np.asarray([[]]*(Y_test.shape[0]+Y_train.shape[0])))

    # observations = (observations-np.mean(observations[:int(len(observations)/2)]))/observation_std

    # Y_train = observations[:int(len(observations)/2)][:, None]
    # Y_test = observations[int(len(observations)/2):][:, None]
    # Y_train = observations[:, None]
    # Y_test = []

    return X_train, X_test, Y_train, Y_test, control_inputs, observation_std, CC, dd, observation_noise_std, Q_diag_val


def create_dataset(file_path):
    # names_seq = ['ballbeam', 'dryer', 'flutter', 'actuator', 'drive', 'gas_furnace']
    pathss = 'data/'
    # file_path = data_i
    # data_i = [index_i if file_path[:-1] == names_seq[index_i] for index_i in range()]
    if (file_path == 'ballbeam/') or (file_path == 'dryer/') or (file_path == 'flutter/'):
        data = pd.read_csv(pathss + file_path[:-1] + '.dat', sep='\t', header=None)
        xx = data.values[:, 0][:, None]
        observations = data.values[:, 1][:, None]
    elif file_path == 'actuator/':
        mat = scipy.io.loadmat(pathss + file_path[:-1] + '.mat')
        xx = mat['u']
        observations = mat['p']
    elif file_path == 'drive/':
        mat = scipy.io.loadmat(pathss + file_path[:-1] + '.mat')
        xx = mat['u1']
        observations = mat['z1']
    elif file_path == 'gas_furnace/':
        data = pd.read_csv(pathss + file_path[:-1] + '.csv', sep=',', header=0)
        xx = data.values[:, 0][:, None]
        observations = data.values[:, 1][:, None]


    control_inputs = tf.convert_to_tensor((xx-np.mean(xx))/np.std(xx), dtype = tf.float64)
    control_inputs_mean = np.mean(xx)
    control_inputs_std = np.std(xx)
    lens = observations.shape[0]

    Y_train_std = np.std(observations[:int(lens/2)])
    Y_train_mean = np.mean(observations[:int(lens/2)])

    observations = (observations-Y_train_mean)/Y_train_std

    Y_test = observations[int(lens/2):]
    Y_train = observations[:int(lens/2)]


    return Y_train, Y_test, control_inputs, Y_train_std, Y_train_mean, control_inputs_mean, control_inputs_std


def save_results(test_mll):
    results = dict()
    results['model'] = args.model
    results['num_inducing'] = args.num_inducing
    results['minibatch_size'] = args.minibatch_size
    results['n_layers'] = args.n_layers
    results['prior_type'] = args.prior_type
    results['fold'] = args.fold
    results['dataset'] = args.dataset
    results['test_mnll'] = -test_mll

    # filepath = next_path(os.path.dirname(os.path.realpath(__file__)) + '/results/' + '/run-%04d/')
    # pprint(results)
    savepath = 'result/'
    with open(savepath + 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def main(file_path, ini_file):

    # log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # tf.summary.create_file_writer(log_dir)
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    now = datetime.now()  # current date and time
    fileid = now.strftime("%Y_%m_%d_%H_%M_%S_%f")

    #if not os.path.isdir(save_path):
    #    os.mkdir(save_path)
    if file_path == 'linear_dynamic_systems/':
        X_train, X_test, Y_train, Y_test, control_inputs, Y_train_std, CC, dd, observation_noise_std, Q_chol = load_synthetic_data(file_path)
        if len(X_train.shape) == 3:
            X_train = X_train[:, :, 0]
            X_test = X_test[:, :, 0]
    else:
       # Y_train, Y_test, control_inputs, Y_train_std, Y_train_mean, control_inputs_mean, control_inputs_std = read_real_data(file_path)
       Y_train, Y_test, control_inputs, Y_train_std, Y_train_mean, control_inputs_mean, control_inputs_std = create_dataset(file_path)

       factnonlin = np.load(ini_file, allow_pickle=True)
       qx1_mu_ini = factnonlin['qx1_mu_ini']
       qx1_cov_chol_ini = factnonlin['qx1_cov_chol_ini']
       Umu_ini = factnonlin['Umu_ini']
       Ucov_chol_ini = factnonlin['Ucov_chol_ini']
       Q_sqrt_ini = factnonlin['Q_sqrt_ini']
       kernel_variance = factnonlin['kernel_variance']
       kernel_lengthscales = factnonlin['kernel_lengthscales']
       y_samples_training_factnonlin = np.mean(factnonlin['y_samples_training'], axis=1)
       y_samples_testing_factnonlin = np.mean(factnonlin['y_samples_testing'][1:, :, 0], axis=1)
       C_val_ini = factnonlin['C_val']
       d_val_ini = factnonlin['d_val']
       Z_val_ini = factnonlin['Z_val']

       x_samples_training_factnonlin = np.mean(factnonlin['x_samples_training'], axis=1)


       R_chol_val_ini = factnonlin['R_chol_val']


    model = RegressionModel(args.prior_type)


    if file_path == 'linear_dynamic_systems/':
        model.ARGS.CC = tf.convert_to_tensor(CC)
        model.ARGS.DD = tf.convert_to_tensor(dd)
        # model.ARGS.QQ_chol = np.sqrt([0.1])
        # model.ARGS.RR_chol = np.sqrt([[0.1]])
        # model.ARGS.QQ_chol = tf.convert_to_tensor(Q_chol)
        model.ARGS.QQ_chol = None
        model.ARGS.RR_chol = tf.convert_to_tensor(observation_noise_std)
        # model.ARGS.RR_chol = None
    else:
        model.ARGS.CC = tf.convert_to_tensor(C_val_ini.T, dtype = tf.float64)
        model.ARGS.DD = tf.convert_to_tensor(d_val_ini, dtype = tf.float64)
        model.ARGS.QQ_chol = Q_sqrt_ini
        model.ARGS.RR_chol = tf.convert_to_tensor(R_chol_val_ini, dtype = tf.float64)

    model.ARGS.lengthscales = kernel_lengthscales
    model.ARGS.variance = kernel_variance

    model.ARGS.UU_ini = Umu_ini.T
    model.ARGS.XX_0_ini = qx1_mu_ini
    model.ARGS.x_initialization = x_samples_training_factnonlin

    model.ARGS.Y_train_std = Y_train_std

    model.ARGS.control_inputs = control_inputs
    model.ARGS.num_inducing = args.num_inducing
    model.ARGS.minibatch_size = args.minibatch_size
    model.ARGS.iterations = args.iterations
    model.ARGS.n_layers = args.n_layers
    model.ARGS.num_posterior_samples = args.samples
    model.ARGS.posterior_sample_spacing = args.posterior_sample_spacing
    model.ARGS.prior_type = args.prior_type
    model.ARGS.full_cov = False
    model.ARGS.x_dims = args.x_dims
    model.ARGS.case_val = args.case_val

    model.ARGS.hyperparameter_sampling = False

    if model.ARGS.case_val == 1:

        model.ARGS.kernel_optimization = True
        model.ARGS.U_optimization = True
        model.ARGS.Z_optimization = True
        model.ARGS.U_collapse = False
        case = 'C1'
        model.ARGS.X_PG = False

    elif model.ARGS.case_val == 2:

        model.ARGS.kernel_optimization = False
        model.ARGS.U_optimization = False
        model.ARGS.Z_optimization = True
        model.ARGS.U_collapse = False
        case = 'C2'
        model.ARGS.X_PG = False

    elif model.ARGS.case_val == 3:

        model.ARGS.kernel_optimization = False
        model.ARGS.U_optimization = False
        model.ARGS.Z_optimization = False
        model.ARGS.U_collapse = False
        case = 'C3'
        model.ARGS.X_PG = False

    elif model.ARGS.case_val == 4:

        model.ARGS.kernel_optimization = True
        model.ARGS.U_optimization = False
        model.ARGS.Z_optimization = True
        model.ARGS.U_collapse = True
        case = 'C4'
        model.ARGS.X_PG = False

    elif model.ARGS.case_val == 5:
        model.ARGS.kernel_optimization = False
        model.ARGS.U_optimization = False
        model.ARGS.Z_optimization = True
        model.ARGS.U_collapse = True
        case = 'C5'
        model.ARGS.X_PG = False

    elif model.ARGS.case_val == 6:

        model.ARGS.kernel_optimization = True
        model.ARGS.U_optimization = True
        model.ARGS.Z_optimization = True
        model.ARGS.U_collapse = False
        case = 'C6'
        model.ARGS.X_PG = True

    model.ARGS.PG_particles = 100

    tensorboard_savepath = 'results'

    model.ARGS.kink_flag = False
    model.ARGS.posterior_sample_spacing = 32
    model.ARGS.kernel_type = args.kernel_type
    model.ARGS.kernel_train_flag = args.kernel_train_flag
    model.ARGS.test_len = len(Y_test)


    fileid += 'file_id'+str(args.file_id)


    model.ARGS.ZZ = tf.convert_to_tensor(Z_val_ini, dtype = tf.float64)
    logger.info('Number of inducing points: %d' % model.ARGS.num_inducing)
    # model.fit(Y_train, kernel_type = model.ARGS.kernel_type, kernel_train_flag = model.ARGS.kernel_train_flag, epsilon=.01, X_train = X_train, X_test = X_test)
    model.fit(Y_train, Y_test = Y_test, tensorboard_savepath = tensorboard_savepath, dataname = file_path[:-1], fileid = fileid, kernel_type = model.ARGS.kernel_type, kernel_train_flag = model.ARGS.kernel_train_flag, epsilon=.01)

    model.model.collect_samples_formal(model.ARGS.num_posterior_samples, model.ARGS.posterior_sample_spacing,
                                      model.ARGS.control_inputs, test_len=model.ARGS.test_len,
                                      sghmc_var_len=len(model.model.vars), U_collapse=model.ARGS.U_collapse,
                                       Y_test = Y_test, Y_train_std=Y_train_std, save_path_file=tensorboard_savepath+'/'+file_path[:-1]+'/'+case+'VFE_result_'+file_path[:-1]+'_'+fileid+'.npz',
                                       Y_train = Y_train, case = case, ll_seq = model.ll_seq, running_time_seq = model.running_time_seq, PG_num = model.ARGS.PG_particles)

    tf.keras.backend.clear_session()

if __name__ == '__main__':
    # argv = sys.argv[1]
    parser = argparse.ArgumentParser(description='Run FFVD-gpssm experiment')
    parser.add_argument('--num_inducing', type=int, default=100)
    parser.add_argument('--minibatch_size', type=int, default=1000)

    parser.add_argument('--iterations', type=int, default=2000)

    parser.add_argument('--posterior_sample_spacing', type=int, default=50)

    parser.add_argument('--file_id', type=int, default=3)
    parser.add_argument('--file_index', type=int, default=2)
    parser.add_argument('--case_val', type=int, default=4)

    parser.add_argument('--x_dims', type=list, default=[4])

    parser.add_argument('--samples', type=int, default=10)

    parser.add_argument('--n_layers', type=int, default=1)
    # parser.add_argument('--dataset', type=str, required=True, choices=['boston'], default='boston')
    parser.add_argument('--ratio', type = float, default=0.5)
    parser.add_argument('--kernel_type', choices=['SquaredExponential', 'LinearK'], default='SquaredExponential')
    parser.add_argument('--kernel_train_flag', type = bool, default=True)
    parser.add_argument('--data_index', type=int, default=4)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--prior_type', choices=['determinantal', 'normal', 'strauss', 'uniform'], default='normal')


    args = parser.parse_args()

    file_path_seq = ['dryer/', 'drive/', 'gas_furnace/', 'actuator/', 'flutter/', 'ballbeam/']

    data_name = file_path_seq[args.file_index]
    ini_seq = glob.glob('Factnonlin_ini/factnonlin_initialized_10000_' + data_name[:-1] + '*.npz')

    print('####################')
    print('')
    print(data_name)
    print('')
    print('####################')

    main(data_name, ini_seq[args.file_id])
