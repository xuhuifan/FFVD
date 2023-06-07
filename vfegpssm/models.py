import numpy as np
from scipy.stats import bernoulli as np_bern
import matplotlib.pyplot as plt
from .dgp_model import DGPSSM
from scipy.stats import norm
from scipy.special import logsumexp
from .kernels_multi_output import SquaredExponential as BgpSE
from .kernels import LinearK as LinearKernel
from .likelihoods import Gaussian
import tensorflow as tf
import time

from datetime import datetime
PRIORS = ['uniform', 'normal', 'determinantal', 'strauss']
import logging
logger = logging.getLogger(__name__)
# tf.compat.v1.reset_default_graph()   # To clear the defined variables and operations of the previous cell

class Model(object):
    def __init__(self, prior_type, output_dim=None):
        class ARGS:
            num_inducing = 100
            iterations = 1000
            minibatch_size = 100
            window_size = 64
            num_posterior_samples = 20
            posterior_sample_spacing = 50
            full_cov = True
            n_layers = 1
            prior_type = None
            logdir = '/tmp/'
            x_dims = [1]

        self.ARGS = ARGS
        self.model = None
        self.output_dim = output_dim
        self.global_step = 0
        if prior_type in PRIORS:
            self.ARGS.prior_type = prior_type
        else:
            raise Exception("Invalid prior type")

    def _fit(self, Y_train, tensorboard_savepath, dataname, fileid, lik, X_train, Ystd, kernel_type, kernel_train_flag, Y_test = None, data_uu = None, **kwargs):
        if len(Y_train.shape) == 1:
            Y_train = Y_train[:, None]

        kerns = []
        if not self.model:
            for i in range(self.ARGS.n_layers):
                # output_dim = 196 if i >= 1 and self.ARGS.x_dims[i] > 700 else self.ARGS.x_dims[i]
                Z_dim = self.ARGS.control_inputs.shape[1]+self.ARGS.x_dims[-1]
                # control_inputs_dim = self.ARGS.control_inputs.shape[1]
                kernels_i = []
                if kernel_type == 'SquaredExponential':
                    # setting_variance = 1.0 if self.ARGS.variance is None else self.ARGS.variance
                    # setting_lengthscale = np.ones(Z_dim) if self.ARGS.lengthscales is None else self.ARGS.lengthscales
                    for kk in range(self.ARGS.x_dims[-1]):
                        kernels_i.append(BgpSE(Z_dim, ARD=True, variance=self.ARGS.variance[kk], lengthscales=self.ARGS.lengthscales[kk], kernel_optimization = self.ARGS.kernel_optimization))
                    kerns.append(kernels_i)
                elif kernel_type == 'LinearK':
                    setting_variance = 1.0 if self.ARGS.variance is None else self.ARGS.variance
                    kerns.append(LinearKernel(Z_dim, ARD=False, variance=setting_variance))

            mb_size = self.ARGS.minibatch_size if Y_train.shape[0] > self.ARGS.minibatch_size else Y_train.shape[0]

            self.model = DGPSSM(Y_train, self.ARGS.x_dims, self.ARGS.num_inducing, kerns, lik,
                             minibatch_size=mb_size,window_size=self.ARGS.window_size,
                             full_cov=self.ARGS.full_cov,prior_type=self.ARGS.prior_type, output_dim=self.output_dim,
                             QQ_chol = self.ARGS.QQ_chol, ZZ = self.ARGS.ZZ, variance = self.ARGS.variance,
                             lengthscales = self.ARGS.lengthscales, control_inputs = self.ARGS.control_inputs,
                             kernel_type = kernel_type, kernel_train_flag = kernel_train_flag, U_ini = self.ARGS.UU_ini, X_0_ini = self.ARGS.XX_0_ini,
                             X_train_ini = self.ARGS.x_initialization, X_PG = self.ARGS.X_PG, PG_particles = self.ARGS.PG_particles,
                             hyperparameter_sampling = self.ARGS.hyperparameter_sampling, kernel_optimization = self.ARGS.kernel_optimization, U_optimization = self.ARGS.U_optimization,
                             U_collapse = self.ARGS.U_collapse, Z_optimization = self.ARGS.Z_optimization, case_val = self.ARGS.case_val, **kwargs)
            print(self.model)


        print('')
        print('Direct optimization: ')
        for vv in tf.compat.v1.trainable_variables():
            print(vv)
        print('')
        print('SG-HMC optimized variables: ')
        for vv in self.model.vars:
            print(vv)
        print('')
        # test_jump = False

        self.nll_seq = []
        self.rmse_seq = []
        self.ll_seq = []
        self.running_time_seq = []

        write_tensorboard = False
        if write_tensorboard == True:
            tensorboard_savepath_1 = '/Users/xuhuifan/GPSSM_results/vfe_Sep/Sep_results_c4'
            writer = tf.compat.v1.summary.FileWriter(tensorboard_savepath_1 +'/'+dataname+'/'+fileid+'/', tf.compat.v1.Session().graph)
            # x_scalar = tf.constant(2.0, dtype=tf.float64)


            DD_summary = tf.compat.v1.summary.histogram('DD-histogram', self.model.likelihood.DD.value())
            CC_summary = tf.compat.v1.summary.histogram('CC-histogram', self.model.likelihood.CC.value())
            log_Rchol_summary = tf.compat.v1.summary.histogram('log-Rchols-histogram', self.model.likelihood.log_Rchols.value())

            log_Q_summary = tf.compat.v1.summary.histogram('log-Q-histogram', self.model.log_Q.value())

            kernel_0_log_variance_summary = tf.compat.v1.summary.scalar('kernel-0-log-variance-histogram', self.model.kernels[0][0].logvariance.value())
            kernel_1_log_variance_summary = tf.compat.v1.summary.scalar('kernel-1-log-variance-histogram', self.model.kernels[0][1].logvariance.value())
            kernel_2_log_variance_summary = tf.compat.v1.summary.scalar('kernel-2-log-variance-histogram', self.model.kernels[0][2].logvariance.value())
            kernel_3_log_variance_summary = tf.compat.v1.summary.scalar('kernel-3-log-variance-histogram', self.model.kernels[0][3].logvariance.value())

            nll_summary = tf.compat.v1.summary.scalar('marginal-ll', self.model.nll)


            kernel_0_log_lengthscales_summary = tf.compat.v1.summary.histogram('kernel-0-log-lengthscales-histogram', self.model.kernels[0][0].loglengthscales.value())
            kernel_1_log_lengthscales_summary = tf.compat.v1.summary.histogram('kernel-1-log-lengthscales-histogram', self.model.kernels[0][1].loglengthscales.value())
            kernel_2_log_lengthscales_summary = tf.compat.v1.summary.histogram('kernel-2-log-lengthscales-histogram', self.model.kernels[0][2].loglengthscales.value())
            kernel_3_log_lengthscales_summary = tf.compat.v1.summary.histogram('kernel-3-log-lengthscales-histogram', self.model.kernels[0][3].loglengthscales.value())

            # self.model.kernels[0][0].loglengthscales.value()

            xx_0_summary = tf.compat.v1.summary.histogram('x0-histogram', self.model.layers[-1].X[:, 0])
            xx_1_summary = tf.compat.v1.summary.histogram('x1-histogram', self.model.layers[-1].X[:, 1])
            xx_2_summary = tf.compat.v1.summary.histogram('x2-histogram', self.model.layers[-1].X[:, 2])
            xx_3_summary = tf.compat.v1.summary.histogram('x3-histogram', self.model.layers[-1].X[:, 3])

            u_0_summary = tf.compat.v1.summary.histogram('U0-histogram', self.model.layers[-1].U[:, 0])
            u_1_summary = tf.compat.v1.summary.histogram('U1-histogram', self.model.layers[-1].U[:, 1])
            u_2_summary = tf.compat.v1.summary.histogram('u2-histogram', self.model.layers[-1].U[:, 2])
            u_3_summary = tf.compat.v1.summary.histogram('U3-histogram', self.model.layers[-1].U[:, 3])

        # second_summary = tf.compat.v1.summary.scalar(name='DD', tensor=x_scalar)
        # init = tf.compat.v1.global_variables_initializer()
        batch_placeholder, adam_learning_rate = self.model.get_minibatch()
        feed_dict = {self.model.batch_placeholder: batch_placeholder,
                     self.model.adam_lr: adam_learning_rate}

        # try:
        _ = 0
        last_50_median_mll = 1000.
        current_time = time.time()
        while (_ < (2*self.ARGS.iterations)):
            # if (_ > self.ARGS.iterations):
            #     if (self.ll_seq[-1]>last_50_median_mll):
            #         break
            _ += 1
            self.global_step += 1
            # self.model.rewhiten()
            # cc_time = time.time()
            self.model.sghmc_step()
            # print('SG-HMC iteration elapsed time is: ', time.time() - cc_time)
            # current_time = time.time()

            # print(self.model.layers[-1].X)
            # if _ < 10:
            if self.ARGS.X_PG:
                # cc_time = time.time()
                self.model.gp_x_sampling()
                # print('PG iteration elapsed time is: ', time.time()-cc_time)
                # self.model.PG_for_X(self.ARGS.control_inputs, self.ARGS.PG_particles)
            # print('elaposed time of one PG step is: ', time.time() - current_time)

            if self.ARGS.prior_type == "determinantal":
                self.model.reset_Lm()

            # for _ in range(10):
            # cc_time = time.time()
            self.model.train_hypers() if hasattr(self.model, 'hyper_train_op') else None
            # print('Optimization iteration elapsed time is: ', time.time() - cc_time)

            # print('')
            # print('New iteration')
            # print(self.model.layers[-1].X)

            # if _ == 0:
            #     self.model.previous_mll = 0.

            # if _ > 100:
            #     last_50_median_mll = np.median(self.ll_seq[(-50):])

            if np.mod(_, 100)==0:
                print('Iteration: ', _)
            # if _ in [10, 100, 500, 1000]:
            #     self.running_time_seq.append(time.time()-current_time)
            #     print('elapsed time is: ', self.running_time_seq[-1])
            #
            #     marginal_ll = self.model.print_sample_performance(_, self.ARGS.U_collapse)
            #     self.ll_seq.append(marginal_ll)
            #
            #     nll, rmse = self.model.RMSE_calculate_per_iteration(self.ARGS.num_posterior_samples, self.ARGS.posterior_sample_spacing,
            #                       self.ARGS.control_inputs, test_len=self.ARGS.test_len,
            #                       sghmc_var_len=len(self.model.vars), U_collapse=self.ARGS.U_collapse,
            #                        Y_test = Y_test, Y_train_std=self.ARGS.Y_train_std, Y_train = Y_train)
            #
            #     self.nll_seq.append(nll)
            #     self.rmse_seq.append(rmse)
            #     current_time = time.time()


        test_time_flag = False
        if test_time_flag:
            save_test_time_path = 'test_time_2023/' + fileid +'_case_'+ str(self.ARGS.case_val) + '_test_time.npz'
            np.savez_compressed(save_test_time_path, run_time = self.running_time_seq, ll = self.ll_seq, nlpd = self.nll_seq,
                                rmse_seq = self.rmse_seq, case=self.ARGS.case_val)

            # train_loss(1 * tf.cast(self.global_step, tf.float64))  # here I just want to display 1*ep...
            # test_loss(2 * tf.cast(self.global_step, tf.float64))
            # tf.compat.v1.Session().run(init)
            # summary = tf.compat.v1.Session().run(first_summary, feed_dict=feed_dict)
        if write_tensorboard:
            summary_CC, summary_DD, summary_log_Rchols, summary_log_Q, summary_x0, summary_x1, \
                summary_x2, summary_x3, summary_u0, summary_u1, summary_u2, summary_u3 = self.model.session.run([CC_summary, DD_summary, log_Rchol_summary, log_Q_summary,
                                                                                 xx_0_summary, xx_1_summary, xx_2_summary,xx_3_summary,
                                                                                 u_0_summary, u_1_summary, u_2_summary, u_3_summary], feed_dict=feed_dict)
            summary_kernel_0_log_variance, summary_kernel_1_log_variance, \
                summary_kernel_2_log_variance, summary_kernel_3_log_variance, \
                summary_kernel_0_log_lengthscales, summary_kernel_1_log_lengthscales, \
                summary_kernel_2_log_lengthscales, summary_kernel_3_log_lengthscales, summary_nll = self.model.session.run([kernel_0_log_variance_summary, kernel_1_log_variance_summary,
                                                                   kernel_2_log_variance_summary, kernel_3_log_variance_summary,
                                                                   kernel_0_log_lengthscales_summary, kernel_1_log_lengthscales_summary,
                                                                   kernel_2_log_lengthscales_summary, kernel_3_log_lengthscales_summary, nll_summary], feed_dict = feed_dict)

            # summary_2 = self.model.session.run(second_summary, feed_dict=feed_dict)

            # summary_3 = self.model.session.run(CC_summary, feed_dict=feed_dict)

            # summary = self.model.session.run(self.model.likelihood.DD.value(), feed_dict=feed_dict)

            writer.add_summary(summary_CC, self.global_step)
            writer.add_summary(summary_DD, self.global_step)
            writer.add_summary(summary_log_Rchols, self.global_step)
            writer.add_summary(summary_log_Q, self.global_step)

            writer.add_summary(summary_x0, self.global_step)
            writer.add_summary(summary_x1, self.global_step)
            writer.add_summary(summary_x2, self.global_step)
            writer.add_summary(summary_x3, self.global_step)

            writer.add_summary(summary_u0, self.global_step)
            writer.add_summary(summary_u1, self.global_step)
            writer.add_summary(summary_u2, self.global_step)
            writer.add_summary(summary_u3, self.global_step)



            writer.add_summary(summary_nll, self.global_step)

            writer.add_summary(summary_kernel_0_log_variance, self.global_step)
            writer.add_summary(summary_kernel_1_log_variance, self.global_step)
            writer.add_summary(summary_kernel_2_log_variance, self.global_step)
            writer.add_summary(summary_kernel_3_log_variance, self.global_step)

            writer.add_summary(summary_kernel_0_log_lengthscales, self.global_step)
            writer.add_summary(summary_kernel_1_log_lengthscales, self.global_step)
            writer.add_summary(summary_kernel_2_log_lengthscales, self.global_step)
            writer.add_summary(summary_kernel_3_log_lengthscales, self.global_step)


            # writer.add_summary(summary_3, self.global_step)
            # writer.add_summary(first_summary, self.global_step)

            # with train_summary_writer.as_default():
            #     tf.summary.scalar('trloss', tf.constant(0.6, dtype=tf.float64), step=self.global_step)
            # with test_summary_writer.as_default():
            #     tf.summary.scalar('tsloss', tf.constant(0.7, dtype=tf.float64), step=self.global_step)
            # with train_summary_writer.as_default():
            #     tf.summary.scalar('likelihood-Rchol', self.model.likelihood.Rchols[0][0], step=self.global_step)
            #     tf.summary.scalar('likelihood-DD', self.model.likelihood.DD[0], step=self.global_step)

            # if _ % 10 == 0:
                # writer.add_scalar('optimisation/marginal_likelihood', marginal_ll*len(X), self.global_step)
                # print('TRAIN | iter = %6d      sample marginal LL = %5.2f       LL reg 1= %5.2f      LL reg 2= %5.2f' % (_, marginal_ll, marginal_ll_part_reg_1, marginal_ll_part_reg_2))
                # print('TRAIN | iter = %6d      sample marginal LL = %5.2f       LL reg 1= %5.2f      LL reg 2= %5.2f' % (_, marginal_ll, marginal_ll_part_reg_1, marginal_ll_part_reg_2))
                # Test with previous samples with Xtest and Ytest are both not None
                # if not (Xtest is None or Ytest is None or Ystd is None):
                #     ms, vs = self.model.predict_y(Xtest, len(self.model.window), posterior=False)
                #     logps = norm.logpdf(np.repeat(Ytest[None, :, :]*Ystd, len(self.model.window), axis=0), ms*Ystd, np.sqrt(vs)*Ystd)
                #     mnll = -np.mean(logsumexp(logps, axis=0) - np.log(len(self.model.window)))
                #     # writer.add_scalar('test/predictive_nloglikelihood', mnll, self.global_step)
                #     print('TEST  | iter = %6d       MNLL = %5.2f' % (_, mnll))
            # if self.ARGS.control_inputs.shape[0]>0:
            #     control_inputs_test = self.ARGS.control_inputs[(-self.ARGS.test_len):]
            # else:
            #     control_inputs_test = tf.convert_to_tensor(np.asarray([[]]*(self.ARGS.test_len+2)))
            # X_train_eval = []
            # X_test_eval = []
            #
            # # %tensorboard --logdir train_log_dir
            # self.model.collect_samples_formal(self.ARGS.num_posterior_samples, self.ARGS.posterior_sample_spacing,
            #                                    self.ARGS.control_inputs, Y_train = Y_train, Y_test = Y_test, save_path_file = tensorboard_savepath+'/'+dataname+fileid,
            #                                   test_len=self.ARGS.test_len, sghmc_var_len = len(self.model.vars), ll_seq = self.ll_seq, U_collapse = self.ARGS.U_collapse, synthetic_data_function_plot = self.ARGS.synthetic_data_test, data_uu = data_uu
            #                                   )

            # if len(self.model.vars)==0:
            #     self.model.collect_samples_concise(self.ARGS.num_posterior_samples, self.ARGS.posterior_sample_spacing, self.ARGS.control_inputs, test_len = self.ARGS.test_len, X_train_eval = X_train_eval, X_test_eval = X_test_eval)
            # else:
            #     self.model.collect_samples_full(self.ARGS.num_posterior_samples, self.ARGS.posterior_sample_spacing, self.ARGS.control_inputs, test_len = self.ARGS.test_len, X_train_eval = X_train_eval, X_test_eval = X_test_eval)
        # except KeyboardInterrupt:  # pragma: no cover
        #     self.model.collect_samples_concise(self.ARGS.num_posterior_samples, self.ARGS.posterior_sample_spacing, X_train_eval = X_train_eval, X_test_eval = X_test_eval)
        #     pass

        # plt.show()

    def _predict(self, Xs, S):
        ms, vs = [], []
        n = max(len(Xs) / 10000, 1) 
        for xs in np.array_split(Xs, n):
            m, v = self.model.predict_y(xs, S)
            ms.append(m)
            vs.append(v)

        return np.concatenate(ms, 1), np.concatenate(vs, 1) 


class RegressionModel(Model):
    def __init__(self, prior_type, output_dim=None):
        super().__init__(prior_type, output_dim)

    def fit(self, Y_train, Y_test = None, tensorboard_savepath = '', dataname = '', fileid = '', kernel_type = 'SquaredExponential', kernel_train_flag = True, likelihood_traning = True, X_train=None, X_test = None, Ystd=None, data_uu = None, **kwargs):
        lik = Gaussian(Y_train.shape[1], self.ARGS.x_dims[-1], CC = self.ARGS.CC, DD = self.ARGS.DD, RR_chol=self.ARGS.RR_chol, hyperparameter_sampling = self.ARGS.hyperparameter_sampling, likelihood_traning = likelihood_traning)
        # lik.trainable
        return self._fit(Y_train, tensorboard_savepath, dataname, fileid, lik, X_train, Ystd, kernel_type, kernel_train_flag, Y_test = Y_test, data_uu = data_uu, **kwargs)

    def predict(self, Xs):
        ms, vs = self._predict(Xs, self.ARGS.num_posterior_samples)
        m = np.average(ms, 0)
        v = np.average(vs + ms**2, 0) - m**2
        return m, v

    def calculate_density(self, Xs, Ys, ystd=1.):
        ms, vs = self._predict(Xs, self.ARGS.num_posterior_samples)
        logps = norm.logpdf(np.repeat(Ys[None, :, :]*ystd, self.ARGS.num_posterior_samples, axis=0), ms*ystd, np.sqrt(vs)*ystd)
        return logsumexp(logps, axis=0) - np.log(self.ARGS.num_posterior_samples)

    def sample(self, Xs, S):
        ms, vs = self._predict(Xs, S)
        return ms + vs**0.5 * np.random.randn(*ms.shape)

