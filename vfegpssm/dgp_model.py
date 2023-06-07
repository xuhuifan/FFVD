import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.cluster.vq import kmeans2

from .likelihoods import logdensity_norm, logdensity_norm_diag, logdensity_norm_diag_nonvec
from .base_model import BaseModel
from . import conditionals_multi_output
from .utils import get_rand

def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


class Strauss(object):

    def __init__(self, gamma=0.5, R=0.5):
        self.gamma = tf.constant(gamma, dtype=tf.float64)
        self.R = tf.constant(R, dtype=tf.float64)

    def _euclid_dist(self, X):
        Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
        dist = -2 * tf.matmul(X, X, transpose_b=True)
        dist += Xs + tf.matrix_transpose(Xs)
        return tf.sqrt(tf.maximum(dist, 1e-40))

    def _get_Sr(self, X):
        """
        Get the # elements in distance matrix dist that are < R
        """
        dist = self._euclid_dist(X)
        val = tf.where(dist <= self.R)
        Sr = tf.shape(val)[0] # number of points satisfying the constraint above
        dim = tf.shape(dist)[0]
        Sr = (Sr - dim)/2  # discounting diagonal and double counts
        return Sr

    def logp(self, X):
        return self._get_Sr(X) * tf.math.log(self.gamma)


class Layer(object):
    def __init__(self, ZZ, U_ini, X_0_ini, X_train_ini, kern, outputs, n_inducing, fixed_mean, x_dims_l, Y_len, full_cov,
                 prior_type="uniform", kernel_type = 'SquaredExponential', U_optimization = False,
                 U_collapse = False, Z_optimization = False, X_PG = False, case_val = 1): # uniform / normal / point
        self.inputs, self.outputs, self.kernel, self.kernel_type = kern[0].input_dim, outputs, kern, kernel_type
        self.M, self.fixed_mean = n_inducing, fixed_mean
        self.full_cov = full_cov
        self.prior_type = prior_type
        # syndata = np.load('result/synthetic_data_dim_1.npz')

        # self.X = tf.Variable(tfp.distributions.Uniform(low = tf.constant(-2, dtype = tf.float64), high = tf.constant(2, dtype = tf.float64)).sample([Y_len+1, x_dims_l]), dtype=tf.float64, trainable=False, name='XX')
        X_ini_val = np.zeros((Y_len + 1, x_dims_l))
        X_ini_val[0] = X_0_ini
        X_ini_val[1:] = X_train_ini

        # X_trainable = (not X_PG)

        if (X_PG) | (case_val == 7):
            # self.X = tf.convert_to_tensor(X_ini_val, dtype=tf.float64)
            self.X = tf.Variable(X_ini_val, dtype=tf.float64, trainable=False, name='XX')
        else:
            self.X = tf.Variable(X_ini_val, dtype=tf.float64, trainable=True, name='XX')

        self.U = tf.Variable(U_ini, dtype=tf.float64, trainable = U_optimization, name='U')
        self.Z = tf.Variable(ZZ, dtype= tf.float64, trainable = Z_optimization, name = 'Z')

        # self.X = tf.Variable(syndata['xx_seq'][:(Y_len+1)], dtype = tf.float64, trainable = False, name = 'XX')

        if prior_type == "strauss":
            self.pZ = Strauss(R=0.5)

        # if len(X) > 1000000:
        #     perm = np.random.permutation(100000)
        #     X = X[perm]
        # self.Z = tf.Variable(kmeans2(X, self.M, minit='points')[0], dtype=tf.float64, trainable=False, name='Z')
        # self.Z = tf.Variable(tf.identity(tf.random.shuffle(self.X[1:])[:self.M]), dtype=tf.float64, trainable=False, name='Z')

        # self.Z = tf.Variable(tfp.distributions.Uniform(low = tf.constant(-2, dtype = tf.float64), high = tf.constant(2, dtype = tf.float64)).sample([self.M, self.inputs]), dtype=tf.float64, trainable=False, name='Z')
        # ZZ_origin = syndata['ZZ']

        if self.inputs == outputs:
            self.mean = np.eye(self.inputs)
        elif self.inputs < self.outputs:
            self.mean = np.concatenate([np.eye(self.inputs), np.zeros((self.inputs, self.outputs - self.inputs))], axis=1)
        else:
            _, _, V = tf.linalg.svd(self.X, full_matrices=False)
            self.mean = tf.transpose(V[:self.outputs, :])

        # self.U = tf.Variable(np.zeros((self.M, self.outputs)), dtype=tf.float64, trainable=False, name='U')
        self.Lm = None


    def conditional(self, X):
        # Caching the covariance matrix from the sghmc steps gives a significant speedup. This is not being done here.
        mean, var, self.Lm = conditionals_multi_output.conditional(X, self.Z, self.kernel, self.U, white=True, full_cov=self.full_cov, return_Lm=True)
        
        # if self.fixed_mean:
        #     mean += tf.matmul(X, tf.cast(self.mean, tf.float64))
        return mean, var

    def prior_Z(self):
        if self.prior_type == "uniform":
            return 0.
        if self.prior_type == "normal":
            return -tf.reduce_sum(tf.square(self.Z)) / 2.0

        if self.prior_type == "strauss":
            return self.pZ.logp(self.Z)

        #if self.Lm is not None: # determinantal;
        if self.prior_type == "determinantal":
            self.Lm = tf.linalg.cholesky(self.kernel.K(self.Z) + tf.eye(self.M, dtype=tf.float64) * 1e-7)
            pZ = tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(self.Lm))))
            return pZ

        else: #
            raise Exception("Invalid prior type")

    def prior_hyper(self):
        if self.kernel_type == 'SquaredExponential':
            prior_hyper_val = 0.
            for kernel_k in self.kernel:
                prior_hyper_val += -tf.cast(tf.reduce_sum(tf.square(kernel_k.loglengthscales)) / 2.0, dtype = tf.float64) - tf.cast(tf.reduce_sum(tf.square(kernel_k.logvariance - tf.cast(tf.math.log(0.05), dtype = tf.float64))) / 2.0, dtype=tf.float64)
            return prior_hyper_val
        elif self.kernel_type == 'LinearK':
            return - tf.reduce_sum(tf.square(self.kernel.logvariance - np.log(0.05))) / 2.0

    def prior_U(self):
        choice = 1
        if choice == 1:
            return -0.5*tf.reduce_sum(tf.square(self.U))
        elif choice == 2:
            Lm = tf.linalg.cholesky(self.kernel.K(self.Z) + tf.eye(self.M, dtype=tf.float64) * 1e-7)
            A = tf.linalg.triangular_solve(Lm, self.U, lower=True)

            return -0.5*tf.reduce_sum(tf.square(A))-self.U.shape[0]*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lm)))

    def prior(self):
        return self.prior_U() + self.prior_hyper() + self.prior_Z()


    def __str__(self):
        str = [
            '============ GP Layer ',
            ' Input dim = %d' % self.inputs,
            ' Output dim = %d' % self.outputs,
            ' Num inducing = %d' % self.M,
            ' Prior on inducing positions = %s' % self.prior_type,
            '\n'.join(map(lambda s: ' |' + s, self.kernel.__str__().split('\n')))
        ]

        return '\n'.join(str)


class DGPSSM(BaseModel):
    def __init__(self, Y, x_dims, n_inducing, kernels, likelihood, minibatch_size, window_size, output_dim=None,
                 prior_type="uniform", full_cov=False, epsilon=0.01, mdecay=0.05, QQ_chol = None, ZZ = None, variance = None, lengthscales = None,
                 # prior_type="uniform", full_cov=False, epsilon=0.01, mdecay=0.05, QQ_chol = None, ZZ = None, variance = None, lengthscales = None,
                 control_inputs = None, kernel_type = 'SquaredExponential', kernel_train_flag = True, U_ini = None, X_0_ini = None, X_train_ini = None,
                 X_PG = False, PG_particles = 100, hyperparameter_sampling = False, kernel_optimization = False, U_optimization = False, U_collapse = False,
                 Z_optimization = False, case_val = 1):
        self.x_dims = x_dims
        self.n_inducing = n_inducing
        self.kernels = kernels
        self.likelihood = likelihood
        self.minibatch_size = minibatch_size
        self.window_size = window_size

        self.rand = lambda x: get_rand(x, full_cov)
        self.output_dim = output_dim or x_dims[-1]

        if hyperparameter_sampling:
            self.log_Q = tf.Variable(tf.ones(self.output_dim, dtype=tf.float64) * tf.math.log(tf.constant(0.1, dtype=tf.float64)),
                trainable=False, name='log-QQ') if QQ_chol is None else 2. * tf.math.log(QQ_chol)
        else:
            # self.log_Q = tf.Variable(tf.ones(self.output_dim, dtype = tf.float64)*tf.math.log(tf.constant(0.1, dtype = tf.float64)), trainable = True, name = 'log-QQ') if QQ_chol is None else 2.*tf.math.log(QQ_chol)
            if case_val != 7:
                self.log_Q = tf.Variable(tf.cast(2. * tf.math.log(QQ_chol), dtype = tf.float64),trainable=True, name='log-QQ')
            elif case_val == 7:
                self.log_Q = tf.Variable(tf.cast(2. * tf.math.log(QQ_chol), dtype=tf.float64), trainable=False, name='log-QQ')
        self.Q = tf.math.exp(self.log_Q)

        n_layers = len(kernels)


        self.layers = []
        # X_running = tf.identity(self.X[0])
        # X_running = self.X[0].copy()
        for l in range(n_layers):
            outputs = self.kernels[l+1][0].input_dim if l+1 < n_layers else self.output_dim #Y.shape[1]
            self.layers.append(Layer(ZZ, U_ini, X_0_ini, X_train_ini, self.kernels[l], outputs, n_inducing, fixed_mean=(l+1 < n_layers), x_dims_l=x_dims[l],
                                     Y_len = Y.shape[0], full_cov=full_cov if l+1<n_layers else False, prior_type=prior_type, kernel_type = kernel_type,
                                     U_optimization = U_optimization, U_collapse = U_collapse, Z_optimization = Z_optimization, X_PG = X_PG, case_val = case_val))
            # X_running = tf.matmul(X_running, self.layers[-1].mean)

        # variables = []
        # for l in self.layers:
        #     variables += [l.U, l.Z, l.kernel.loglengthscales, l.kernel.logvariance, l.X]
        self.X_N = self.layers[-1].X.shape[0]

        # current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        # train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        # test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
        # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        # self.layers[-1].kernel.logvariance.trainable = kernel_optimization
        # self.layers[-1].kernel.loglengthscales.trainable = kernel_optimization
        variables = []

        if case_val == 7:
            for l in self.layers:
                variables += [l.U]
                variables += [l.X]

        else:
            if not kernel_optimization:
                for l in self.layers:
                    if kernel_train_flag:
                        if kernel_type == 'SquaredExponential':
                            for kk_kernel in l.kernel:
                                variables += [kk_kernel.logvariance, kk_kernel.loglengthscales]
                            # variables += [l.kernel.logvariance, l.kernel.loglengthscales]
                        elif kernel_type == 'LinearK':
                            variables += [l.kernel.logvariance]

            if not U_optimization and not U_collapse:
                for l in self.layers:
                    variables += [l.U]
                    # variables += [l.U, l.X]
                    # variables += [l.Z]

            if not Z_optimization:
                for l in self.layers:
                    variables += [l.Z]

        if hyperparameter_sampling:
            variables += [self.log_Q]
            variables += [likelihood.CC, likelihood.DD, likelihood.log_Rchols]
        # variables += [self.CC, self.DD]

        super().__init__(Y, variables, minibatch_size, window_size)

        y_mean = self.likelihood.predict_mean(self.layers[-1].X[(self.batch_placeholder[0]+1):self.batch_placeholder[1]])

        self.log_likelihood = logdensity_norm_diag(self.Y[(self.batch_placeholder[0]):(self.batch_placeholder[1]-1)], y_mean, self.likelihood.Rchols[0])

        self.prior_x_0 = -tf.reduce_sum(tf.square(self.layers[-1].X[0])) / 2.0

        if control_inputs.shape[0]>0:
            control_inputs_batch = control_inputs[(self.batch_placeholder[0]):(self.batch_placeholder[1]-1)]
        else:
            control_inputs_batch = []

        hyperparameter_prior_val = self.hypaparameter_prior(likelihood.CC, likelihood.DD, likelihood.log_Rchols)

        batch_size = tf.cast(self.batch_placeholder[1] - self.batch_placeholder[0]-1, tf.float64)
        Y_N = tf.cast(self.X_N - 1, dtype=tf.float64)

        self.nll_log_likelihood = - (tf.reduce_sum(self.log_likelihood))/batch_size


        if U_collapse:
            if len(control_inputs_batch.shape) > 0:
                x_control_inputs_combine = tf.concat((self.layers[-1].X[(self.batch_placeholder[0]):(self.batch_placeholder[1]-1)], control_inputs_batch), axis=1)
            else:
                x_control_inputs_combine = self.layers[-1].X[(self.batch_placeholder[0]):self.batch_placeholder[1]][:-1]

            Lm_inverse_seq = conditionals_multi_output.kernel_pre_cal(self.layers[-1].Z, self.layers[-1].kernel)

            self.later_term1, self.later_term2, self.nll_reg_trace_inverse_Q_B = conditionals_multi_output.collapse_after_kernel_precalculation(Lm_inverse_seq,
                                                                                                x_control_inputs_combine,
                                                                                                self.layers[-1].X[(self.batch_placeholder[0]):(self.batch_placeholder[1])],
                                                                                                self.layers[-1].Z,
                                                                                                self.layers[-1].kernel,
                                                                                                self.Q, batch_size, Y_N)


            self.x_t_prior_Q = -tf.reduce_sum(logdensity_norm_diag_nonvec(self.layers[-1].X[(self.batch_placeholder[0]+1):(self.batch_placeholder[1])],\
                                             self.layers[-1].X[(self.batch_placeholder[0]):(self.batch_placeholder[1]-1)], (self.Q)**(0.5)))/batch_size

            self.nll_part_prior = - (self.layers[-1].prior_hyper() + self.layers[-1].prior_Z() + self.prior_x_0 + hyperparameter_prior_val) / Y_N

            self.nll = self.nll_part_prior + self.nll_log_likelihood + self.x_t_prior_Q + self.nll_reg_trace_inverse_Q_B + self.later_term1 + self.later_term2
        else:
            self.reg_trace_inverse_Q_B, self.reg_x_prior = self.regularizer(
                self.layers[-1].X[(self.batch_placeholder[0]):self.batch_placeholder[1]], control_inputs_batch)
            self.nll_reg_trace_inverse_Q_B = - (tf.reduce_sum(self.reg_trace_inverse_Q_B)) / batch_size

            self.x_t_prior_Q = - (tf.reduce_sum(self.reg_x_prior)) / batch_size

            self.nll_part_prior = - (self.layers[-1].prior() + self.prior_x_0 + hyperparameter_prior_val) / Y_N
            self.nll = self.nll_part_prior + self.nll_log_likelihood + self.x_t_prior_Q + self.nll_reg_trace_inverse_Q_B

        # if X_PG:
        #     self.PG_for_X_algorithm = self.PG_for_X_speedup(control_inputs, PG_particles)

        self.generate_update_step(self.nll, epsilon, mdecay)
        self.adam = tf.compat.v1.train.AdamOptimizer(self.adam_lr)
        try:
            self.hyper_train_op = self.adam.minimize(self.nll)
        except ValueError:
            pass

        self.pg_x_sampling_op = self.PG_for_X_speedup(control_inputs, PG_particles)



        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config=config)


        # set_seed()
        init_op = tf.compat.v1.global_variables_initializer()
        try:
            # self.session.run(init_op, feed_dict=self.likelihood.initializable_feeds )
            self.session.run(init_op)
        except:
            self.session.run(init_op)

    def hypaparameter_prior(self, CC, DD, log_Rchols):
        self.log_Q_variance = 1.0
        log_q_prior = -tf.reduce_sum(tf.square(self.log_Q)) / tf.cast(self.log_Q_variance*2.0, dtype=tf.float64)
        # log_q_prior = 0.
        C_prior = -tf.reduce_sum(tf.square(CC)) / 2.0
        D_prior = -tf.reduce_sum(tf.square(DD)) / 2.0
        log_Rchols_prior = -tf.reduce_sum(tf.square(log_Rchols)) / 2.0

        return log_q_prior + C_prior + D_prior + log_Rchols_prior


    def regularizer(self, X_batch, control_inputs_batch):
        # x_control_inputs_combine = np.hstack((X_batch[:-1], control_inputs_batch[:-1]))
        if len(control_inputs_batch.shape)>0:
            x_control_inputs_combine = tf.concat((X_batch[:-1], control_inputs_batch), axis=1)
        else:
            x_control_inputs_combine = X_batch[:-1]
        mean_reg, var_reg = conditionals_multi_output.conditional(x_control_inputs_combine, self.layers[-1].Z, self.kernels[-1], self.layers[-1].U, white=True, return_Lm=False)

        # add the mean function, which is the identity function (X_batch[:-1) here
        mean_reg = mean_reg + X_batch[:-1]

        reg_trace_inverse_Q_B = -0.5*tf.reduce_sum(((self.Q[None, :])**(-1))*var_reg, axis=1)
        # reg2 = logdensity_norm(X_batch[1:], mean_reg, tf.linalg.diag(self.Q**(0.5)))

        reg_x_prior = logdensity_norm_diag(X_batch[1:], mean_reg, self.Q**(0.5))

        # batch_placeholder = self.get_minibatch()
        # feed_dict = {self.batch_placeholder: batch_placeholder}
        # reg1.eval(session=tf.compat.v1.Session())
        # reg2.eval(session=tf.compat.v1.Session())


        return reg_trace_inverse_Q_B, reg_x_prior


    def propagate(self, X):
        Fs = [X, ]
        Fmeans, Fvars = [], []

        for l, layer in enumerate(self.layers):
            mean, var = layer.conditional(Fs[-1])
            # eps = tf.random_normal(tf.shape(mean), dtype=tf.float64)
            # F = mean + eps * tf.sqrt(var)
            if l+1 < len(self.layers):
                F = self.rand([mean, var])
            else:
                F = get_rand([mean, var], False)
                
            Fs.append(F)
            Fmeans.append(mean)
            Fvars.append(var)

        return Fs[1:], Fmeans, Fvars

    def reset_Lm(self):
        for layer in self.layers:
            layer.Lm = None


    # def predict_y(self, X, S, posterior=True):
    #     # assert S <= len(self.posterior_samples)
    #     ms, vs = [], []
    #     for i in range(S):
    #         feed_dict = {self.X_placeholder: X}
    #         feed_dict.update(self.posterior_samples[i]) if posterior else feed_dict.update(self.window[-(i+1)])
    #         m, v = self.session.run((self.y_mean, self.y_var), feed_dict=feed_dict)
    #         ms.append(m)
    #         vs.append(v)
    #     return np.stack(ms, 0), np.stack(vs, 0)
    #
    # def predict_f(self, xx):
    #     conditionals.conditional()

    def predict_y_samples(self, test_Time):
        # assert S <= len(self.posterior_samples)
        batch_placeholder, adam_learning_rate = self.get_minibatch()
        feed_dict = {self.batch_placeholder: batch_placeholder,
                     self.adam_lr: adam_learning_rate}
        # self.xx_test_seq = []
        # self.xx_test_seq.append(self.layers[-1].X[-1])
        x_previous = self.layers[-1].X[-2][:, None]
        yy_test = []
        for test_i in range(test_Time):
            # feed_dict = {self.X_placeholder: X}
            # feed_dict.update(self.posterior_samples[i])
            xx_test_i, _, _ = conditionals_multi_output.conditional(x_previous, self.layers[-1].Z, self.layers[-1].kernel, self.layers[-1].U, white=True,full_cov=self.layers[-1].full_cov, return_Lm=True)
            yy_test_i = tf.matmul(xx_test_i, self.likelihood.CC)+self.likelihood.DD
            x_previous = xx_test_i
            yy_test_i_val = self.session.run(yy_test_i, feed_dict=feed_dict)
            yy_test.append(yy_test_i_val[0][0])
        return yy_test

    def __str__(self):
        str = [
            '================= DGP',
            ' Input dim = %d' % self.layers[0].inputs,
            ' Output dim = %d' % self.layers[-1].outputs,
            ' Depth = %d' % len(self.layers)
        ]
        return '\n'.join(str + ['\n'.join(map(lambda s: ' |' + s, l.__str__().split('\n'))) for l in self.layers])