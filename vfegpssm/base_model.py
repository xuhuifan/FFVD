import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .likelihoods import logdensity_norm
import matplotlib.pyplot as plt
from . import conditionals_multi_output
from .kernels import SquaredExponential as BgpSE
import copy
from scipy.stats import norm

class BaseModel(object):
    def __init__(self, Y, vars, minibatch_size, window_size):
        # self.X_placeholder = tf.compat.v1.placeholder(tf.float64, shape=[None, self.x_dims[-1]])
        # self.x_dims = x_dims
        # self.Y_placeholder = tf.compat.v1.placeholder(tf.float64, shape=[None, Y.shape[1]])
        self.batch_placeholder = tf.compat.v1.placeholder(tf.int64, shape = [2])
        self.adam_lr = tf.compat.v1.placeholder(tf.float64, shape = [])
        self.Y = tf.convert_to_tensor(Y, dtype = tf.float64)
        # self.N = Y.shape[0]
        self.vars = vars
        self.minibatch_size = min(minibatch_size, self.X_N)
        self.data_iter = 0
        self.window_size = window_size
        self.window = []
        self.posterior_samples = []
        self.sample_op = None
        self.burn_in_op = None

    def PG_for_X(self, control_inputs, PG_particles):
        # particles_X = tf.zeros((PG_particles, self.X_N, self.layers[-1].X.shape[1]))
        # particles_X = []
        # for particle_i in range(PG_particles-1):
        #     particles_X.append([tfp.distributions.Normal(loc = tf.zeros((1, self.layers[-1].X.shape[1]), dtype = tf.float64), scale = 0.1).sample()])
        particles_X = tf.unstack(tfp.distributions.Normal(loc = tf.zeros((PG_particles-1, 1, self.layers[-1].X.shape[1]), dtype = tf.float64), scale = 1.0).sample())
        particles_X.append(self.layers[-1].X[0][None, :])
        # particles_X_condition = self.layers[-1].X
        Lm_inverse_seq = conditionals_multi_output.kernel_pre_cal(self.layers[-1].Z, self.layers[-1].kernel)

        for tt in range(self.X_N-1):
            # weight_log_tt = tf.zeros((PG_particles))
            # for partile_i in range(PG_particles - 1):
                # x_t = particles_X[partile_i][-1][None, :]
                # if (control_inputs.shape[0]) > 0:
                #     x_control_inputs_combine = tf.concat((x_t, control_inputs[tt][None, :]),axis=1)
                # else:
                #     x_control_inputs_combine = x_t

            x_t = tf.stack(particles_X)[:(-1), -1]
            x_control_inputs_combine = tf.concat((x_t, control_inputs[tt]*tf.ones((PG_particles-1, 1), dtype = tf.float64)), axis=1)

            # f_t_mu, f_t_var = conditionals_multi_output.conditional(x_control_inputs_combine, self.layers[-1].Z,self.layers[-1].kernel,self.layers[-1].U, white=True,full_cov=self.layers[-1].full_cov, return_Lm=False)
            f_t_mu, f_t_var = conditionals_multi_output.conditional_after_kernel_precalculation(Lm_inverse_seq, x_control_inputs_combine, self.layers[-1].Z,
                                                                                                self.layers[-1].kernel, self.layers[-1].U, white=True,
                                                                                                full_cov=self.layers[-1].full_cov, return_Lm=False)
            f_t_mu += x_t

            x_tplus1 = f_t_mu + tf.random.normal(f_t_mu.shape, dtype=tf.float64) * tf.math.sqrt(f_t_var + self.Q)

            particles_X[:-1] = tf.unstack(tf.concat((tf.stack(particles_X[:-1]), x_tplus1[:, None]), axis=1))

            y_t_mu = self.likelihood.predict_mean(x_tplus1)
            weight_log_tt = tf.unstack(logdensity_norm(self.Y[tt], y_t_mu,self.likelihood.Rchols))

            y_t_mu = self.likelihood.predict_mean(self.layers[-1].X[tt+1][None, :])
            weight_log_tt.append(logdensity_norm(self.Y[tt], y_t_mu,self.likelihood.Rchols)[0])
            particles_X[-1]=(self.layers[-1].X[:(tt+2)])
            # do re-sampling
            if tt<(self.X_N-2):
                indexes_tt = tfp.distributions.Categorical(logits = tf.stack(weight_log_tt)).sample(PG_particles-1)
                particles_X = tf.unstack(tf.gather(particles_X, indexes_tt))
                particles_X.append(self.layers[-1].X[:(tt+2)])
                # particles_X[:PG_particles] = particles_X[indexes_tt]
            else:
                indexes_tt = tfp.distributions.Categorical(logits = tf.stack(weight_log_tt)).sample()
                self.layers[-1].X = tf.gather(particles_X, indexes_tt)


    def PG_for_X_speedup(self, control_inputs, PG_particles):
        particles_X = tfp.distributions.Normal(loc = tf.zeros(self.layers[-1].X.shape[1], dtype = tf.float64), scale = 1.0).sample(PG_particles-1)

        Lm_inverse_seq = conditionals_multi_output.kernel_pre_cal(self.layers[-1].Z, self.layers[-1].kernel)

        particles_x_seq = tf.TensorArray(size=self.X_N, dtype=tf.float64, clear_after_read=False,
                                   infer_shape=False, element_shape=(PG_particles-1, self.layers[-1].X.shape[1]))

        particles_x_seq = particles_x_seq.write(0, particles_X)

        def _loop_body(*args):
            tt, particles_x_seq = args[:2]

            x_t = particles_x_seq.read(tt)

            x_control_inputs_combine = tf.concat((x_t, control_inputs[tt] * tf.ones((PG_particles - 1, 1), dtype=tf.float64)), axis=1)

            f_t_mu, f_t_var = conditionals_multi_output.conditional_after_kernel_precalculation(Lm_inverse_seq, x_control_inputs_combine, self.layers[-1].Z,
                                                                                                self.layers[-1].kernel, self.layers[-1].U, white=True,
                                                                                                full_cov=self.layers[-1].full_cov, return_Lm=False)

            f_t_mu += x_t

            x_tplus1 = f_t_mu + tf.random.normal(f_t_mu.shape, dtype=tf.float64) * tf.math.sqrt(f_t_var + self.Q)

            particles_X_new = tf.unstack(x_tplus1[:, :])

            y_t_mu = self.likelihood.predict_mean(x_tplus1)
            weight_log_tt = tf.unstack(logdensity_norm(self.Y[tt], y_t_mu,self.likelihood.Rchols))

            y_t_mu = self.likelihood.predict_mean(self.layers[-1].X[tt+1][None, :])
            weight_log_tt.append(logdensity_norm(self.Y[tt], y_t_mu,self.likelihood.Rchols)[0])

            particles_X_new.append(self.layers[-1].X[tt+1])

            indexes_tt = tfp.distributions.Categorical(logits = tf.stack(weight_log_tt)).sample(PG_particles-1)
            particles_X_new = tf.gather(particles_X_new, indexes_tt)

            particles_x_seq.write(tt+1, particles_X_new)

            rest_values = [tt +1, particles_x_seq]

            # del particles_x_seq, control_inputs, particles_X_new, x_t, x_tplus1, y_t_mu, x_control_inputs_combine, f_t_var, f_t_mu, weight_log_tt

            return rest_values

        _loop_vars = [0, particles_x_seq]

        rest_values = tf.compat.v1.while_loop(
            cond=lambda tt, *args: tt < (self.X_N - 1),
            body=_loop_body,
            loop_vars=_loop_vars,
            back_prop=False,
            parallel_iterations=10)

        resampled_X = rest_values[1].stack()

        final_index = np.random.choice(PG_particles)
        if final_index <(PG_particles-1):
            tf.compat.v1.assign(self.layers[-1].X, resampled_X[:, final_index])
        return tf.ones(1)

        # del resampled_X, rest_values, particles_x_seq, particles_X, _loop_vars, Lm_inverse_seq


    def generate_update_step(self, nll, epsilon, mdecay):
        self.epsilon = epsilon
        burn_in_updates = []
        sample_updates = []

        grads = tf.gradients(nll, self.vars)

        for theta, grad in zip(self.vars, grads):
            xi = tf.Variable(tf.ones_like(theta), dtype=tf.float64, trainable=False)
            g = tf.Variable(tf.ones_like(theta), dtype=tf.float64, trainable=False)
            g2 = tf.Variable(tf.ones_like(theta), dtype=tf.float64, trainable=False)
            p = tf.Variable(tf.zeros_like(theta), dtype=tf.float64, trainable=False)

            r_t = 1. / (xi + 1.)
            g_t = (1. - r_t) * g + r_t * grad
            g2_t = (1. - r_t) * g2 + r_t * grad ** 2
            xi_t = 1. + xi * (1. - g * g / (g2 + 1e-16))
            Minv = 1. / (tf.sqrt(g2 + 1e-16) + 1e-16)

            burn_in_updates.append((xi, xi_t))
            burn_in_updates.append((g, g_t))
            burn_in_updates.append((g2, g2_t))

            epsilon_scaled = epsilon / tf.sqrt(tf.cast(self.X_N, tf.float64))
            # epsilon_scaled = 1.

            noise_scale = 2. * epsilon_scaled ** 2 * mdecay * Minv
            sigma = tf.sqrt(tf.maximum(noise_scale, 1e-16))
            sample_t = tf.random.normal(tf.shape(theta), dtype=tf.float64) * sigma
            p_t = p - epsilon ** 2 * Minv * grad - mdecay * p + sample_t
            theta_t = theta + p_t

            sample_updates.append((theta, theta_t))
            sample_updates.append((p, p_t))

        self.sample_op = [tf.compat.v1.assign(var, var_t) for var, var_t in sample_updates]
        self.burn_in_op = [tf.compat.v1.assign(var, var_t) for var, var_t in burn_in_updates + sample_updates]

        # xx = np.linspace(-3, 3, 1000)[:, None]
        # self.f_mu, self.f_var = self.layers[-1].conditional(tf.cast(xx, tf.float64))

    def reset(self, X, Y):
        self.X, self.Y, self.X_N = X, Y, X.shape[0]
        self.data_iter = 0

    def get_minibatch(self, global_step = 1):
        # decayed_learning_rate = 0.01 * (0.95**(global_step/1000))
        decayed_learning_rate = 0.003 * (0.95**(global_step/1000))
        # position_initial = tf.random.categorical(tf.zeros((1, self.Y.shape[0])), 1)
        # position_terminal = (position_initial + self.minibatch_size) if (position_initial + self.minibatch_size)<(self.Y.shape[0]+1) else (self.Y.shape[0]+1)
        # return [position_initial, position_terminal], decayed_learning_rate
        return [0, self.X_N], decayed_learning_rate


    def collect_samples_formal(self, num, spacing, control_inputs, test_len, sghmc_var_len = 0, U_collapse = False, Y_test = None, Y_train_std = 1., save_path_file = None, Y_train = None, case = 'C1', ll_seq = [0.], running_time_seq = [0.], PG_num = None, synthetic_data_function_plot = False, data_uu = None):

        batch_placeholder, adam_learning_rate = self.get_minibatch()
        feed_dict = {self.batch_placeholder: batch_placeholder,
                     self.adam_lr: adam_learning_rate}

        self.fit_x = self.session.run((self.layers[-1].X), feed_dict=feed_dict)

        if sghmc_var_len == 0:
            # pre-calculate the Cholesky decomposition of the kernel matrix
            Lm_inverse_seq = conditionals_multi_output.kernel_pre_cal(self.layers[-1].Z, self.layers[-1].kernel)

        pre_index = 1
        prediction_length = test_len + pre_index - 1

        predict_x_whole = []
        predict_x_combine_whole = []
        predict_x_var_whole = []

        # mc_posterior_samples = [[], []]
        mc_posterior_samples = [[]]*sghmc_var_len

        xx_test_coor_whole = []
        ff_test_coor_mu_whole = []
        ff_test_coor_var_whole = []

        for num_i in range(num):

            if sghmc_var_len > 0:
                # run spacing-number of iterations for SG-HMC samplers
                for j in range(spacing):
                    batch_placeholder, adam_learning_rate = self.get_minibatch()
                    feed_dict = {self.batch_placeholder: batch_placeholder,
                                 self.adam_lr: adam_learning_rate}
                    self.session.run((self.sample_op), feed_dict=feed_dict)

                # pre-calculate the Cholesky decomposition of the kernel matrix
                Lm_inverse_seq = conditionals_multi_output.kernel_pre_cal(self.layers[-1].Z, self.layers[-1].kernel)

            # set the initial value of latent state for prediction
            x_t = self.layers[-1].X[-pre_index][None, :]

            for len_i in range(sghmc_var_len):
                mc_posterior_samples[len_i].append(self.session.run(self.vars[len_i], feed_dict=feed_dict))

            if U_collapse:
                if len(control_inputs.shape) > 0:
                    x_control_inputs_combine = tf.concat((self.layers[-1].X[:(self.X_N-1)],control_inputs[:(self.X_N-1)]), axis=1)
                else:
                    x_control_inputs_combine = self.layers[-1].X[:-1]

                # U_val, U_variance_cholesky = conditionals_multi_output.collapse_u_mean_after_kernel_precalculation(Lm_inverse_seq, x_control_inputs_combine, self.layers[-1].X, self.layers[-1].Z, self.layers[-1].kernel, self.Q)
                # U_val = U_val[0]

                U_val, U_val_variance_cholesky = conditionals_multi_output.collapse_u_mean_after_kernel_precalculation(Lm_inverse_seq, x_control_inputs_combine, self.layers[-1].X, self.layers[-1].Z, self.layers[-1].kernel, self.Q)
                U_val = U_val[0]
                U_variance_cholesky = U_val_variance_cholesky
            else:
                U_val = self.layers[-1].U
                U_variance_cholesky = None


            #########################

            #########################

            if synthetic_data_function_plot:
                xx_test_coor = tf.cast(np.linspace(-3, 3, 1000), dtype=tf.float64)[:, None]
                # ff_mu, ff_var, _ = conditionals_multi_output.conditional(xx_test_coor, ZZ, kerns[0], UU, white=True, full_cov=False,
                #                                             return_Lm=True)
                ff_test_coor_mu, ff_test_coor_var = conditionals_multi_output.conditional_after_kernel_precalculation(Lm_inverse_seq, xx_test_coor, self.layers[-1].Z, self.layers[-1].kernel,
                                                           U_val, white=True,full_cov=self.layers[-1].full_cov, q_sqrt=U_variance_cholesky, return_Lm=False)

                xx_test_coor = self.session.run(xx_test_coor, feed_dict=feed_dict)
                ff_test_coor_mu = self.session.run(ff_test_coor_mu, feed_dict=feed_dict)
                ff_test_coor_var = self.session.run(ff_test_coor_var, feed_dict=feed_dict)

                xx_test_coor_whole.append(xx_test_coor)
                ff_test_coor_mu_whole.append(ff_test_coor_mu)
                ff_test_coor_var_whole.append(ff_test_coor_var)


            #########################

            #########################

            predict_x_val = []
            # predict_x_combine = []
            predcit_x_val_var = []
            for test_i in range(prediction_length):
                if (control_inputs.shape[1]) > 0:
                    # x_control_inputs_combine = tf.concat((x_t, control_inputs[test_i - (prediction_length)][None, :]), axis=1)
                    x_control_inputs_combine = tf.concat((x_t, control_inputs[test_i +Y_train.shape[0]][None, :]), axis=1)
                else:
                    x_control_inputs_combine = (x_t)

                # x_ci_val = np.asarray(self.session.run(x_control_inputs_combine, feed_dict=feed_dict))
                # predict_x_combine.append(x_ci_val)

                f_t_mu, f_t_var = conditionals_multi_output.conditional_after_kernel_precalculation(Lm_inverse_seq, x_control_inputs_combine, self.layers[-1].Z, self.layers[-1].kernel,
                                                           U_val, white=True,full_cov=self.layers[-1].full_cov, q_sqrt=U_variance_cholesky, return_Lm=False)

                # add the mean function, which is the identity function (x_previous) here
                f_t_mu += x_t
                # x_tplus1 = f_t_mu
                x_tplus1 = f_t_mu + tf.random.normal(f_t_mu.shape, dtype=tf.float64) * tf.math.sqrt(f_t_var + self.Q)
                # aa = self.session.run((f_t_mu), feed_dict=feed_dict)
                # bb = self.session.run((f_t_var), feed_dict=feed_dict)

                # x_tplus1_val = np.asarray(self.session.run(x_tplus1, feed_dict=feed_dict))

                predict_x_val.append(x_tplus1)
                predcit_x_val_var.append(f_t_var+self.Q)
                x_t = x_tplus1

            predict_x_whole.append(predict_x_val)
            # predict_x_combine_whole.append(predict_x_combine)
            predict_x_var_whole.append(predcit_x_val_var)

            print('')
            print('Posterior: ', num_i, '-th samples obtained.')

        # np.savez_compressed(save_path_file + '_function_fiting.npz', ff_test_coor_mu=ff_test_coor_mu_whole,
        #                     xx_test_coor=xx_test_coor_whole, ff_test_coor_var=ff_test_coor_var_whole)

        predict_x_whole = tf.stack(predict_x_whole)

        predict_x_var_whole = tf.stack(predict_x_var_whole)

        predict_x_whole = self.session.run(predict_x_whole, feed_dict=feed_dict)[:, :, 0]
        predict_x_var_whole = self.session.run(predict_x_var_whole, feed_dict=feed_dict)[:, :, 0]



        CC_val = self.session.run((self.likelihood.CC), feed_dict=feed_dict)
        DD_val = self.session.run((self.likelihood.DD), feed_dict=feed_dict)

        fit_x_value = np.asarray(self.fit_x)[1:]

        # self.predict_y = np.zeros((self.predict_x_whole.shape[1], CC_val.shape[1]))
        log_R_cholesky = self.session.run((self.likelihood.log_Rchols), feed_dict=feed_dict)


        self.predict_y = (np.mean(np.einsum('ijk,kl->ijl', predict_x_whole, CC_val), axis=0)+DD_val[None, :]).reshape((-1))
        self.predict_y_var = (np.mean(np.einsum('ijk,kl->ijl', predict_x_var_whole, CC_val**2), axis=0)).reshape((-1))+np.exp(2*log_R_cholesky)

        self.fit_y = (np.matmul(fit_x_value, CC_val) + DD_val).reshape((-1))

        Y_test_30 = Y_test[:30].reshape((-1))
        Y_predict_30 = self.predict_y[:30]
        RMSE_val = np.sqrt(np.mean((Y_test_30-Y_predict_30)**2))*Y_train_std

        self.RMSE_val = RMSE_val
        print('RMSE: ', RMSE_val)


        # step_by_step_check = False
        # if step_by_step_check:
        #     index_val = 6
        #     # predict_x_var_whole[0][index_val, 0]
        #     # predict_x_var_whole[0][index_val+1, 0]
        #
        #     # predict_x_combine_whole = np.asarray(self.session.run(tf.stack(predict_x_combine_whole), feed_dict = feed_dict))
        #     # predict_x_combine_whole = np.asarray(predict_x_combine_whole)[:, :, 0]
        #     # predict_x_whole = np.asarray(predict_x_whole)[:, :, 0]
        #
        #     # print('1-dimensional x combine: ', predict_x_combine_whole[0][:, 0])
        #
        #     x_control_inputs_combine = tf.concat((predict_x_val[index_val-1], control_inputs[index_val +Y_train.shape[0]][None, :]), axis=1)
        #
        #     f_t_mu_i, f_t_var_i = conditionals_multi_output.conditional_after_kernel_precalculation(Lm_inverse_seq,
        #                                                                                         x_control_inputs_combine,
        #                                                                                         self.layers[-1].Z,
        #                                                                                         self.layers[-1].kernel,
        #                                                                                         U_val, white=True,
        #                                                                                         full_cov=self.layers[
        #                                                                                             -1].full_cov,
        #                                                                                         q_sqrt=U_variance_cholesky,
        #                                                                                         return_Lm=False)
        #
        #     print('val is: ', self.session.run(f_t_mu_i, feed_dict = feed_dict))
        #
        #
        #     x_control_inputs_combine = tf.concat((predict_x_val[index_val-2], control_inputs[index_val-1 +Y_train.shape[0]][None, :]), axis=1)
        #     f_t_mu_i, f_t_var_i = conditionals_multi_output.conditional_after_kernel_precalculation(Lm_inverse_seq,
        #                                                                                         x_control_inputs_combine,
        #                                                                                         self.layers[-1].Z,
        #                                                                                         self.layers[-1].kernel,
        #                                                                                         U_val, white=True,
        #                                                                                         full_cov=self.layers[
        #                                                                                             -1].full_cov,
        #                                                                                         q_sqrt=U_variance_cholesky,
        #                                                                                         return_Lm=False)
        #
        #     print('val is: ', self.session.run(f_t_mu_i, feed_dict = feed_dict))
        #
        #     U_val_val = self.session.run(U_val, feed_dict = feed_dict)
        #
        #     plt.plot(U_val_val[:, 0], label = 'dim 0')
        #     plt.plot(U_val_val[:, 1], label = 'dim 1')
        #     plt.plot(U_val_val[:, 2], label = 'dim 2')
        #     plt.plot(U_val_val[:, 3], label = 'dim 3')
        #     plt.legend()
        #     plt.show()
        #
        #
        #
        #
        #     # print('1-dimensional x combine: ', predict_x_combine_whole[0][:, 0])
        #
        #     # print(predict_x_var_whole[0][:, 0])
        #
        #     # print('value of normal x is: ', predict_x_whole[0][index_val-2, :])
        #     print('value of normal x is: ', predict_x_whole[0][index_val-2, :])
        #
        #     print('value of normal x is: ', predict_x_whole[0][index_val-1, :])
        #
        #     print('value of AB-normal x is: ', predict_x_whole[0][index_val, :])
        #
        #
        #
        #     Q_val = self.session.run(self.Q, feed_dict = feed_dict)
        #
        #     kernel_0_variance = np.exp(self.session.run(self.layers[-1].kernel[0].logvariance, feed_dict = feed_dict))
        #     kernel_1_variance = np.exp(self.session.run(self.layers[-1].kernel[1].logvariance, feed_dict = feed_dict))
        #     kernel_2_variance = np.exp(self.session.run(self.layers[-1].kernel[2].logvariance, feed_dict = feed_dict))
        #     kernel_3_variance = np.exp(self.session.run(self.layers[-1].kernel[3].logvariance, feed_dict = feed_dict))
        #     kernel_0_lengthscales = np.exp(self.session.run(self.layers[-1].kernel[0].loglengthscales, feed_dict = feed_dict))
        #
        #     z_val = np.exp(self.session.run(self.layers[-1].Z, feed_dict = feed_dict))
        #     control_inputs_val = np.exp(self.session.run(control_inputs, feed_dict = feed_dict))


        # print(predict_x_whole[0][:, 0])
        #
        # print(predict_x_var_whole[0][:, 0])


        # if step_by_step_check:
        #
        #     plt.subplot(2,1,1)
        #     plt.plot(predict_x_whole[0][:, 0])
        #
        #     plt.subplot(2,1,2)
        #     plt.plot(predict_x_var_whole[0][:, 0])
        #     plt.show()
        #
        #
        #     state_check_index = 21
        #     plt.subplot(2,1,1)
        #     x_coor1_previous = predict_x_var_whole[0][state_check_index-1][:2]
        #     x_coor1 = predict_x_var_whole[0][state_check_index][:2]
        #     plt.scatter(x_coor1[0], x_coor1[1], c = 'r', label = 'states')
        #     plt.scatter(x_coor1_previous[0], x_coor1_previous[1], c = 'b', label = 'states-1')
        #     plt.scatter(z_val[:, 0], z_val[:, 1], c= 'g', label = 'inducing inputs')
        #     plt.legend()
        #     plt.title('Dimensions - (1, 2)')
        #     plt.subplot(2,1,2)
        #     x_coor2 = predict_x_var_whole[0][state_check_index][2:]
        #     x_coor2_previous = predict_x_var_whole[0][state_check_index-1][2:]
        #     plt.scatter(x_coor2[0], x_coor2[1], c = 'r')
        #     plt.scatter(x_coor2_previous[0], x_coor2_previous[1], c = 'b', label = 'states-1')
        #     plt.scatter(z_val[:, 3], z_val[:, 4], c= 'g')
        #     plt.title('Dimensions - (3, 4)')
        #
        #     plt.tight_layout()
        #     plt.show()
        #
        #     plt.plot(control_inputs_val)
        #     plt.title('Control inputs unique values: '+ np.array2string(np.unique(control_inputs_val)))
        #     plt.show()
        #
        # plt.subplot(2,1,1)
        # plt.plot(Y_train.reshape((-1)), label = 'ground-truth')
        # plt.plot(self.fit_y.reshape((-1)), label = 'fitted')
        # plt.legend()
        #
        # plt.subplot(2,1,2)
        # plt.plot(Y_test.reshape((-1)), label = 'ground-truth')
        # plt.plot(self.predict_y.reshape((-1)), label = 'fitted')
        # # plt.plot(mean_predict_y.reshape((-1)), label = 'fitted-other')
        # plt.legend()
        # plt.title('case: '+case+', Q train: '+str(self.log_Q.trainable)+', log-Q var: '+str(self.log_Q_variance)+', RMSE: '+str(RMSE_val)[:5]+', nll: '+str(ll_seq[-1])[:5])
        # plt.tight_layout()
        # plt.savefig(save_path_file[:(-4)]+'.pdf', edgecolor='none', format='pdf', bbox_inches='tight')
        # # plt.show()
        # plt.close()
        # # Y_train

        save_npz_flag = True
        if save_npz_flag:
            save_parameters = True
            if save_parameters:

                log_QQ = self.session.run((self.log_Q), feed_dict=feed_dict)
                Z_val = self.session.run((self.layers[-1].Z), feed_dict=feed_dict)

                U_val = self.session.run((self.layers[-1].U), feed_dict=feed_dict)
                # X_val = self.session.run((self.layers[-1].X), feed_dict=feed_dict)

                k_log_lengthscales = []
                k_log_variances = []
                for ki in range(len(self.kernels[0])):
                    k_log_lengthscales.append(self.session.run(self.kernels[0][ki].loglengthscales, feed_dict = feed_dict))
                    k_log_variances.append(self.session.run(self.kernels[0][ki].logvariance, feed_dict=feed_dict))
                # k_lengthscales_0 = self.session.run((self.kernels[0][0].loglengthscales), feed_dict=feed_dict)
                # k_lengthscales_1 = self.session.run((self.kernels[0][1].loglengthscales), feed_dict=feed_dict)
                # k_lengthscales_2 = self.session.run((self.kernels[0][2].loglengthscales), feed_dict=feed_dict)
                # k_lengthscales_3 = self.session.run((self.kernels[0][3].loglengthscales), feed_dict=feed_dict)
                #
                # k_log_variance_0 = self.session.run((self.kernels[0][0].logvariance), feed_dict=feed_dict)
                # k_log_variance_1 = self.session.run((self.kernels[0][1].logvariance), feed_dict=feed_dict)
                # k_log_variance_2 = self.session.run((self.kernels[0][2].logvariance), feed_dict=feed_dict)
                # k_log_variance_3 = self.session.run((self.kernels[0][3].logvariance), feed_dict=feed_dict)

                np.savez_compressed(save_path_file+'_results.npz',y_train_vfe=self.fit_y, y_test_vfe= self.predict_y,
                                    v_test_vfe_var = self.predict_y_var, Y_test_data=Y_test,Y_train_data=Y_train,
                                    Y_train_std=Y_train_std,CC_val=CC_val, DD_val=DD_val, log_R_cholesky=log_R_cholesky,
                                    log_QQ=log_QQ, Z_val=Z_val,U_val=U_val, X_val=fit_x_value, k_lengthscales=k_log_lengthscales,
                                    k_log_variances=k_log_variances,case = case, ll_seq = ll_seq, running_time_seq = running_time_seq,
                                    PG_num = PG_num, mc_posterior_samples = mc_posterior_samples)

            else:

                np.savez_compressed(save_path_file, y_train_vfe = self.fit_y, y_test_vfe = self.predict_y, Y_test_data = Y_test,
                                    Y_train_data = Y_train, Y_train_std = Y_train_std, RMSE_val = RMSE_val, case = case)


    def collect_samples_2023(self, num, spacing, control_inputs, test_len, sghmc_var_len = 0, U_collapse = False, Y_test = None, Y_train_std = 1., save_path_file = None, Y_train = None, case = 'C1', ll_seq = [0.], running_time_seq = [0.], PG_num = None, synthetic_data_function_plot = False, data_uu = None):

        batch_placeholder, adam_learning_rate = self.get_minibatch()
        feed_dict = {self.batch_placeholder: batch_placeholder,
                     self.adam_lr: adam_learning_rate}

        self.fit_x = self.session.run((self.layers[-1].X), feed_dict=feed_dict)

        if sghmc_var_len == 0:
            # pre-calculate the Cholesky decomposition of the kernel matrix
            Lm_inverse_seq = conditionals_multi_output.kernel_pre_cal(self.layers[-1].Z, self.layers[-1].kernel)

        pre_index = 1
        prediction_length = test_len + pre_index - 1

        predict_x_whole = []
        predict_x_combine_whole = []
        predict_x_var_whole = []

        total_rmse = []
        total_nll = []


        CC_val = self.session.run((self.likelihood.CC), feed_dict=feed_dict)
        DD_val = self.session.run((self.likelihood.DD), feed_dict=feed_dict)
        log_R_cholesky = self.session.run((self.likelihood.log_Rchols), feed_dict=feed_dict)

        if sghmc_var_len > 0:
            # run spacing-number of iterations for SG-HMC samplers
            # for j in range(spacing):
            #     batch_placeholder, adam_learning_rate = self.get_minibatch()
            #     feed_dict = {self.batch_placeholder: batch_placeholder,
            #                  self.adam_lr: adam_learning_rate}
            #     self.session.run((self.sample_op), feed_dict=feed_dict)

            # pre-calculate the Cholesky decomposition of the kernel matrix
            Lm_inverse_seq = conditionals_multi_output.kernel_pre_cal(self.layers[-1].Z, self.layers[-1].kernel)

        # set the initial value of latent state for prediction
        x_t = self.layers[-1].X[-pre_index][None, :]

        if U_collapse:
            if len(control_inputs.shape) > 0:
                x_control_inputs_combine = tf.concat(
                    (self.layers[-1].X[:(self.X_N - 1)], control_inputs[:(self.X_N - 1)]), axis=1)
            else:
                x_control_inputs_combine = self.layers[-1].X[:-1]

            U_val, U_val_variance_cholesky = conditionals_multi_output.collapse_u_mean_after_kernel_precalculation(
                Lm_inverse_seq, x_control_inputs_combine, self.layers[-1].X, self.layers[-1].Z, self.layers[-1].kernel,
                self.Q)
            U_val = U_val[0]
            U_variance_cholesky = U_val_variance_cholesky
        else:
            U_val = self.layers[-1].U
            U_variance_cholesky = None

        for num_i in range(num):

            predict_x_val = []
            predict_x_val_var = []

            for test_i in range(prediction_length):
                if (control_inputs.shape[1]) > 0:
                    # x_control_inputs_combine = tf.concat((x_t, control_inputs[test_i - (prediction_length)][None, :]), axis=1)
                    x_control_inputs_combine = tf.concat((x_t, control_inputs[test_i +Y_train.shape[0]][None, :]), axis=1)
                else:
                    x_control_inputs_combine = (x_t)

                f_t_mu, f_t_var = conditionals_multi_output.conditional_after_kernel_precalculation(Lm_inverse_seq, x_control_inputs_combine, self.layers[-1].Z, self.layers[-1].kernel,
                                                           U_val, white=True,full_cov=self.layers[-1].full_cov, q_sqrt=U_variance_cholesky, return_Lm=False)

                # add the mean function, which is the identity function (x_previous) here
                f_t_mu += x_t
                # x_tplus1 = f_t_mu
                x_tplus1 = f_t_mu + tf.random.normal(f_t_mu.shape, dtype=tf.float64) * tf.math.sqrt(f_t_var + self.Q)
                # aa = self.session.run((f_t_mu), feed_dict=feed_dict)
                # bb = self.session.run((f_t_var), feed_dict=feed_dict)

                # x_tplus1_val = np.asarray(self.session.run(x_tplus1, feed_dict=feed_dict))

                predict_x_val.append(x_tplus1)
                predict_x_val_var.append(f_t_var+self.Q)
                x_t = x_tplus1


            print('')
            print('Posterior: ', num_i, '-th samples obtained.')

            Y_predict = (tf.matmul(tf.stack(predict_x_val)[:, 0], self.likelihood.CC)+self.likelihood.DD)
            Y_predict = self.session.run(Y_predict, feed_dict=feed_dict).reshape((-1))
            Y_predict_var = (tf.matmul(tf.stack(predict_x_val_var), self.likelihood.CC**2))
            Y_predict_var = (self.session.run(Y_predict_var, feed_dict=feed_dict)+np.exp(2*log_R_cholesky)).reshape((-1))

            Y_test_30 = Y_test[:30].reshape((-1))
            Y_predict_30 = Y_predict[:30]
            Y_predict_var_30 = Y_predict_var[:30]

            print('y_predict_var: ', np.mean(Y_predict_var_30))

            RMSE_val = np.sqrt(np.mean((Y_test_30-Y_predict_30)**2))*Y_train_std
            total_rmse.append(RMSE_val)
            print('RMSE: ', RMSE_val)

            nll = -np.mean(norm.logpdf(Y_test_30, Y_predict_30, (Y_predict_var_30[:30]) ** (0.5)))
            total_nll.append(nll)
            print('NLL: ', nll)



        return total_rmse, total_nll

        # self.predict_y = (np.mean(np.einsum('ijk,kl->ijl', predict_x_whole, CC_val), axis=0)+DD_val[None, :]).reshape((-1))
        # self.predict_y_var = (np.mean(np.einsum('ijk,kl->ijl', predict_x_var_whole, CC_val**2), axis=0)).reshape((-1))+np.exp(2*log_R_cholesky).reshape((-1))

    def collect_samples_2023_speed_up(self, num, spacing, control_inputs, test_len, sghmc_var_len=0, U_collapse=False,
                             Y_test=None, Y_train_std=1., save_path_file=None, Y_train=None, case='C1', ll_seq=[0.],
                             running_time_seq=[0.], PG_num=None, synthetic_data_function_plot=False, data_uu=None):

        Y_test = tf.convert_to_tensor(Y_test, dtype=tf.float64)
        batch_placeholder, adam_learning_rate = self.get_minibatch()
        feed_dict = {self.batch_placeholder: batch_placeholder,
                     self.adam_lr: adam_learning_rate}

        # self.fit_x = self.session.run((self.layers[-1].X), feed_dict=feed_dict)

        if sghmc_var_len == 0:
            # pre-calculate the Cholesky decomposition of the kernel matrix
            Lm_inverse_seq = conditionals_multi_output.kernel_pre_cal(self.layers[-1].Z, self.layers[-1].kernel)

        pre_index = 1
        prediction_length = test_len + pre_index - 1

        log_R_cholesky = self.session.run((self.likelihood.log_Rchols), feed_dict=feed_dict)

        if sghmc_var_len > 0:

            # pre-calculate the Cholesky decomposition of the kernel matrix
            Lm_inverse_seq = conditionals_multi_output.kernel_pre_cal(self.layers[-1].Z, self.layers[-1].kernel)

        if U_collapse:
            if len(control_inputs.shape) > 0:
                x_control_inputs_combine = tf.concat(
                    (self.layers[-1].X[:(self.X_N - 1)], control_inputs[:(self.X_N - 1)]), axis=1)
            else:
                x_control_inputs_combine = self.layers[-1].X[:-1]

            U_val, U_val_variance_cholesky = conditionals_multi_output.collapse_u_mean_after_kernel_precalculation(
                Lm_inverse_seq, x_control_inputs_combine, self.layers[-1].X, self.layers[-1].Z, self.layers[-1].kernel,
                self.Q)
            U_val = U_val[0]
            U_variance_cholesky = U_val_variance_cholesky
        else:
            U_val = self.layers[-1].U
            U_variance_cholesky = None

        # for num_i in range(num):


        def _loop_body(ii, rmse_seq, y_predict_30_samples):

            # set the initial value of latent state for prediction
            x_t = self.layers[-1].X[-pre_index][None, :]

            predict_x_val = []
            predict_x_val_var = []

            for test_i in range(prediction_length):
                if (control_inputs.shape[1]) > 0:
                    # x_control_inputs_combine = tf.concat((x_t, control_inputs[test_i - (prediction_length)][None, :]), axis=1)
                    x_control_inputs_combine = tf.concat((x_t, control_inputs[test_i + Y_train.shape[0]][None, :]),
                                                         axis=1)
                else:
                    x_control_inputs_combine = (x_t)

                f_t_mu, f_t_var = conditionals_multi_output.conditional_after_kernel_precalculation(Lm_inverse_seq,
                                                                                                    x_control_inputs_combine,
                                                                                                    self.layers[-1].Z,
                                                                                                    self.layers[
                                                                                                        -1].kernel,
                                                                                                    U_val, white=True,
                                                                                                    full_cov=
                                                                                                    self.layers[
                                                                                                        -1].full_cov,
                                                                                                    q_sqrt=U_variance_cholesky,
                                                                                                    return_Lm=False)

                # add the mean function, which is the identity function (x_previous) here
                f_t_mu += x_t
                # x_tplus1 = f_t_mu
                x_tplus1 = f_t_mu + tf.random.normal(f_t_mu.shape, dtype=tf.float64) * tf.math.sqrt(f_t_var + self.Q)
                # aa = self.session.run((f_t_mu), feed_dict=feed_dict)
                # bb = self.session.run((f_t_var), feed_dict=feed_dict)

                # x_tplus1_val = np.asarray(self.session.run(x_tplus1, feed_dict=feed_dict))

                predict_x_val.append(x_tplus1)
                predict_x_val_var.append(f_t_var + self.Q)
                x_t = x_tplus1

            batch_placeholder, adam_learning_rate = self.get_minibatch()
            # feed_dict = {self.batch_placeholder: batch_placeholder,
            #              self.adam_lr: adam_learning_rate}

            Y_predict = (tf.matmul(tf.stack(predict_x_val)[:, 0], self.likelihood.CC) + self.likelihood.DD)

            # Y_predict_var = (tf.matmul(tf.stack(predict_x_val_var), self.likelihood.CC ** 2))+tf.math.exp(2 * log_R_cholesky)

            Y_test_30 = Y_test[:30, 0]
            Y_predict_30 = Y_predict[:30, 0]
            # Y_predict_var_30 = Y_predict_var[:30, 0, 0]

            RMSE_val = tf.math.sqrt(tf.reduce_mean((Y_test_30 - Y_predict_30) ** 2)) * Y_train_std
            # nll = 0.

            rmse_seq = rmse_seq.write(ii, RMSE_val)
            y_predict_30_samples = y_predict_30_samples.write(ii, Y_predict_30)
            ii = tf.add(ii, 1)
            rest_values = [ii, rmse_seq, y_predict_30_samples]

            return rest_values

        rmse_seq = tf.TensorArray(tf.float64, size=num, dynamic_size=False)
        y_predict_30_samples = tf.TensorArray(tf.float64, size=num, element_shape=[30,], dynamic_size=False)

        _loop_vars = [tf.constant(0, dtype = tf.int32), rmse_seq, y_predict_30_samples]


        ii, rmse_seq, y_predict_30_samples = tf.compat.v1.while_loop(
            cond=lambda ii, *args: ii < (num),
            body=_loop_body,
            loop_vars=_loop_vars,
            back_prop=False,
            parallel_iterations=10)

        rmse_seq = rmse_seq.stack()
        rmse_seq_val = self.session.run((rmse_seq), feed_dict=feed_dict)

        # nll_seq = nll_seq.stack()
        # nll_seq_val = self.session.run((nll_seq), feed_dict=feed_dict)

        # logpdf_norm = tfp.distributions.Normal(Y_predict_30, (Y_predict_var_30[:30]) ** (0.5))
        # nll = -tf.reduce_mean(logpdf_norm.log_prob(Y_test_30))
        y_predict_30_samples = y_predict_30_samples.stack()
        y_predict_30_samples_val = self.session.run((y_predict_30_samples), feed_dict=feed_dict)


        tf.keras.backend.clear_session()

        return rmse_seq_val, y_predict_30_samples_val


    def RMSE_calculate_per_iteration(self, num, spacing, control_inputs, test_len, sghmc_var_len = 0, U_collapse = False, Y_test = None, Y_train_std = 1., Y_train = None):

        batch_placeholder, adam_learning_rate = self.get_minibatch()
        feed_dict = {self.batch_placeholder: batch_placeholder,
                     self.adam_lr: adam_learning_rate}
        #
        # self.fit_x = self.session.run((self.layers[-1].X), feed_dict=feed_dict)

        if sghmc_var_len == 0:
            # pre-calculate the Cholesky decomposition of the kernel matrix
            Lm_inverse_seq = conditionals_multi_output.kernel_pre_cal(self.layers[-1].Z, self.layers[-1].kernel)

        pre_index = 1
        prediction_length = test_len + pre_index - 1

        predict_x_whole = []
        # predict_x_combine_whole = []
        predict_x_var_whole = []

        # mc_posterior_samples = [[]]*sghmc_var_len

        # xx_test_coor_whole = []
        # ff_test_coor_mu_whole = []
        # ff_test_coor_var_whole = []

        for num_i in range(num):

            if sghmc_var_len > 0:
                # run spacing-number of iterations for SG-HMC samplers
                batch_placeholder, adam_learning_rate = self.get_minibatch()
                feed_dict = {self.batch_placeholder: batch_placeholder,
                             self.adam_lr: adam_learning_rate}
                for j in range(spacing):
                    self.session.run((self.sample_op), feed_dict=feed_dict)

                # pre-calculate the Cholesky decomposition of the kernel matrix
                Lm_inverse_seq = conditionals_multi_output.kernel_pre_cal(self.layers[-1].Z, self.layers[-1].kernel)

            # set the initial value of latent state for prediction
            x_t = self.layers[-1].X[-pre_index][None, :]

            # for len_i in range(sghmc_var_len):
            #     mc_posterior_samples[len_i].append(self.session.run(self.vars[len_i], feed_dict=feed_dict))

            if U_collapse:
                if len(control_inputs.shape) > 0:
                    x_control_inputs_combine = tf.concat((self.layers[-1].X[:(self.X_N-1)],control_inputs[:(self.X_N-1)]), axis=1)
                else:
                    x_control_inputs_combine = self.layers[-1].X[:-1]

                U_val, U_val_variance_cholesky = conditionals_multi_output.collapse_u_mean_after_kernel_precalculation(Lm_inverse_seq, x_control_inputs_combine, self.layers[-1].X, self.layers[-1].Z, self.layers[-1].kernel, self.Q)
                U_val = U_val[0]
                U_variance_cholesky = U_val_variance_cholesky
            else:
                U_val = self.layers[-1].U
                U_variance_cholesky = None


            #########################

            #########################


            #########################

            #########################

            predict_x_val = []
            predcit_x_val_var = []
            for test_i in range(prediction_length):
                if (control_inputs.shape[1]) > 0:
                    # x_control_inputs_combine = tf.concat((x_t, control_inputs[test_i - (prediction_length)][None, :]), axis=1)
                    x_control_inputs_combine = tf.concat((x_t, control_inputs[test_i +Y_train.shape[0]][None, :]), axis=1)
                else:
                    x_control_inputs_combine = (x_t)

                # x_ci_val = np.asarray(self.session.run(x_control_inputs_combine, feed_dict=feed_dict))
                # predict_x_combine.append(x_ci_val)

                f_t_mu, f_t_var = conditionals_multi_output.conditional_after_kernel_precalculation(Lm_inverse_seq, x_control_inputs_combine, self.layers[-1].Z, self.layers[-1].kernel,
                                                           U_val, white=True,full_cov=self.layers[-1].full_cov, q_sqrt=U_variance_cholesky, return_Lm=False)

                # add the mean function, which is the identity function (x_previous) here
                f_t_mu += x_t
                # x_tplus1 = f_t_mu
                x_tplus1 = f_t_mu + tf.random.normal(f_t_mu.shape, dtype=tf.float64) * tf.math.sqrt(f_t_var + self.Q)
                # aa = self.session.run((f_t_mu), feed_dict=feed_dict)
                # bb = self.session.run((f_t_var), feed_dict=feed_dict)

                # x_tplus1_val = np.asarray(self.session.run(x_tplus1, feed_dict=feed_dict))

                predict_x_val.append(x_tplus1)
                predcit_x_val_var.append(f_t_var+self.Q)
                x_t = x_tplus1

            predict_x_whole.append(predict_x_val)
            # predict_x_combine_whole.append(predict_x_combine)
            predict_x_var_whole.append(predcit_x_val_var)

            print('')
            print('Posterior: ', num_i, '-th samples obtained.')


        predict_x_whole = tf.stack(predict_x_whole)

        predict_x_var_whole = tf.stack(predict_x_var_whole)

        predict_x_whole = self.session.run(predict_x_whole, feed_dict=feed_dict)[:, :, 0]
        predict_x_var_whole = self.session.run(predict_x_var_whole, feed_dict=feed_dict)[:, :, 0]



        CC_val = self.session.run((self.likelihood.CC), feed_dict=feed_dict)
        DD_val = self.session.run((self.likelihood.DD), feed_dict=feed_dict)

        # fit_x_value = np.asarray(self.fit_x)[1:]

        log_R_cholesky = self.session.run((self.likelihood.log_Rchols), feed_dict=feed_dict)


        predict_y = (np.mean(np.einsum('ijk,kl->ijl', predict_x_whole, CC_val), axis=0)+DD_val[None, :]).reshape((-1))
        predict_y_var = (np.mean(np.einsum('ijk,kl->ijl', predict_x_var_whole, CC_val**2), axis=0)).reshape((-1))+np.exp(2*log_R_cholesky)

        Y_test_30 = Y_test[:30].reshape((-1))
        Y_predict_30 = predict_y[:30]

        nll = -np.mean(norm.logpdf(Y_test_30, Y_predict_30, (predict_y_var.reshape((-1))[:30]) ** (0.5)))

        RMSE_val = np.sqrt(np.mean((Y_test_30-Y_predict_30)**2))*Y_train_std

        del predict_x_whole, predict_x_var_whole, CC_val, DD_val, log_R_cholesky, predict_y, predict_y_var, Y_test_30, Y_predict_30

        return nll, RMSE_val





    def sghmc_step(self):
        batch_placeholder, adam_learning_rate = self.get_minibatch()
        feed_dict = {self.batch_placeholder: batch_placeholder,
                     self.adam_lr: adam_learning_rate}
        self.session.run(self.burn_in_op, feed_dict=feed_dict)
        for j in range(10):
            batch_placeholder, adam_learning_rate = self.get_minibatch()
            feed_dict = {self.batch_placeholder: batch_placeholder,
                         self.adam_lr: adam_learning_rate}
            self.session.run(self.burn_in_op, feed_dict=feed_dict)
            self.session.run((self.sample_op), feed_dict=feed_dict)

        values = self.session.run((self.vars))
        sample = {}
        for var, value in zip(self.vars, values):
            sample[var] = value
        self.window.append(sample)
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]


    def gp_x_sampling(self):
        batch_placeholder, adam_learning_rate = self.get_minibatch()
        feed_dict = {self.batch_placeholder: batch_placeholder,
                     self.adam_lr: adam_learning_rate}
        # i = np.random.randint(len(self.window))
        # feed_dict.update(self.window[i])
        self.session.run(self.pg_x_sampling_op, feed_dict=feed_dict)

    def train_hypers(self):
        batch_placeholder, adam_learning_rate = self.get_minibatch()
        feed_dict = {self.batch_placeholder: batch_placeholder,
                     self.adam_lr: adam_learning_rate}
        i = np.random.randint(len(self.window))
        feed_dict.update(self.window[i])
        self.session.run(self.hyper_train_op, feed_dict=feed_dict)

    def print_sample_performance(self, iter, U_collapse= False, posterior=False):
        batch_placeholder, adam_learning_rate = self.get_minibatch()
        feed_dict = {self.batch_placeholder: batch_placeholder,
                     self.adam_lr: adam_learning_rate}
        if posterior:
            feed_dict.update(np.random.choice(self.posterior_samples))
        if U_collapse:
            marginal_ll = -self.session.run((self.nll), feed_dict=feed_dict)

            ll_nll_reg_trace_inverse_Q_B = -self.session.run((self.nll_reg_trace_inverse_Q_B), feed_dict=feed_dict)
            ll_nll_log_likelihood = -self.session.run((self.nll_log_likelihood), feed_dict=feed_dict)
            ll_log_priors = -self.session.run((self.nll_part_prior), feed_dict=feed_dict)
            ll_prior_term = -self.session.run((self.x_t_prior_Q), feed_dict=feed_dict)
            ll_later_term1 = -self.session.run((self.later_term1), feed_dict=feed_dict)
            ll_later_term2 = -self.session.run((self.later_term2), feed_dict=feed_dict)

            if np.mod(iter, 10)==0:
                print(
                    'TRAIN | iter = %6d      sample marginal LL = %5.2f       later_term1= %5.2f      later_term2= %5.2f      x_prior_term= %5.2f        LL trace_inverse_Q_B= %5.2f        log-likelihood = %5.2f        log-prior= %5.2f' % (
                        iter, marginal_ll, ll_later_term1, ll_later_term2, ll_prior_term, ll_nll_reg_trace_inverse_Q_B, ll_nll_log_likelihood, ll_log_priors))


        else:
            marginal_ll = -self.session.run((self.nll), feed_dict=feed_dict)
            Y_val = self.session.run((self.Y), feed_dict=feed_dict)


            val_trace_inverse_Q_B = -self.session.run((self.nll_reg_trace_inverse_Q_B), feed_dict=feed_dict)
            val_reg_x_prior = -self.session.run((self.x_t_prior_Q), feed_dict=feed_dict)
            val_log_likelihood = -self.session.run((self.nll_log_likelihood), feed_dict=feed_dict)
            val_other_priors = -self.session.run((self.nll_part_prior), feed_dict=feed_dict)

            if np.mod(iter, 10)==0:
                print(
                    'TRAIN | iter = %6d      sample marginal LL = %5.2f       val_trace_inverse_Q_B= %5.2f      val_reg_x_prior= %5.2f        val_log_likelihood= %5.2f      val_other_priors= %5.2f' % (
                        iter, marginal_ll, val_trace_inverse_Q_B, val_reg_x_prior, val_log_likelihood, val_other_priors))

        return marginal_ll
