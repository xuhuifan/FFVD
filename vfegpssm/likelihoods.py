# Credit to GPFlow.

import tensorflow as tf
# import tensorflow_probability.math.fill_triangular as fill_triangular
import tensorflow_probability as tfp
import numpy as np
from .quadrature import ndiagquad
# import tensorflow_probability as tfp

class Gaussian(object):

    def __init__(self, Y_dim, X_output_dim, CC=None, DD = None, RR_chol = None, hyperparameter_sampling = False, likelihood_traning = True):
        if hyperparameter_sampling:
            self.CC = tf.Variable(tf.ones((X_output_dim, Y_dim), dtype=tf.float64), trainable=False,name='C') if CC is None else CC
            self.DD = tf.Variable(tf.zeros((Y_dim), dtype=tf.float64), trainable=False, name='d') if DD is None else DD
        else:
            if CC is None:
                self.CC = tf.Variable(tf.ones((X_output_dim, Y_dim), dtype = tf.float64), trainable = likelihood_traning, name = 'C')
            else:
                self.CC = tf.Variable(CC, trainable=likelihood_traning, name='C')
            if DD is None:
                self.DD = tf.Variable(tf.zeros((Y_dim), dtype = tf.float64), trainable = likelihood_traning, name = 'd')
            else:
                self.DD = tf.Variable(DD, trainable = likelihood_traning, name = 'd')

        # # Rchols_vector = tf.Variable(tf.ones(int(Y_dim*(Y_dim+1)/2), dtype=tf.float64)*0.2, trainable=False, name='R-Cholesky-vector')
        # Rchols_diag = tf.Variable(tf.ones(Y_dim, dtype= tf.float64)*tf.math.log(tf.constant(1.5, dtype = tf.float64)), trainable = False, name = 'R-Cholesky-diagonal')
        # Rchols_lower_tri = tf.Variable(tf.ones(int(Y_dim*(Y_dim-1)/2), dtype = tf.float64), trainable = False, name = 'R-Cholesky-lower-tri')
        # # tf.linalg.set_diag(Rchols_matrix, tf.math.exp(tf.linalg.tensor_diag_part(Rchols_matrix)))
        #
        # # self.Rchols = Rchols_matrix
        #
        # Rchols_matrix = tf.zeros((Y_dim, Y_dim), dtype = tf.float64)
        # rindex = 0
        # for ii in range(Y_dim):
        #     for jj in range(ii+1):
        #         if ii == jj:
        #             Rchols_matrix[ii, jj] = tf.math.exp(Rchols_diag[ii])
        #         else:
        #             Rchols_matrix[ii, jj] = Rchols_lower_tri[rindex]
        #             rindex += 1
        # self.Rchols = Rchols_matrix
        #

        if Y_dim == 1:
            # self.Rchols = tf.exp(tf.Variable(tf.ones((Y_dim, Y_dim), dtype= tf.float64)*tf.math.log(tf.constant((0.1)**(0.5), dtype = tf.float64)), trainable = False, name = 'R-Cholesky-diagonal')) if RR_chol is None else RR_chol
            if hyperparameter_sampling:
                self.log_Rchols = tf.Variable(tf.ones((Y_dim, Y_dim), dtype=tf.float64) * tf.math.log(tf.constant((0.1), dtype=tf.float64)),trainable=False, name='log-R-Cholesky-diagonal')
                self.Rchols = tf.exp(self.log_Rchols) if RR_chol is None else RR_chol
            else:
                if RR_chol is None:
                    self.log_Rchols = tf.Variable(tf.ones((Y_dim, Y_dim), dtype=tf.float64) * tf.math.log(tf.constant((0.1), dtype=tf.float64)),trainable=likelihood_traning, name='log-R-Cholesky-diagonal')
                else:
                    self.log_Rchols = tf.Variable(tf.math.log(RR_chol),trainable=likelihood_traning, name='log-R-Cholesky-diagonal')
                self.Rchols = tf.exp(self.log_Rchols)
        else:
            self.Rchols_initial = tf.Variable(tf.ones((Y_dim, Y_dim), dtype= tf.float64), trainable = False, name = 'R-Cholesky-diagonal')

            mask = tf.Variable(np.tril(np.ones((Y_dim, Y_dim)), -1), trainable = False)
            self.Rchols_diag = tf.Variable(tf.ones(Y_dim, dtype=tf.float64) * tf.math.log(tf.constant(1.5, dtype=tf.float64)), trainable=False,name='R-Cholesky-diagonal')
            self.Rchols = self.Rchols_initial*mask+tf.linalg.diag(tf.exp(self.Rchols_diag))
        # self.Rchols = tf.Variable(tf.eye(Y_dim, dtype = tf.float64), trainable = True, name = 'R-Cholesky') if R is None else tf.compt.v1.linalg.cholesky(R)

        # self.log_RR = tf.exp(tf.Variable(np.log(tf.ones((Y_dim, Y_dim))), dtype=tf.float64, name='lik_log_variance'))


    # def logp(self, F, Y):
    #     return self.logdensity(Y, F, self.variance)

    def conditional_mean(self, F):
        return tf.identity(F)

    # def conditional_variance(self, F):
    #     return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def predict_mean(self, X_end):
        predict_v = tf.matmul(X_end, self.CC)+self.DD

        return predict_v

    def predict_density(self, ymean, Rchols, Y):
        # ycov = tf.matmul(Rchols, tf.transpose(Rchols))
        return logdensity_norm(Y, ymean, Rchols)

    # def variational_expectations(self, Fmu, Fvar, Y):
    #     return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) \
    #            - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance

def logdensity_norm_diag_nonvec(y, ymean, Rchols):
    exp_term = -0.5*(((y-ymean)/Rchols[None, :])**2)
    log_Rchols = -tf.math.log(Rchols)[None, :]

    return (exp_term+log_Rchols)


def logdensity_norm_diag(y, ymean, Rchols):
    # log_prob_v = tfp.distributions.MultivariateNormalFullCovariance.log_prob(value = y, loc =ymean, covariance_matrix=ycov)

    # Rchols_inv = tf.linalg.triangular_solve(Rchols, np.eye(Rchols.shape[0]), lower = True)
    # exp_term = -0.5*tf.reduce_sum(tf.square(tf.matmul(y - ymean, tf.transpose(Rchols_inv))), axis=1)
    exp_term = -0.5*tf.reduce_sum(((y-ymean)/Rchols[None, :])**2, axis=1)
    log_Rchols = -tf.reduce_sum(tf.math.log(Rchols))

    # alphav = tf.linalg.triangular_solve(Rchols, tf.transpose(y - ymean), lower = True)
    # exp_term = -0.5*tf.reduce_sum(tf.square(alphav), axis=0)
    #
    # # exp_term = 0.5*tf.matmul((tf.matmul(y-ymean, Rchols_inv)), tf.transpose(tf.matmul(y-ymean, Rchols_inv)))
    # # R_inv = tf.matmul(tf.transpose(Rchols_inv), Rchols_inv)
    # logdet_R = -tf.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(Rchols)))

    return (exp_term+log_Rchols)


def logdensity_norm(y, ymean, Rchols):
    # log_prob_v = tfp.distributions.MultivariateNormalFullCovariance.log_prob(value = y, loc =ymean, covariance_matrix=ycov)

    # Rchols_inv = tf.linalg.triangular_solve(Rchols, np.eye(Rchols.shape[0]), lower = True)
    # exp_term = -0.5*tf.reduce_sum(tf.square(tf.matmul(y - ymean, tf.transpose(Rchols_inv))), axis=1)

    alphav = tf.linalg.triangular_solve(Rchols, tf.transpose(y - ymean), lower = True)
    exp_term = -0.5*tf.reduce_sum(tf.square(alphav), axis=0)

    # exp_term = 0.5*tf.matmul((tf.matmul(y-ymean, Rchols_inv)), tf.transpose(tf.matmul(y-ymean, Rchols_inv)))
    # R_inv = tf.matmul(tf.transpose(Rchols_inv), Rchols_inv)
    logdet_R = -tf.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(Rchols)))

    return (exp_term+logdet_R)

def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter


class Bernoulli(object):
    def __init__(self, invlink=inv_probit, **kwargs):
        self.invlink = invlink
        self.num_gauss_hermite_points = 20

    def logdensity(self, x, p):
        return tf.math.log(tf.where(tf.equal(x, 1), p, 1-p))

    def logp(self, F, Y):
        return self.logdensity(Y, self.invlink(F))

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - tf.square(p)

    def predict_density(self, Fmu, Fvar, Y):
        p = self.predict_mean_and_var(Fmu, Fvar)[0]
        return self.logdensity(Y, p)

    def predict_mean_and_var(self, Fmu, Fvar):
        if self.invlink is inv_probit:
            p = inv_probit(Fmu / tf.sqrt(1 + Fvar))
            return p, p - tf.square(p)
        else:
            # for other invlink, use quadrature
            integrand2 = lambda *X: self.conditional_variance(*X) + tf.square(self.conditional_mean(*X))
            E_y, E_y2 = ndiagquad([self.conditional_mean, integrand2],
                                  self.num_gauss_hermite_points,
                                  Fmu, Fvar)
            V_y = E_y2 - tf.square(E_y)
            return E_y, V_y

    def variational_expectations(self, Fmu, Fvar, Y):
        r"""
        Compute the expected log density of the datasets, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           \int (\log p(y|f)) q(f) df.
        """
        return ndiagquad(self.logp, self.num_gauss_hermite_points, Fmu, Fvar, Y=Y)

