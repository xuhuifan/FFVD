# Credit to GPflow

import tensorflow as tf


def base_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, q_sqrt=None, white=False, return_Lm=False):
    """
    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2;0,Kmm)
      p(g1) = N(g1;0,Knn)
      p(g1|g2) = N(g1;0,Knm)
    And
      q(g2) = N(g2;f,q_sqrt*q_sqrt^T)
    This method computes the mean and (co)variance of
      q(g1) = \int q(g2) p(g1|g2)
    :param Kmn: M x N
    :param Kmm: M x M
    :param Knn: N x N  or  N
    :param f: M x R
    :param full_cov: bool
    :param q_sqrt: None or R x M x M (lower triangular)
    :param white: bool
    :return: N x R  or R x N x N
    """
    # compute kernel stuff
    num_func = tf.shape(f)[1]  # R
    try:
        Lm = tf.linalg.cholesky(Kmm)
    except:
        Lm = tf.linalg.cholesky(Kmm + tf.eye(Kmm.shape[0], dtype=tf.float64) * 1e-4)


    # Compute the projection matrix A
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.matmul(A, A, transpose_a=True)
        fvar = tf.tile(fvar[None, :, :], [num_func, 1, 1])  # R x N x N
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), 0)
        fvar = tf.tile(fvar[None, :], [num_func, 1])  # R x N

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.linalg.triangular_solve(tf.transpose(Lm), A, lower=False)

    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # R x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = q_sqrt
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # R x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # R x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # R x N

    if not full_cov:
        fvar = tf.transpose(fvar)  # N x R
    if return_Lm:
        return fmean, fvar, Lm

    return fmean, fvar # N x R, R x N x N or N x R


def conditional(Xnew, X, kern, f, *, full_cov=False, q_sqrt=None, white=False, return_Lm=False):
    """
    Given f, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.
    Additionally, there may be Gaussian uncertainty about f as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.
    Additionally, the GP may have been centered (whitened) so that
        p(v) = N(0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case `f` represents the values taken by v.
    The method can either return the diagonals of the covariance matrix for
    each output (default) or the full covariance matrix (full_cov=True).
    We assume R independent GPs, represented by the columns of f (and the
    first dimension of q_sqrt).
    :param Xnew: datasets matrix, size N x D. Evaluate the GP at these new points
    :param X: datasets points, size M x D.
    :param kern: GPflow kernel.
    :param f: datasets matrix, M x R, representing the function values at X,
        for K functions.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size M x R or R x M x M.
    :param white: boolean of whether to use the whitened representation as
        described above.
    :return:
        - mean:     N x R
        - variance: N x R (full_cov = False), R x N x N (full_cov = True)
    """

    num_data = tf.shape(X)[0]  # M
    f_mu = []
    f_var = []
    for kk in range(len(kern)):
        Kmm = kern[kk].K(X) + tf.eye(num_data, dtype=tf.float64) * 1e-5
        Kmn = kern[kk].K(X, Xnew)
        if full_cov:
            Knn = kern[kk].K(Xnew)
        else:
            Knn = kern[kk].Kdiag(Xnew)

        f_k_mu, f_k_var = base_conditional(Kmn, Kmm, Knn, f[:, kk][:, None], full_cov=full_cov, q_sqrt=q_sqrt, white=white, return_Lm=return_Lm) # N x R, N x R or R x N x N

        f_mu.append(f_k_mu)
        f_var.append(f_k_var)

    return tf.transpose(tf.convert_to_tensor(f_mu)[:, :, 0]), tf.transpose(tf.convert_to_tensor(f_var)[:, :, 0])



def kernel_pre_cal(X, kern):
    """
    Given f, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.
    Additionally, there may be Gaussian uncertainty about f as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.
    Additionally, the GP may have been centered (whitened) so that
        p(v) = N(0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case `f` represents the values taken by v.
    The method can either return the diagonals of the covariance matrix for
    each output (default) or the full covariance matrix (full_cov=True).
    We assume R independent GPs, represented by the columns of f (and the
    first dimension of q_sqrt).
    :param Xnew: datasets matrix, size N x D. Evaluate the GP at these new points
    :param X: datasets points, size M x D.
    :param kern: GPflow kernel.
    :param f: datasets matrix, M x R, representing the function values at X,
        for K functions.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size M x R or R x M x M.
    :param white: boolean of whether to use the whitened representation as
        described above.
    :return:
        - mean:     N x R
        - variance: N x R (full_cov = False), R x N x N (full_cov = True)
    """

    num_data = tf.shape(X)[0]  # M

    Lm_inverse_seq = []
    for kk in range(len(kern)):
        Kmm = kern[kk].K(X) + tf.eye(num_data, dtype=tf.float64) * 1e-5

        try:
            Lm_kk = (tf.linalg.cholesky(Kmm))
        except:
            Lm_kk = tf.linalg.cholesky(Kmm + tf.eye(Kmm.shape[0], dtype=tf.float64) * 1e-4)

        Lm_inverse_seq.append(tf.linalg.triangular_solve(tf.transpose(Lm_kk), tf.eye(num_data, dtype = tf.float64), lower=False))
        # Lm_inverse_seq.append(tf.linalg.inv(tf.transpose(Lm_kk)))

    return Lm_inverse_seq

# def collapse_u_mean_after_kernel_precalculation(Lm_inverse_seq, X_combine, X, Z, kern, Q):
#
#     tilde_F = []
#     for kk in range(len(kern)):
#         Kmn = kern[kk].K(Z, X_combine)
#         tilde_F.append(tf.matmul(Lm_inverse_seq[kk],Kmn))
#
#     tilde_F_value = tf.stack(tilde_F) # shape: (4, 100, 500)
#     tilde_F_Q_inverse = tilde_F_value*((Q**(-1))[:, None, None]) # shape: (4, 100, 500)
#
#     inverse_H_matrix = tf.einsum('jik,jbk->jib', tilde_F_Q_inverse, tilde_F_value)+tf.eye(tf.cast(Z.shape[0], dtype = tf.int32), dtype = tf.float64)[None, :, :]
#     X_Qinvserse_transpose = (X[1:]-X[:(-1)])*((Q**(-1))[None, :]) # shape: (500, 4)
#     # X_Qinverse_tilde_F = tf.reduce_sum(tilde_F_value*tf.transpose(X_Qinvserse_transpose)[:, None, :], axis=2)
#     # X_Qinverse_tilde_F = tf.reduce_sum(tilde_F_value*tf.transpose(X_Qinvserse_transpose)[:, None, :], axis=2)
#     X_Qinverse_tilde_F = tf.einsum('ijk,ki->ij', tilde_F_value, X_Qinvserse_transpose)
#
#     U_mean = []
#     Lm_spe = []
#     for kk in range(inverse_H_matrix.shape[0]):
#         U_mean.append(tf.linalg.solve(inverse_H_matrix[kk], X_Qinverse_tilde_F[kk][:, None])[:, 0])
#
#         try:
#             Lm_kk = (tf.linalg.cholesky(inverse_H_matrix[kk]))
#         except:
#             Lm_kk = tf.linalg.cholesky(inverse_H_matrix[kk] + tf.eye(inverse_H_matrix[kk].shape[0], dtype=tf.float64) * 1e-5)
#
#         Lm_spe.append(Lm_kk)
#
#         # Lm_inverse_seq.append(tf.linalg.triangular_solve(Lm_kk, tf.eye(num_data, dtype=tf.float64), lower=True))
#
#     U_mean = tf.stack(U_mean)
#     Lm_spe = tf.stack(Lm_spe)
#
#     return tf.transpose(U_mean), Lm_spe

def collapse_u_mean_after_kernel_precalculation(Lm_inverse_seq, X_combine, X, Z, kern, Q):

    U_mean = []
    Lm_inverse_dd_seq = []
    for dd in range(len(kern)):

        Knm = kern[dd].K(X_combine, Z)
        tilde_F = tf.matmul(Knm, Lm_inverse_seq[dd]) # shape: (None, 100)

        inverse_H_matrix = tf.matmul(tf.transpose(tilde_F), tilde_F)/Q[dd]+tf.eye(tf.cast(Z.shape[0], dtype = tf.int32), dtype = tf.float64)
        X_transpose = (X[1:, dd]-X[:(-1), dd])[None, :] # shape: (1, None)
        X_Qinverse_tilde_F = tf.matmul(X_transpose, tilde_F)/Q[dd] # shape: (1, 100)

        U_mean.append(tf.linalg.solve(inverse_H_matrix, tf.transpose(X_Qinverse_tilde_F)))

        Lm_dd = (tf.linalg.cholesky(inverse_H_matrix))
        Lm_inverse_dd_seq.append(tf.linalg.triangular_solve(tf.transpose(Lm_dd), tf.eye(tf.cast(Z.shape[0], dtype = tf.int32), dtype = tf.float64), lower=False))

    U_mean = tf.stack(U_mean)
    Lm_inverse_dd_seq = tf.stack(Lm_inverse_dd_seq)

    return tf.transpose(U_mean), Lm_inverse_dd_seq


def collapse_after_kernel_precalculation(Lm_inverse_seq, X_combine, X, Z, kern, Q, batch_size, Y_N):

    term1 = 0.
    term2 = 0.
    trace_Q_inverse_B = 0.

    # mean_term = []

    for dd in range(len(kern)):

        Knm = kern[dd].K(X_combine, Z)

        tilde_F = tf.matmul(Knm, Lm_inverse_seq[dd]) # shape: (None, 100)

        Knn_diag = kern[dd].Kdiag(X_combine)

        inverse_H_matrix = tf.matmul(tf.transpose(tilde_F), tilde_F)/(batch_size*Q[dd])*Y_N+tf.eye(tf.cast(Z.shape[0], dtype = tf.int32), dtype = tf.float64)
        X_transpose = (X[1:, dd]-X[:(-1), dd])[None, :] # shape: (1, None)
        X_Qinverse_tilde_F = tf.matmul(X_transpose, tilde_F)/(batch_size*Q[dd])*Y_N # shape: (1, 100)
        # X_Qinverse_tilde_F = tf.matmul(X_transpose, tilde_F)/Q[dd] # shape: (1, 100)

        # mean_term.append(tf.linalg.solve(inverse_H_matrix, tf.transpose(X_Qinverse_tilde_F)))

        term1 += -0.5*tf.linalg.logdet(inverse_H_matrix)
        term2 += 0.5*(tf.matmul(X_Qinverse_tilde_F, tf.linalg.solve(inverse_H_matrix, tf.transpose(X_Qinverse_tilde_F))))[0,0]
        trace_Q_inverse_B += -0.5*tf.reduce_sum((Knn_diag - tf.reduce_sum(tilde_F**2, axis=1))/Q[dd])

    return -term1/Y_N, -term2/Y_N, -trace_Q_inverse_B/Y_N


def uncollapse_after_kernel_precalculation(Lm_inverse_seq, X_combine, X, Z, U, kern, Q, batch_size, Y_N):
    term3 = 0.
    term4 = 0.
    trace_Q_inverse_B = 0.

    for kk in range(len(kern)):

        Knm = kern[kk].K(X_combine, Z)
        tilde_F = tf.matmul(Knm, Lm_inverse_seq[kk]) # shape: (None, 100)
        Knn_diag = kern[kk].Kdiag(X_combine)

        inverse_H_matrix = tf.matmul(tf.transpose(tilde_F), tilde_F)/(batch_size*Q[kk])*Y_N+tf.eye(tf.cast(Z.shape[0], dtype = tf.int32), dtype = tf.float64)
        X_transpose = (X[1:, kk]-X[:(-1), kk])[None, :] # shape: (1, None)
        X_Qinverse_tilde_F = tf.matmul(X_transpose, tilde_F)/(batch_size*Q[kk])*Y_N # shape: (1, 100)

        term3 += -0.5*tf.matmul(tf.matmul(U[:, kk][None, :], inverse_H_matrix), U[:, kk][:, None])
        term4 += tf.matmul(X_Qinverse_tilde_F, U[:, kk][:, None])
        trace_Q_inverse_B += -0.5*tf.reduce_sum((Knn_diag - tf.reduce_sum(tilde_F**2, axis=1))/Q[kk])

    return -term3/Y_N, -term4/Y_N, -trace_Q_inverse_B/Y_N

def uncollapse_after_kernel_precalculation_v1(Lm_inverse_seq, X_combine, X, Z, U, kern, Q, batch_size, Y_N):
    term3 = 0.
    term4 = 0.
    term5 = 0.
    trace_Q_inverse_B = 0.

    for kk in range(len(kern)):

        Knm = kern[kk].K(X_combine, Z)
        tilde_F = tf.matmul(Knm, Lm_inverse_seq[kk]) # shape: (None, 100)
        Knn_diag = kern[kk].Kdiag(X_combine)

        inverse_H_matrix = tf.matmul(tf.transpose(tilde_F), tilde_F)/(batch_size*Q[kk])*Y_N+tf.eye(tf.cast(Z.shape[0], dtype = tf.int32), dtype = tf.float64)
        X_transpose = (X[1:, kk]-X[:(-1), kk])[None, :] # shape: (1, None)
        X_Qinverse_tilde_F = tf.matmul(X_transpose, tilde_F)/(batch_size*Q[kk])*Y_N # shape: (1, 100)

        term5 += (-0.5*tf.reduce_sum(X_transpose**2)/(batch_size*Q[kk])*Y_N-0.5*tf.math.log(Q[kk])*Y_N)

        term3 += -0.5*tf.matmul(tf.matmul(U[:, kk][None, :], inverse_H_matrix), U[:, kk][:, None])
        term4 += tf.matmul(X_Qinverse_tilde_F, U[:, kk][:, None])
        trace_Q_inverse_B += -0.5*tf.reduce_sum((Knn_diag - tf.reduce_sum(tilde_F**2, axis=1))/Q[kk])

    return -term3/Y_N, -term4/Y_N, -trace_Q_inverse_B/Y_N, -term5/Y_N


def conditional_after_kernel_precalculation(Lm_inverse_seq, Xnew, Z, kern, f, *, full_cov=False, q_sqrt=None, white=False, return_Lm=False):

    f_mu = []
    f_var = []
    for kk in range(len(kern)):
        Kmn = kern[kk].K(Z, Xnew)
        if full_cov:
            Knn = kern[kk].K(Xnew)
        else:
            Knn = kern[kk].Kdiag(Xnew)

        f_k_mu, f_k_var = base_conditional_after_kernel_precalculation(Kmn, Lm_inverse_seq[kk], Knn, f[:, kk][:, None], full_cov=full_cov, q_sqrt=q_sqrt, white=white, return_Lm=return_Lm) # N x R, N x R or R x N x N

        f_mu.append(f_k_mu)
        f_var.append(f_k_var)

    return tf.transpose(tf.convert_to_tensor(f_mu)[:, :, 0]), tf.transpose(tf.convert_to_tensor(f_var)[:, :, 0])

def base_conditional_after_kernel_precalculation(Kmn, Lm_inverse_seq_kk, Knn, f, *, full_cov=False, q_sqrt=None, white=False, return_Lm=False):
    """
    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2;0,Kmm)
      p(g1) = N(g1;0,Knn)
      p(g1|g2) = N(g1;0,Knm)
    And
      q(g2) = N(g2;f,q_sqrt*q_sqrt^T)
    This method computes the mean and (co)variance of
      q(g1) = \int q(g2) p(g1|g2)
    :param Kmn: M x N
    :param Kmm: M x M
    :param Knn: N x N  or  N
    :param f: M x R
    :param full_cov: bool
    :param q_sqrt: None or R x M x M (lower triangular)
    :param white: bool
    :return: N x R  or R x N x N
    """
    # compute kernel stuff
    num_func = tf.shape(f)[1]  # R


    # Compute the projection matrix A

    A = tf.matmul(tf.transpose(Lm_inverse_seq_kk), Kmn)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.matmul(A, A, transpose_a=True)
        fvar = tf.tile(fvar[None, :, :], [num_func, 1, 1])  # R x N x N
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), 0)
        fvar = tf.tile(fvar[None, :], [num_func, 1])  # R x N

    # another backsubstitution in the unwhitened case
    if not white:
        # A = tf.linalg.triangular_solve(tf.transpose(Lm_inverse_seq_kk), A, lower=False)
        A = tf.matmul(tf.transpose(Lm_inverse_seq_kk), A)
        print('May have some problems with the non-white case')

    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # R x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = q_sqrt
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # R x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # R x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # R x N

    if not full_cov:
        fvar = tf.transpose(fvar)  # N x R
    if return_Lm:
        return fmean, fvar, Lm_inverse_seq_kk

    return fmean, fvar # N x R, R x N x N or N x R
