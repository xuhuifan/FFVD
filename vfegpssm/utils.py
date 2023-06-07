import tensorflow as tf


def get_rand(x, full_cov=False):
    mean = x[0]
    var = x[1]
    if full_cov:
        chol = tf.linalg.cholesky(var + tf.eye(tf.shape(mean)[0], dtype=tf.float64)[None, :, :] * 1e-7)
        rnd = tf.transpose(tf.squeeze(tf.matmul(chol, tf.random.normal(tf.shape(tf.transpose(mean)), dtype=tf.float64)[:, :, None])))
        return mean + rnd
    return mean + tf.random.normal(tf.shape(mean), dtype=tf.float64) * tf.sqrt(var)
