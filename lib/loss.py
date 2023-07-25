import tensorflow as tf

from config import triplet_loss_alpha


def simple_triplet_loss(y_true, y_pred):
    # shape of y_pred: batch_size x 3 x embedding_size
    anchor_pos_dists = tf.reduce_sum(tf.square(y_pred[:, 0, :] - y_pred[:, 1, :]), axis=-1)
    anchor_neg_dists = tf.reduce_sum(tf.square(y_pred[:, 0, :] - y_pred[:, 2, :]), axis=-1)
    loss = tf.maximum(0.0, anchor_pos_dists - anchor_neg_dists + triplet_loss_alpha)
    return tf.reduce_mean(loss)
