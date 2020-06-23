import tensorflow as tf


def weighted_cross_entropy_loss(y_true_labels, y_pred_logits, class_weights):
    # y_true_labels: (batch_size, img_h, img_w)
    # y_pred_logits: (batch_size, img_h, img_w, num_classes)

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_labels, logits=y_pred_logits)  # (batch_size, img_h, img_w)

    weights = tf.gather(class_weights, y_true_labels)  # (batch_size, img_h, img_w)
    losses = tf.multiply(losses, weights)

    return tf.reduce_mean(losses)
    