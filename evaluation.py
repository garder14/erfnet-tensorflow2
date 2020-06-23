import tensorflow as tf


def compute_intersection_and_union_in_batch(y_true_labels, y_pred_labels, num_classes):
    # y_true_labels: (val_batch_size, img_h, img_w)
    # y_pred_labels: (val_batch_size, img_h, img_w)
    
    batch_intersection, batch_union = [], []  # for each class, store the sum of intersections and unions in the batch

    for class_label in range(num_classes - 1):  # ignore class 'other'
        true_equal_class = tf.cast(tf.equal(y_true_labels, class_label), tf.int32)
        pred_equal_class = tf.cast(tf.equal(y_pred_labels, class_label), tf.int32)

        intersection = tf.reduce_sum(tf.multiply(true_equal_class, pred_equal_class))  # TP (true positives)
        union = tf.reduce_sum(true_equal_class) + tf.reduce_sum(pred_equal_class) - intersection  # TP + FP + FN = (TP + FP) + (TP + FN) - TP
        
        batch_intersection.append(intersection)
        batch_union.append(union)

    return tf.cast(tf.stack(batch_intersection, axis=0), tf.int64), tf.cast(tf.stack(batch_union, axis=0), tf.int64)  # (19,)


def evaluate(dataset, network, val_batch_size, image_size):
    # Compute IoU on validation set (IoU = Intersection / Union)

    total_intersection = tf.zeros((19), tf.int64)
    total_union = tf.zeros((19), tf.int64)
    
    print('Evaluating on validation set...')
    num_val_batches = dataset.num_val_images // val_batch_size
    for batch in range(num_val_batches):
        x, y_true_labels = dataset.get_validation_batch(batch, val_batch_size, image_size)

        y_pred_logits = network(x, is_training=False)
        y_pred_labels = tf.math.argmax(y_pred_logits, axis=-1, output_type=tf.int32)

        batch_intersection, batch_union = compute_intersection_and_union_in_batch(y_true_labels, y_pred_labels, dataset.num_classes)
        total_intersection += batch_intersection
        total_union += batch_union

    iou_per_class = tf.divide(total_intersection, total_union)  # IoU for each of the 19 classes
    iou_mean = tf.reduce_mean(iou_per_class)  # Mean IoU over the 19 classes
    
    return iou_per_class, iou_mean
