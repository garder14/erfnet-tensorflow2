import tensorflow as tf


def normalize_image(img):  # map pixel intensities to float32 in [-1, 1]
    return tf.cast(img, tf.float32) / 127.5 - 1.0


def unnormalize_image(img):  # map pixel intensities to uint8 in [0, 255]
    img = (img + 1.0) * 127.5
    img = tf.clip_by_value(img, 0.0, 255.0)
    return tf.cast(img, tf.uint8)


def read_image(path, image_size):  # read image, and resize it to image_size
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3, dtype=tf.uint8)
    img = normalize_image(img)
    img = tf.image.resize(img, image_size, method='bilinear')  # (image_size[0], image_size[1], 3)
    return img


def read_segmentation(path, image_size, id2label):  # read segmentation (class label for each pixel), and resize it to image_size
    seg = tf.io.read_file(path)
    seg = tf.image.decode_png(seg, channels=1, dtype=tf.uint8)
    seg = tf.image.resize(seg, image_size, method='nearest')  # resize with 'nearest' method to avoid creating new classes
    seg = tf.squeeze(seg)
    seg = tf.gather(id2label, tf.cast(seg, tf.int32))  # (image_size[0], image_size[1])
    return seg
