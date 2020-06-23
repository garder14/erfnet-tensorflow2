import os
import glob
import random
import tensorflow as tf

from utils import read_image, read_segmentation


class CityscapesDataset:
    def __init__(self):
        self.image_paths = sorted(glob.glob(os.path.join(os.getcwd(), 'data', 'images', 'train', '*', '*.png')))
        self.segmentation_paths = sorted(glob.glob(os.path.join(os.getcwd(), 'data', 'segmentations', 'train', '*', '*labelIds.png')))
        self.num_images = len(self.image_paths)

        self.val_image_paths = sorted(glob.glob(os.path.join(os.getcwd(), 'data', 'images', 'val', '*', '*.png')))
        self.val_segmentation_paths = sorted(glob.glob(os.path.join(os.getcwd(), 'data', 'segmentations', 'val', '*', '*labelIds.png')))
        self.num_val_images = len(self.val_image_paths)

        self.class_names = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 
                            'traffic light', 'traffic sign', 'vegetation', 'terrain', 
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 
                            'motorcycle', 'bicycle', 'other']  # class 'other' includes the rest of classes
        self.name2label = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)  # 20

        self.class_colors = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), 
                             (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), 
                             (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), 
                             (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), 
                             (0, 0, 230), (119, 11, 32), (0, 0, 0)]

        # As ids are used in the 'labelIds' annotation files, we need the class label (0 to 19) for every id (0 to 34).
        self.id2label = tf.constant([19, 19, 19, 19, 19, 19, 19, 0, 1, 19, 19, 2, 3, 
                                     4, 19, 19, 19, 5, 19, 6, 7, 8, 9, 10, 11, 12, 
                                     13, 14, 15, 19, 19, 16, 17, 18, 19], tf.int32)

        # Weight for every class (used to compute the loss). Under-represented classes have a larger weight. Note: output of compute_class_weights()
        self.class_weights = [2.602, 6.707, 3.522, 9.877, 9.685, 9.398, 10.288, 9.969, 
                                4.336, 9.454, 7.617, 9.405, 10.359, 6.373, 10.231, 10.262, 
                                10.264, 10.394, 10.094, 0.0]  # class 'other' has weight 0 in order to disregard its pixels


    def get_training_batch(self, batch_id, batch_size, image_size):
        start_id = batch_id * batch_size
        end_id = start_id + batch_size

        batch_images = [read_image(self.image_paths[i], image_size) for i in range(start_id, end_id)]
        batch_segmentations = [read_segmentation(self.segmentation_paths[i], image_size, self.id2label) for i in range(start_id, end_id)]

        # Data augmentation
        for i in range(batch_size):
            if random.random() > 0.5:  # horizontal flip with probability 0.5
                batch_images[i] = tf.reverse(batch_images[i], axis=[1])
                batch_segmentations[i] = tf.reverse(batch_segmentations[i], axis=[1])
            batch_images[i] = tf.image.random_brightness(batch_images[i], max_delta=0.08)  # random brightness
            batch_images[i] = tf.image.random_contrast(batch_images[i], lower=0.95, upper=1.05)  # random contrast
            
        x = tf.stack(batch_images, axis=0)  # (batch_size, img_h, img_w, 3)
        y_true_labels = tf.stack(batch_segmentations, axis=0)  # (batch_size, img_h, img_w)
        return x, y_true_labels
    

    def get_validation_batch(self, batch_id, val_batch_size, image_size):
        start_id = batch_id * val_batch_size
        end_id = start_id + val_batch_size

        batch_images = [read_image(self.image_paths[i], image_size) for i in range(start_id, end_id)]  # list of images
        batch_segmentations = [read_segmentation(self.segmentation_paths[i], image_size, self.id2label) for i in range(start_id, end_id)]  # list of list of segmentations
            
        x = tf.stack(batch_images, axis=0)  # (batch_size, img_h, img_w, 3)
        y_true_labels = tf.stack(batch_segmentations, axis=0)  # (batch_size, img_h, img_w)
        return x, y_true_labels


    def shuffle_training_paths(self):
        # Training image and segmentation paths are shuffled in unison
        aux = list(zip(self.image_paths, self.segmentation_paths))
        random.shuffle(aux)
        self.image_paths, self.segmentation_paths = zip(*aux)


    def compute_class_weights(self, image_size):
        num_pixels_per_class = [0] * self.num_classes  # store the number of pixels for each class

        for i, segmentation_path in enumerate(self.segmentation_paths):
            segmentation = read_segmentation(segmentation_path, image_size, self.id2label)
            for class_label in range(self.num_classes):
                num_pixels = tf.reduce_sum(tf.cast(tf.equal(segmentation, class_label), tf.int32))
                num_pixels_per_class[class_label] += num_pixels
            if (i + 1) % 100 == 0:
                print('{}/{}'.format(i+1, self.num_images))

        class_probs = tf.divide(num_pixels_per_class[:-1], tf.reduce_sum(num_pixels_per_class[:-1]))
        class_weights = 1 / tf.math.log(1.1 + class_probs)
        return class_weights
