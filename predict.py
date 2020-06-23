import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import glob
import argparse
from PIL import Image
import tensorflow as tf

from datasets import CityscapesDataset
from models import ERFNet

from utils import read_image


def main(args):
    img_h_orig, img_w_orig = 1024, 2048  # original size of images in Cityscapes dataset
    img_h, img_w = args.img_height, args.img_width

    if not os.path.exists('test_segmentations'):
        os.makedirs('test_segmentations')
        print('test_segmentations directory created.')

    dataset = CityscapesDataset()

    print('Creating network and loading weights...')
    network = ERFNet(dataset.num_classes)

    # Initialize network weights
    inp_test = tf.random.normal(shape=(1, img_h, img_w, 3))
    out_test = network(inp_test, is_training=False)
    print('Shape of network\'s output:', out_test.shape)

    # Load weights and images from given paths
    weights_path = os.path.join(os.getcwd(), args.weights)
    image_paths = sorted(glob.glob(os.path.join(os.getcwd(), 'test_images', '*.png')))

    network.load_weights(weights_path)
    print('Weights from {} loaded correctly.'.format(weights_path))

    inference_times = []
    for image_path in image_paths:
        t0 = time.time()

        image = read_image(image_path, (img_h, img_w))
        x = tf.expand_dims(image, axis=0)  # (1, img_h, img_w, 3)
        
        y_pred_logits = network(x, is_training=False)  # (1, img_h, img_w, num_classes)
        y_pred_labels = tf.math.argmax(y_pred_logits[0], axis=-1, output_type=tf.int32)  # (img_h, img_w)
        y_pred_colors = tf.gather(dataset.class_colors, y_pred_labels)
        y_pred_colors = tf.cast(y_pred_colors, tf.uint8)  # (img_h, img_w, 3)
        y_pred_colors = tf.image.resize(y_pred_colors, (img_h_orig, img_w_orig), method='nearest')  # (img_h_orig, img_w_orig, 3)
        t1 = time.time()
        
        # Save segmentation
        save_path = image_path.replace('leftImg8bit', 'segmentation').replace('test_images', 'test_segmentations')
        segmentation = Image.fromarray(y_pred_colors.numpy())
        segmentation.save(save_path)
        
        print()
        print('Segmentation of image\n {}\nsaved in\n {}.'.format(image_path, save_path))
        inference_times.append(t1 - t0)
    
    mean_inference_time = sum(inference_times) / len(inference_times)
    print('\nAverage inference time: {:.3f} s'.format(mean_inference_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img_height', type=int, default=512, help='Image height after resizing')
    parser.add_argument('--img_width', type=int, default=1024, help='Image width after resizing')
    parser.add_argument('--weights', type=str, required=True, help='Relative path of network weights')

    args = parser.parse_args()
    main(args)
