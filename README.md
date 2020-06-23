# EfficientDet - TensorFlow 2

This is an unofficial implementation of [ERFNet](http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf) for semantic segmentation on [Cityscapes dataset](https://www.cityscapes-dataset.com/).

## Results

segmentations_plot

To achieve these results, I have trained the network for 70 epochs on a single Tesla P100 GPU (~10 hours), and have used the weights with the largest IoU score on validation set (0.7084 after epoch 67).

loss_plot

The inference time on Tesla P100 GPU is ~0.2 seconds/image.

## Software installation

Clone this repository:

```bash
git clone https://github.com/garder14/erfnet-tensorflow2.git
cd erfnet-tensorflow2.git
```

Install the dependencies:

```bash
conda create -n tf-gpu tensorflow-gpu cudatoolkit=10.1
conda activate tf-gpu
pip install matplotlib=3.2.2 tensorflow_addons=0.10.0 Pillow=7.1.2
```
