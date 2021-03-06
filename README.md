# ERFNet - TensorFlow 2

This is an unofficial implementation of [ERFNet](http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf) for semantic segmentation on the [Cityscapes dataset](https://www.cityscapes-dataset.com/).

## Results

![](assets/test1.png)
![](assets/test2.png)
![](assets/test3.png)

The above predictions are produced by a network trained for 67 epochs that achieves a mean class IoU score of 0.7084 on the validation set. To get this model, I completed 70 epochs (almost 10 hours on a single Tesla P100 GPU) and selected the checkpoint with maximum validation score. The progression of this metric is shown below.

<img src="assets/iou_plot.png" width="65%">

The inference time on a Tesla P100 GPU is around 0.2 seconds per image.

## Software installation

Clone this repository:

```bash
git clone https://github.com/garder14/erfnet-tensorflow2.git
cd erfnet-tensorflow2/
```

Install the dependencies:

```bash
conda create -n tf-gpu tensorflow-gpu cudatoolkit=10.1
conda activate tf-gpu
pip install tensorflow_addons==0.10.0 Pillow==7.1.2
```

## Training

Before training the network, you need to download the Cityscapes dataset. For this purpose, create an account in [www.cityscapes-dataset.com](https://www.cityscapes-dataset.com/), and run the following command (indicating your username and password):

```bash
bash download_data.sh username password
```

To train the network, run this command:

```bash
python train.py --num_epochs 70 --batch_size 8 --evaluate_every 1 --save_weights_every 1
```

By default, training resumes from the latest saved checkpoint. If the checkpoints/ directory is missing, the training starts from zero.

## Inference

Run the following command to predict the semantic segmentation of every image in the test_images/ directory (note: results are saved in the test_segmentations/ directory). It is required to specify the location of the weights (you can use either the pretrained weights or any other weights saved during training in the saved_weights/ directory).

```bash
python predict.py --weights pretrained/pretrained.h5
```

## References

* [E. Romera et al., "ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation", 2017](http://www.robesafe.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)

* [Official PyTorch implementation of ERFNet](https://github.com/Eromera/erfnet_pytorch)
