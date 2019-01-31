# Processing of Data Science Bowl 2018 kaggle dataset with using U-net neural network

## Overview
Notebook "unet-dsb" and accompanied python files contain implementation of image semantic segmentation using Keras/Tensorflow frameworks and its application to Data Science Bowl 2018 kaggle dataset.

## Dataset
Following dataset is used: [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018) from [kaggle](https://www.kaggle.com). This dataset contains a large number of segmented nuclei images. The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (brightfield vs. fluorescence). 
### Image sizes
The dataset contains images of different sizes. The smallest image is 256x256 pixels. This size was takes as a basis. Images larger as the basis are tiled to the tiles of 256x256 pixels. The tiles are created in the way  that they always overlap. It is taken in account by the combination of the images in order to get an image of the original size. 
![](images/dsb-tiling.png)

### Image brightness
Dataset contains images of different brightness. Images with high brightness show nuclei as dark spots, images with low brightness show nuclei as bright spots. In order to mitgate the difference, images with mean intensity more than 80 are inverted. All images are converted to grayscale
![](images/dsb.png)

### Training, validation and test sets

In order to achieve equal distribution of images with different brightness, density of nuclei, we compute image standard statistics: min, max, mean and standard deviation and execute clustering algoritm to split the dataset to several clusters according to their statistics. For details see [Dataset notebook](data/data-science-bowl.ipynb) Training, validation and test datasets are created with equal proportions of different clusters.
Image below shows several rows of images. Each row represents a cluster.
![](images/dsb-clusters.png)

## Network architecture
Originaly U-net neural network architecture was proposed here: [U-Net: Convolutional Networks for Biomedical
Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf). Symmetrical architecture used in https://github.com/zhixuhao/unet is used for the training since sizes of input images and output masks are the same. It is achieve by usin of 'same' padding instead of 'valid'.

![](images/unet-dsb.png)

### Training
Training of the model was done using Adam optimizer with learning rate 1.e-4. 
Picture below shows evolution of the prediction masks generated during the training on different steps. The last row shows manual annotation.

![](images/dsb-masks.png)

Loss plot

![](images/unet-dsb-loss.png)

Accuracy plot

![](images/unet-dsb-acc.png)

Pixel difference plot

![](images/unet-dsb-pix.png)

## Results

After 3 epochs 4000 steps each, the training loss dropped to 0.006, validation loss dropped to 0.14. Accuracy of the training dataset achieved 0.9975, accuracy of the validation dataset - 0.98.
Picture below shows result of prediction of several images from the test dataset (first row is image, second - prediction, third - annotation) :

![](images/unet-dsb-test.png)

Error distribution over the test dataset. As the error here we take percentage of wrongly predicted pixels.

![](images/unet-dsb-test.png)

Following picture shows prediction, annotation and pixel difference between prediction and annotation for images with the largest pixel difference between prediction and annotation (more than 6%):

![](images/unet-dsb-diff.png)

Despite of relatively large amount of pixel different in annotation and predicted mask, quality of the generated segmentation is not so bad. Most difference appear on diffuse nucleus border and on overlapping nuclei. Nevertheless all nuclei were detected.