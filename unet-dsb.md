# Processing of Data Science Bowl 2018 kaggle dataset with using U-net neural network

## Overview
Notebook "unet-dsb" and accompanied python files contain implementation of image semantic segmentation using Keras/Tensorflow frameworks and its application to Data Science Bowl 2018 kaggle dataset.

## Dataset
Following dataset is used: [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018) from [kaggle](https://www.kaggle.com). This dataset contains a large number of segmented nuclei images. The images were acquired under a variety of conditions and vary in the cell type, magnification, and imaging modality (brightfield vs. fluorescence). 
 
## Network architecture
Originaly U-net neural network architecture was proposed here: [U-Net: Convolutional Networks for Biomedical
Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf). Symmetrical architecture used in https://github.com/zhixuhao/unet is used for the training since sizes of input images and output masks are the same. It is achieve by usin of 'same' padding instead of 'valid'.

![](images/unet-dsb.png)
