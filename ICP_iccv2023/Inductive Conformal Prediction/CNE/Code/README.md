# UNet and DeepLabV3 for Semantic Segmentation with Monte Carlo Dropout to estimate Epistemic Uncertainty

## Importing Libraries

The script begins by importing necessary libraries such as PyTorch, torchvision, PIL, pandas, os, tifffile, cv2, tqdm, and numpy. The random seed is set to 78 for reproducibility.

## Data Settings

Paths to the CSV file containing image file names, input image folder, target segmentation map folder, training dataset ratio, and batch size are defined.

## Transformations

A transformation pipeline function is defined to convert input images to PyTorch tensors.

## Custom Dataset

A custom PyTorch Dataset class is implemented to handle loading and preprocessing of images and target segmentation maps.

## UNet Model Settings

The UNet model is defined with specific architecture details including convolutional and deconvolutional blocks, max pooling, upsampling, and dropout layers.

## DeepLabV3 Model Settings

The DeepLabV3 model architecture is specified with features like atrous convolution, dilated convolutions, and ASPP modules.

## Training Settings

Functions are defined to set up the models for training, including moving models to the device, initializing optimizers, and setting loss functions.

## Monte Carlo Dropout

Monte Carlo Dropout is implemented for both UNet and DeepLabV3 models to generate multiple predictions for uncertainty quantification.

## Standard Deviation Calculation

The standard deviation of Monte Carlo Dropout predictions is calculated for each class in both models and scaled to -1 to 1. Mean standard deviation values are computed for each class across all images.

By integrating both UNet and DeepLabV3 models with Monte Carlo Dropout, the script allows for comparative analysis of uncertainty quantification in semantic segmentation tasks.
