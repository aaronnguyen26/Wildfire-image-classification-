# Wildfire-image-classification-

This project implements a Convolutional Neural Network (CNN) to detect wildfires from images. The model is trained on a dataset of images containing both "fire" and "nofire" classes.

Project Overview
The goal of this project is to build a deep learning model that can accurately classify images as either containing a wildfire or not. This can be a crucial component in early wildfire detection systems.

The project involves the following steps:

Data Loading and Preparation: Loading image data from a zip file stored in Google Drive, extracting it, and organizing it into training, validation, and testing sets.
Data Preprocessing: Applying transformations to the images, including resizing and normalization, to prepare them for the CNN model.
Model Definition: Designing a custom CNN architecture (FireCNN) for image classification.
Model Training: Training the CNN model using a binary cross-entropy loss function and an Adam optimizer.
Hyperparameter Tuning (Grid Search): Experimenting with different learning rates and dropout rates to find the optimal combination for improved model performance and to mitigate overfitting.
Evaluation: Evaluating the trained model's performance on the validation set.
Dataset
The dataset used for this project is assumed to be a zip file named wildfire.zip containing a directory structure with train, val, and test subdirectories, each containing fire and nofire image classes. The dataset is expected to be located in your Google Drive at /content/drive/My Drive/wildfire.zip.

Requirements
Python 3.x
PyTorch
Torchvision
Pillow
Matplotlib
Google Colab (for easy access to GPU and Google Drive)

