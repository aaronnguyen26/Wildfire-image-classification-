# Wildfire-image-classification-

Wildfire Detection using CNN
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
Installation
Clone the repository:
# Replace with your repository URL
git clone <your-repository-url>
cd <your-repository-directory>
Install dependencies:
pip install torch torchvision pillow matplotlib
Usage
Upload your dataset: Make sure your wildfire.zip file is uploaded to your Google Drive in the specified path (/content/drive/My Drive/wildfire.zip).
Open the Colab notebook: Open the provided Jupyter Notebook in Google Colab.
Run the cells: Execute the cells sequentially to:
Mount your Google Drive.
Extract the dataset.
Load and preprocess the data.
Define and train the CNN model.
Perform hyperparameter tuning (if included in the notebook).
Evaluate the model.
Model Architecture
The CNN model (FireCNN) consists of:

Three convolutional layers with ReLU activation and max pooling.
Two fully connected layers.
A dropout layer for regularization (in the FireCNN_WithDropout version).
The output layer uses a single neuron with no activation, as BCEWithLogitsLoss is used for binary classification.

Hyperparameter Tuning
The notebook includes code for a basic grid search over different learning rates and dropout rates to find the best combination for the model. The results of these trials are printed to the console.

Results
The training and validation loss and accuracy are plotted to visualize the model's performance during training. The best validation accuracy achieved during training is also reported.

Future Work
Experiment with different CNN architectures (e.g., transfer learning with pre-trained models like ResNet or VGG).
Implement more advanced data augmentation techniques.
Explore different optimizers and learning rate scheduling strategies.
Perform more extensive hyperparameter tuning.
Evaluate the model on a separate, unseen test dataset.
Integrate the model into a real-time wildfire detection system.
