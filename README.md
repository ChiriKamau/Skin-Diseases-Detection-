# Skin Disease Classification using CNN

This repository contains a Python script that uses a Convolutional Neural Network (CNN) to classify skin diseases. The model is trained on a dataset of images of skin diseases and can classify images into four classes: Acne, Actinic Keratosis (AK), Basal Cell Carcinoma (BCC), and Melanoma (MEL).

## Table of Contents

* [Introduction](#introduction)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Usage](#usage)
* [Results](#results)
* [Contributing](#contributing)
* [License](#license)


## Introduction

Skin cancer is a major health concern, and early detection is crucial for effective treatment. This project aims to develop a CNN-based model that can automatically classify skin diseases, aiding in early diagnosis and treatment.

## Dataset

The model is trained on a dataset of images of skin diseases. The dataset is organized into four folders, one for each class. The images are preprocessed and resized before being fed into the model.

The dataset used in this project is the HAM10000 dataset, which can be found here:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

The dataset is organized into the following folders:

* `ACNE`: Images of Acne
* `AK`: Images of Actinic Keratosis
* `BCC`: Images of Basal Cell Carcinoma
* `MEL`: Images of Melanoma

## Model Architecture

The model used in this project is a CNN with the following layers:

* Input Layer: Accepts images of size 224x224x3.
* Convolutional Layers: Multiple convolutional layers with ReLU activation functions are used to extract features from the images.
* Max Pooling Layers: Max pooling layers are used to reduce the spatial dimensions of the feature maps.
* Fully Connected Layers: Fully connected layers are used to classify the images.
* Output Layer: The output layer has four neurons, one for each class, with a softmax activation function.

The model is trained using the Adam optimizer and categorical cross-entropy loss function.

## Usage

To use the model, follow these steps:

1. Clone the repository.
2. Install the required dependencies.
3. Run the script.

The script will load the model and use it to classify images.

To classify a new image, pass the path to the image to the script as an argument. The script will print the predicted class of the image.

## Results

The model achieves an accuracy of around 80% on the test set. The model is also evaluated using precision, recall, and F1 score. The confusion matrix is also generated to visualize the performance of the model.

## Contributing

Contributions are welcome! If you find a bug or have a suggestion, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
