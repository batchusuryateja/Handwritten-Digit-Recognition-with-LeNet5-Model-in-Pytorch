# Handwritten Digit Classification using CNN, MLP & LeNet 5
#### This project focuses on the classification of handwritten digits using three different models: a Multilayer Perceptron (MLP), a Convolutional Neural Network (CNN), and the LeNet-5 model. The goal is to compare the performance and effectiveness of these neural network architectures on the MNIST dataset, a large database of handwritten digits commonly used for training various image processing systems.

## Python Libraries Used:
##### Sklearn
##### OpenCV
##### PIL
##### NumPy
##### Pandas  

## Data Collection:

#### The MNIST dataset, a benchmark dataset for handwritten digit classification, is used in this project. It comprises 60,000 training images and 10,000 test images of handwritten digits from 0 to 9.

## Loading MNIST Dataset:

#### To load the MNIST dataset, we utilize the tensorflow.keras.datasets module, which provides a straightforward interface for fetching the dataset. The dataset is split into training and test sets.
![Load Data](https://github.com/batchusuryateja/Handwritten-Digit-Recognition-with-LeNet5-Model-in-Pytorch/raw/main/Images/Load%20Data.png)

## Displaying 5 Random Images in the Dataset
#### To get an overview of the dataset, we display 5 random images along with their corresponding labels.
![Load Data](https://github.com/batchusuryateja/Handwritten-Digit-Recognition-with-LeNet5-Model-in-Pytorch/blob/main/Images/Random.png)

## Preprocessing the Data
#### Preprocessing involves normalizing the image data to the range [0, 1] by dividing by 255. Additionally, for the MLP model, we flatten the 28x28 images into 784-dimensional vectors.

## Building the MLP Model
#### The Multilayer Perceptron (MLP) model is a fully connected neural network with an input layer, one or more hidden layers, and an output layer.
![Load Data](https://github.com/batchusuryateja/Handwritten-Digit-Recognition-with-LeNet5-Model-in-Pytorch/blob/main/Images/MLP.png)

## Building the CNN Model
#### The Convolutional Neural Network (CNN) model uses convolutional layers to capture spatial features of the images, followed by pooling layers and fully connected layers.
![Load Data](https://github.com/batchusuryateja/Handwritten-Digit-Recognition-with-LeNet5-Model-in-Pytorch/blob/main/Images/CNN.jpeg)

## Building the LeNet-5 Model
#### LeNet-5 is a classic CNN architecture designed specifically for digit recognition. It consists of two sets of convolutional and pooling layers, followed by fully connected layers.
![image](https://github.com/batchusuryateja/Handwritten-Digit-Recognition-with-LeNet5-Model-in-Pytorch/assets/171341053/18df991a-3d11-4a3f-9ba2-9e489bdda00f)


## Training the MLP Model
#### The MLP model is trained using the training dataset. The number of epochs and batch size can be adjusted as needed.
![Load Data](https://github.com/batchusuryateja/Handwritten-Digit-Recognition-with-LeNet5-Model-in-Pytorch/blob/main/Images/T%20MLP.png)

## Training the CNN Model
#### The CNN model is trained similarly, but the input data must be reshaped to include the channel dimension.
![Load Data](https://github.com/batchusuryateja/Handwritten-Digit-Recognition-with-LeNet5-Model-in-Pytorch/blob/main/Images/T%20CNN.png)

## Training the LeNet-5 Model
#### The LeNet-5 model is trained with the same reshaped data.
![image](https://github.com/batchusuryateja/Handwritten-Digit-Recognition-with-LeNet5-Model-in-Pytorch/assets/171341053/f7da15af-ceca-4b99-86d2-8c2e1f9cd023)

## Testing Model Accuracy
![Load Data](https://github.com/batchusuryateja/Handwritten-Digit-Recognition-with-LeNet5-Model-in-Pytorch/blob/main/Images/Model%20Accuracy.png)

## Prediction of MLP Model
![Load Data](https://github.com/batchusuryateja/Handwritten-Digit-Recognition-with-LeNet5-Model-in-Pytorch/blob/main/Images/P%20MLP.png)
## Prediction of CNN Model
![Load Data](https://github.com/batchusuryateja/Handwritten-Digit-Recognition-with-LeNet5-Model-in-Pytorch/blob/main/Images/P%20CNN.png)
## Prediction of LeNet-5 Model
![Load Data](https://github.com/batchusuryateja/Handwritten-Digit-Recognition-with-LeNet5-Model-in-Pytorch/blob/main/Images/P%20Net.png)
