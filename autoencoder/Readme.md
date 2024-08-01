
# Autoencoder from Scratch
==========================

This repository contains an implementation of an Autoencoder in Python, **built from scratch**. The Autoencoder is a type of neural network that is trained to copy its input to its output, and is often used for dimensionality reduction, anomaly detection, and generative modeling.

## Features
------------

* Implementation of a basic Autoencoder with one hidden layer
* Training using backpropagation with mean squared error loss
* Sigmoid activation function for hidden layer
* Derivative of sigmoid function for backpropagation

## Code Structure
-----------------

The code is organized into a single class `autoencoder` with the following methods:

### `__init__`
Initializes the Autoencoder with weights and biases

### `forward`
Computes the output of the Autoencoder for a given input

### `error`
Computes the mean squared error between the input and output

### `backpropagate`
Trains the Autoencoder using backpropagation

## Usage
---------

To use the Autoencoder, simply create an instance of the `autoencoder` class and call the `backpropagate` method to train the model. For example:
```python
import numpy as np

# Create an instance of the Autoencoder
W = [np.random.rand(4, 3), np.random.rand(3, 4)]
B = [np.random.rand(4, 1), np.random.rand(3, 1)]
ae = autoencoder(W, B)

# Train the Autoencoder
ae.backpropagate(X, lr=0.01, max_iter=1000)
```
