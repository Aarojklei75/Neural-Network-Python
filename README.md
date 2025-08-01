# WORK IN PROGRESS
# Neural Network from Scratch – MNIST Digit Classifier

This project implements a basic feedforward neural network from scratch using **NumPy** for handwritten digit classification with the **MNIST** dataset. It includes support for various activation functions and training visualizations (cost, accuracy, prediction distribution).

## 🚀 Features

- Custom neural network implementation (no deep learning frameworks)
- Flexible architecture: supports hidden layers of any size
- Activation functions: `relu`, `sigmoid`, `tanh`, `leaky_relu`
- One-hot encoding and normalization
- Forward propagation, backpropagation, and softmax output
- Training visualization (cost and accuracy curves)
- Save/load models using `pickle`

## 🧠 Architecture

Example architecture used:
- Input Layer: 784 nodes (28x28 image)
- Hidden Layers: 128 → 32
- Output Layer: 10 nodes (digits 0-9)

## 🧰 Technologies

- Python 3
- NumPy
- Matplotlib
- Scikit-learn
- tqdm (for progress bars)

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/Neural-Network-Python.git
   cd Neural-Network-Python
