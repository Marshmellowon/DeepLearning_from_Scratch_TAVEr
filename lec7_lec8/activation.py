import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


x = np.arange(-2, 2, 0.1)

l1 = plt.plot(x, sigmoid(x), label="sigmoid")
l2 = plt.plot(x, tanh(x), label="tanh")
l3 = plt.plot(x, relu(x), label="ReLU")
plt.legend()
plt.show()
