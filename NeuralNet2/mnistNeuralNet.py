import sys
import os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
import matplotlib.pylab as plt
from PIL import Image

'''conda install -c conda-forge pickle5'''
import pickle


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("../dataset/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def sigmpoid(x):
    return 1 / (1 + np.exp(-x))


def predict(network, x):
    w1, w2, w3 = network['w1'], network['w2'], network['w3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmpoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmpoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = sigmpoid(a3)

    return y


x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
