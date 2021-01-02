import numpy as np
import sys
import os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


def CEE(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

        batch_size_ = y.shape[0]
        return -np.sum(t * np.log(y + 1e-7)) / batch_size_


print(CEE(x_batch, t_batch))
