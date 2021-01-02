import numpy as np

'''conda install -c conda-forge matplotlib'''
import matplotlib.pylab as plt


def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


def step_function_np(x):
    y = x > 0
    return y.astype(np.int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function_np(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.savefig('/stepFunction.png')
plt.show()
