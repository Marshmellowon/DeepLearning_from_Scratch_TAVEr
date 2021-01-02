import numpy as np
import matplotlib.pylab as plt


def sigmpoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmpoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.savefig('sigmoid.png')
plt.show()
