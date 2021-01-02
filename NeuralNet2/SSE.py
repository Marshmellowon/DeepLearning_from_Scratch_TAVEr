import numpy as np
import matplotlib.pylab as plt
import math


def SSE(y, t):
    return 0.5 * np.sum((y - t) ** 2)


t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
y2 = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.6, 0.3, 0.0, 0.0])

print(SSE(y, t))
SSE(y2, t)

x = np.arange(-10, 10, 1)
fx = -math.log2((1 / 25) * x)

plt.plot(x, y)
plt.show()
