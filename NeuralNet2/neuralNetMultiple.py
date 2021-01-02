import numpy as np
import matplotlib.pylab as plt

x = np.arange(-100, 100, 0.1)
y = -2 * (x ** 4) - 0.5 * (x ** 3) + (-4) * x + 1

plt.plot(x, y)
plt.show()
