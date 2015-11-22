import numpy as np
import matplotlib.pyplot as plt

x, y = np.random.normal(size=(2, 10000))

fig, ax = plt.subplots()
im = ax.hexbin(x, y, gridsize=20)
fig.colorbar(im, ax=ax)

fig, ax = plt.subplots()
H = ax.hist2d(x, y, bins=20)
print H
fig.colorbar(H[3], ax=ax)

plt.show()