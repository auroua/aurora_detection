__author__ = 'auroua'
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
sub1 = fig.add_subplot(1, 1, 1)

sub1.hist(np.random.normal(0, 3, size=(10000,)), bins=30, color='b', alpha= 0.5)

plt.show()