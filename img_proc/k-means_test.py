from scipy.cluster.vq import *
import numpy as np

if __name__=='__main__':
    class1 = 1.5*np.random.randn(100, 2)
    class2 = np.random.randn(100, 2)+np.array([5, 5])
    features = np.vstack((class1, class2))

    centroids, variance = kmeans(features, 2)
    print centroids
    print variance

    code, distance = vq(features, centroids)
    print code
    print distance