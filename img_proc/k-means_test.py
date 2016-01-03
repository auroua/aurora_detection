from scipy.cluster.vq import *
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
    url = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/selectedfontimages/'
    class1 = 1.5*np.random.randn(100, 2)
    class2 = np.random.randn(100, 2)+np.array([5, 5])
    features = np.vstack((class1, class2))

    centroids, variance = kmeans(features, 2)
    # print centroids
    # print variance

    code, distance = vq(features, centroids)
    # print code
    # print distance

    plt.figure()
    ndx = np.where(code==0)[0]
    print ndx
    print np.where(code==0)
    plt.plot(features[ndx, 0], features[ndx, 1], '*')
    ndx = np.where(code==1)[0]
    plt.plot(features[ndx, 0], features[ndx, 1], 'r.')
    plt.plot(centroids[:, 0], centroids[:, 1], 'yo')
    plt.axis('off')
    plt.show()
