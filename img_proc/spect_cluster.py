#encoding:UTF-8
from scipy.cluster.vq import *
import numpy as np
import img_tools
import numpy as np
import pca
from PIL import Image
import pickle
import matplotlib.pyplot as plt

if __name__=='__main__':
    url = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/selectedfontimages/a_selected_thumbs/'
    imlist = img_tools.get_imlist(url)
    imlist.sort()
    imnbr = len(imlist)
    url_pkl = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/selectedfontimages/a_pca_modes.pkl'
    # img size 25*25=625
    with open(url_pkl) as f:
        immean = pickle.load(f)   # size 625
        V = pickle.load(f)        # size 66*625    pca的特征向量  图像乘以特征向量得到pca投影

    # 66*625      original
    immatrix = np.array([np.array(Image.open(im)).flatten() for im in imlist], 'f')
    immean = immean.flatten()
    projected = np.array([np.dot(V[:40], immatrix[i]-immean) for i in range(imnbr)])
    n = len(projected)
    S = np.array([[np.sqrt(np.sum((projected[i]-projected[j])**2)) for i in range(n)] for j in range(n)], 'f')

    rowsum = np.sum(S, axis=0)
    D = np.diag(1/np.sqrt(rowsum))
    I = np.identity(n)
    L = I - np.dot(D, np.dot(S, D))

    U, sigma, V = np.linalg.svd(L)

    k = 5
    features = np.array(V[:k]).T

    features = whiten(features)
    centroids, distortion = kmeans(features, k)
    code, distance = vq(features, centroids)

    for c in range(k):
        ind = np.where(code==c)[0]
        plt.figure()
        for i in range(np.minimum(len(ind), 39)):
            imname = imlist[ind[i]]
            # print imname
            im = Image.open(imlist[ind[i]])
            plt.subplot(4, 10, i+1)
            plt.imshow(np.array(im))
            plt.axis('equal')
            plt.axis('off')
    plt.show()
