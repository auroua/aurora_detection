# encoding:UTF-8
import pickle
import numpy as np
from scipy.cluster.vq import *
import img_tools
from PIL import Image
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
    projected = np.array([np.dot(V[:60], immatrix[i]-immean) for i in range(imnbr)])
    # 白化  去相关性  相当于去冗余
    projected = whiten(projected)
    centroids, distortion = kmeans(projected, 4)
    code, distnce = vq(projected, centroids)
    for k in range(4):
        ind = np.where(code == k)[0]
        plt.figure()
        plt.gray()
        for i in range(np.minimum(len(ind), 40)):
            plt.subplot(4, 10, i+1)
            plt.imshow(immatrix[ind[i]].reshape((25, 25)))
            plt.axis('off')
    plt.show()