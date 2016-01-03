from scipy.cluster.vq import *
from scipy.misc import imresize
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

url = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/empire.jpg'
steps = 50
im = np.array(Image.open(url))
dx = im.shape[0]/steps
dy = im.shape[1]/steps
features = []
for x in range(steps):
    for y in range(steps):
        R = np.mean(im[x*dx:(x+1)*dx, y*dy:(y+1)*dy, 0])
        G = np.mean(im[x*dx:(x+1)*dx, y*dy:(y+1)*dy, 1])
        B = np.mean(im[x*dx:(x+1)*dx, y*dy:(y+1)*dy, 2])
        features.append([R, G, B])
features = np.array(features, 'f')

centroids, variance = kmeans(features, 10)
code, distance = vq(features, centroids)

codeim = code.reshape(steps, steps)
# 大小为原始图像大小  插值方式为临近点插值
codeim = imresize(codeim, im.shape[:2], interp='nearest')

plt.figure()
plt.imshow(codeim)
plt.show()
