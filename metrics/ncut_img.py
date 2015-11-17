__author__ = 'auroua'
import numpy as np
import cv2
import python_ncut_lib as ncut

if __name__ == '__main__':
    img_weights = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/ncuts_weight.npy')
    eigval, eigvec = ncut.ncut(img_weights, 2)
    print eigval
    print eigvec