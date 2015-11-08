__author__ = 'auroua'
import numpy as np
import cv2
import python_ncut_lib as ncut

if __name__ == '__main__':
    # img_weights = np.load('/home/auroua/workspace/PycharmProjects/data/ncuts_weight.npy')
    tt = np.random.randn(30000, 30000)
    img_weights = tt+tt.T
    cv2.imshow('ttt', img_weights)
    eigval, eigvec = ncut.ncut(img_weights, 2)
    print eigval
    print eigvec

    cv2.waitKey(0)