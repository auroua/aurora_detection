#encoding:UTF-8
__author__ = 'auroua'

import numpy as np
import cv2

if __name__ == '__main__':
    matrix = np.load('/home/auroua/workspace/PycharmProjects/data/similary_gausses.npy').T
    print matrix.shape
    print matrix.size
    print matrix.max(),matrix.min()
    print '--------------------------------'

    matrix = matrix/matrix.max()
    D = np.diag(1/matrix.mean(axis = 1))
    D = np.sqrt(D)
    L = np.eye(matrix.shape[0]) - np.dot(np.dot(D, matrix), D)

    temp_matrix = 1 - matrix
    img_resize = cv2.resize(temp_matrix,(512,512))
    # cv2.imshow('img',matrix)
    cv2.imshow('img',img_resize)
    cv2.waitKey(0)
    # value, vector = np.linalg.eig(L)
    # print value[::-1]
    # print value.max(),value.min()
    # print value.searchsorted(0.005)