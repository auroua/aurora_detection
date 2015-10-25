#encoding:UTF-8
__author__ = 'auroua'

import numpy as np

if __name__ == '__main__':
    matrix = np.load('/home/auroua/workspace/PycharmProjects/data/similary.npy').T
    print matrix.shape
    print matrix.size
    print matrix.max(),matrix.min()

    # matrix = matrix/matrix.max()
    # D = np.diag(1/matrix.mean(axis = 1))
    # D = np.sqrt(D)
    # L = np.eye(matrix.shape[0]) - np.dot(np.dot(D, matrix), D)
    #
    # value, vector = np.linalg.eig(L)
    # value
    # print value[::-1]
    # print value.max(),value.min()
    # print value.searchsorted(0.005)