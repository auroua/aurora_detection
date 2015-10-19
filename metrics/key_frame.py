__author__ = 'auroua'

import numpy as np

if __name__ == '__main__':
    matrix = np.load('/home/auroua/workspace/PycharmProjects/data/similary.npy').T
    matrix = matrix/matrix.max()
    print matrix
    value,vector = np.linalg.eig(matrix)
    print value.shape
    print value.max(), value.min()
    print vector.shape