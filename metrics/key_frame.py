import numpy as np
import cv2
from imgutil import getFiles


def get_index(array1, index):
    results = np.diff(array1)
    nonzero_index = np.flatnonzero(results)
    nonzero_index += 1
    # np.savetxt('/home/aurora/workspace/PycharmProjects/data/test%i.out' %index, array1, delimiter=',')
    return nonzero_index


def generate_key_frame(indexs):
    # for index in xrange(len(lists)):
    files = getFiles()
    for k in xrange(len(indexs)):
        for j in xrange(indexs[k].shape[0]):
            temp_matrixs = files[indexs[k][j-1]:indexs[k][j]]
            print files[indexs[k][j]]

if __name__ == '__main__':
    matrix = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/ncuts_results_sigma_33.npy')
    matrix_index = []
    for i in xrange(matrix.shape[1]):
        temp = get_index(matrix[0, i, :], i)
        matrix_index.append(temp)
    generate_key_frame(matrix_index)

    # matrix = matrix/matrix.max()
    # D = np.diag(1/matrix.mean(axis = 1))
    # D = np.sqrt(D)
    # L = np.eye(matrix.shape[0]) - np.dot(np.dot(D, matrix), D)
    #
    # temp_matrix = 1 - matrix
    # img_resize = cv2.resize(temp_matrix,(512,512))
    # # cv2.imshow('img',matrix)
    # cv2.imshow('img', img_resize)
    # cv2.waitKey(0)
    # value, vector = np.linalg.eig(L)
    # print value[::-1]
    # print value.max(),value.min()
    # print value.searchsorted(0.005)