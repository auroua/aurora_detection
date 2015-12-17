import numpy as np
from imgutil import getFiles
from imgutil import generate_vector, gen_matrix_gausses
from sklearn.utils.graph import graph_laplacian
from numpy.linalg import eig

def get_index(array1, index):
    results = np.diff(array1)
    nonzero_index = np.flatnonzero(results)
    nonzero_index += 1
    # np.savetxt('/home/aurora/workspace/PycharmProjects/data/test%i.out' %index, array1, delimiter=',')
    return nonzero_index


def caculate_key_frame(filenames):
    vector_data = generate_vector(filenames)
    adjacent_matrix = gen_matrix_gausses(vector_data)
    lap_m, diag_m = graph_laplacian(adjacent_matrix, normed=True, return_diag=True)
    w, v = eig(lap_m)
    w = np.sort(w)[::-1]
    cum_w = np.cumsum(w)
    print cum_w
    print np.diff(cum_w)


def generate_key_frame(indexs):
    # for index in xrange(len(lists)):
    keyframes = {}
    files = getFiles()
    for k in xrange(len(indexs)):
        print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
        nums = indexs[k].shape[0]
        keyframes[nums] = []
        for j in xrange(nums):
            if j == 0:
                temp_matrixs = files[0:indexs[k][j]]
            else:
                temp_matrixs = files[indexs[k][j-1]:indexs[k][j]]
            caculate_key_frame(temp_matrixs)


if __name__ == '__main__':
    # matrix = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/hist_ncuts_constraint_sigma_33.npy')
    matrix = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/sub_matrix_distance_2015_1211.npy')
    print matrix.shape
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