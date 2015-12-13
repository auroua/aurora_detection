# encoding:UTF-8

import os
import cv2
import numpy as np
import math
from time import clock
import img_proc.sift as sift
from PIL import Image
import img_proc.dsift as dsift

# original file path  /home/auroua/workspace/PycharmProjects/data/N20040103G/
# test file path  /home/auroua/workspace/PycharmProjects/data/test/


def getFiles(path='/home/aurora/hdd/workspace/PycharmProjects/data/N20040103G/'):
    '''获取制定目录下的文件的绝对路径,带文件名'''
    filelist = []
    FileNames = os.listdir(path)
    if len(FileNames)>0:
       for fn in FileNames:
            fullfilename = os.path.join(path, fn)
            filelist.append(fullfilename)

    # 对文件名排序
    if len(filelist)>0:
        filelist.sort()
    return filelist


def showImg_url(url):
    ''' 显示制定url中的图片 '''
    img = cv2.imread(url)
    cv2.namedWindow('aurora')
    cv2.imshow('aurora img', img)
    cv2.waitKey(0)


def showImg(img):
    '''显示制定url中的图片'''
    cv2.namedWindow('aurora')
    cv2.imshow('aurora img', img)
    cv2.waitKey(0)


def getChannel(img):
    '''because the three channels have the same value,so it is sensible to use one single channel to
    represent the img'''
    b, g, r = cv2.split(img)
    return b


def getImg(url):
    '''返回图像数据'''
    img = cv2.imread(url)
    return img


def avg_channel(img):
    b, g, r = cv2.split(img)
    return (b+r+g)/3


def hist(img):
    b = getChannel(img)
    hists, bins = np.histogram(b.ravel(), 16, density=True)
    return hists


def generate_hist(file_lists):
    hist_matrix = np.zeros((len(file_lists), 16))
    for file_index in xrange(len(file_lists)):
        hist_matrix[file_index, :] = hist(getImg(file_lists[file_index]))
    return hist_matrix


def hist_adjacent_matrix(hist_matrix):
    """
       caculate the adjacent matrix without distance constraint
    """
    adjacent_matrix_v = np.zeros((hist_matrix.shape[0], hist_matrix.shape[0]))
    for i in xrange(hist_matrix.shape[0]):
        for j in xrange(hist_matrix.shape[0]):
            adjacent_matrix_v[i, j] = hist_adjacent(hist_matrix[i], hist_matrix[j])
            adjacent_matrix_v[i, j] *= caucal_gausses(i, j)
    np.save('/home/aurora/workspace/PycharmProjects/data/hist_adjacent_matrix', adjacent_matrix_v)
    print adjacent_matrix_v
    return adjacent_matrix_v


def hist_adjacent(hist1, hist2):
    value = np.sqrt(np.sum((hist1-hist2)**2))
    return 1 - (1/np.sqrt(2))*value


def hist_matrix_add_distance_constraint(weight_matrix):
    """
       add distance constraint when caculate the distance matrix
       if the distance of two frames is large than 10 then the weights is zero
    """
    adjacent_matrix_value = np.zeros((weight_matrix.shape[0], weight_matrix.shape[0]))
    for i in xrange(weight_matrix.shape[0]):
        for j in xrange(weight_matrix.shape[0]):
            v_distance = math.fabs(i-j)
            if v_distance <= 10:
                adjacent_matrix_value[i, j] = 0
            else:
                adjacent_matrix_value[i, j] = hist_adjacent(weight_matrix[i], weight_matrix[j])*caucal_gausses(i, j)
    np.save('/home/aurora/workspace/PycharmProjects/data/hist_adjacent_matrix_constraint', adjacent_matrix_value)
    return adjacent_matrix_value


def bgr2hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv_img


def hsv_hist_16bins(img):
    h, s, v = cv2.split(img)
    print img.shape
    h_bins = 8
    s_bins = 4
    v_bins = 4
    h_range = [0,180]
    s_range = [0,255]
    v_range = [0,255]
    print np.max(h), np.min(h)
    print np.max(s), np.min(s)
    print np.max(v), np.min(v)


def generate_vector(files):
    img_b = getImg(files[0])
    img_b = getChannel(img_b)
    data_vector = np.zeros((len(files), img_b.shape[0], img_b.shape[1]),dtype=img_b.dtype)
    for index, url in enumerate(files):
        img_b = getImg(url)
        img_b = getChannel(img_b)
        data_vector[index, :, :] = img_b
        # print (data_matrix[index, :, :] == img_b).all()
        # cv2.imshow('image',data_matrix[0, :, :])
        # cv2.imshow('original',img_b)
        # cv2.waitKey(0)
    return data_vector


def gen_matrix(img_vector):
    img_vector = img_vector.astype(dtype=np.int32)
    img_matrix = np.zeros((img_vector.shape[0], img_vector.shape[0]))
    for i in range(0, img_vector.shape[0]):
        for j in range(0,img_vector.shape[0]):
            img_matrix[i, j] = np.abs(img_vector[i, :, :] - img_vector[j, :, :]).sum()
    np.save('/home/auroua/workspace/PycharmProjects/data/sub_matrix_distance_2015_1211',img_matrix)
    return img_matrix


def caucal_gausses(i, j, sigma=3, d=20):
    d = -1*(1/20.)
    # return math.exp(d*((i-j)/3.)**2)
    return math.exp(d*((i-j)/33.)**2)


def gen_matrix_gausses(img_vector):
    #d=20, sigma = 3
    img_vector = img_vector.astype(dtype=np.int32)
    img_matrix = np.zeros((img_vector.shape[0], img_vector.shape[0]))
    w_gauss = 0
    for i in range(0, img_vector.shape[0]):
        for j in range(0, img_vector.shape[0]):
            v_distance = math.fabs(i-j)
            if v_distance <= 10:
                img_matrix[i, j] = 0
            else:
                img_matrix[i, j] = np.abs(img_vector[i, :, :] - img_vector[j, :, :]).sum()
                img_matrix[i, j] *= caucal_gausses(i, j)
    np.save('/home/aurora/hdd/workspace/PycharmProjects/data/sub_matrix_distance_2015_1211',img_matrix)
    return img_matrix


def sfit_desc_generator(path='/home/aurora/hdd/workspace/PycharmProjects/data/N20040103G/'):
    filelists = getFiles(path)
    for index, file in enumerate(filelists):
        sift.process_image(file, 'aurora'+str(index)+'.sift')


def sift_distance(desc1, desc2):
    matchs = sift.match_twosided(desc1, desc2)
    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])
    values = 0.0
    counts = 0
    for i, m in enumerate(matchs):
        # print 'the value of i '+str(i)+' the value of m is '+str(m)
        if m > 0:
            scores = np.dot(desc1[i, :], desc2[m, :].T)
            values += scores
            counts += 1
    if counts==0:
        return 0
    else:
        return values/counts


def sift_pan_desc_generator(path='/home/aurora/hdd/workspace/PycharmProjects/data/N20040103G/'):
    filelists = getFiles(path)
    feature = []
    for index, file in enumerate(filelists):
        sift.process_image(file, 'aurora'+str(index)+'.sift')
        feature.append('aurora'+str(index)+'.sift')
    return feature


def sift_feature_listnames_generator(path='/home/aurora/hdd/workspace/PycharmProjects/data/N20040103G/'):
    filelists = getFiles(path)
    feature = []
    for index, file in enumerate(filelists):
        feature.append('aurora'+str(index)+'.sift')
    return feature

def sift_matrix():
    featurelist = sift_feature_listnames_generator()
    imlist = getFiles('/home/aurora/hdd/workspace/PycharmProjects/data/N20040103G/')
    nbr_images = len(imlist)
    matchscores = np.zeros((nbr_images, nbr_images))
    for i in range(nbr_images):
        for j in range(i, nbr_images):
            # print 'comparing ', imlist[i], imlist[j]
            l1, d1 = sift.read_feature_from_file(featurelist[i])
            l2, d2 = sift.read_feature_from_file(featurelist[j])

            if d1.shape[0] == 0 or d2.shape[0] == 0:
                matchscores[i, j] = 0
            else:
                matches = sift.match_twosided(d1, d2)
                nbr_matches = sum(matches > 0)
                print 'number of matches = ', nbr_matches
                matchscores[i, j] = nbr_matches
    for i in range(nbr_images):
        for j in range(i + 1, nbr_images):
            matchscores[j, i] = matchscores[i, j]
    np.save('/home/aurora/hdd/workspace/PycharmProjects/data/aurora_img_matches_matrix_20151212', matchscores)
    print matchscores


def get_sift_distance_matrix_with_constraint(img_vectors):
    img_vector = img_vectors.astype(dtype=np.int32)
    img_matrix = np.zeros((img_vector.shape[0], img_vector.shape[0]))
    for i in range(0, img_vector.shape[0]):
        for j in range(0, img_vector.shape[0]):
            v_distance = math.fabs(i-j)
            if v_distance <= 10:
                img_matrix[i, j] = 0
            else:
                l1, d1 = sift.read_feature_from_file('aurora'+str(i)+'.sift')
                l2, d2 = sift.read_feature_from_file('aurora'+str(j)+'.sift')
                img_matrix[i, j] = sift_distance(d1, d2)
    np.save('/home/aurora/hdd/workspace/PycharmProjects/data/sift_distance_2015_1211_with_constraint',img_matrix)
    return img_matrix


def get_sift_distance_matrix_without_constraint(img_vectors):
    img_vector = img_vectors.astype(dtype=np.int32)
    img_matrix = np.zeros((img_vector.shape[0], img_vector.shape[0]))
    for i in range(0, img_vector.shape[0]):
        for j in range(0, img_vector.shape[0]):
            l1, d1 = sift.read_feature_from_file('aurora'+str(i)+'.sift')
            l2, d2 = sift.read_feature_from_file('aurora'+str(j)+'.sift')
            img_matrix[i, j] = sift_distance(d1, d2)
    np.save('/home/aurora/hdd/workspace/PycharmProjects/data/sift_distance_2015_1211_without_constraint',img_matrix)
    return img_matrix


def dsfit_desc_generator(filelist):
    filelists = getFiles()
    for index, file in enumerate(filelists):
        dsift.process_image_dsift(file, 'dsiftaurora'+str(index)+'.sift', 30, 10, True)



def get_dsift_distance_matrix_with_constraint(img_vectors):
    img_vector = img_vectors.astype(dtype=np.int32)
    img_matrix = np.zeros((img_vector.shape[0], img_vector.shape[0]))
    for i in range(0, img_vector.shape[0]):
        for j in range(0, img_vector.shape[0]):
            v_distance = math.fabs(i-j)
            if v_distance <= 10:
                img_matrix[i, j] = 0
            else:
                l1, d1 = sift.read_feature_from_file('dsiftaurora'+str(i)+'.sift')
                l2, d2 = sift.read_feature_from_file('dsiftaurora'+str(j)+'.sift')
                if d1.shape[0]<d2.shape[0]:
                    d1 = np.vstack((d1, np.zeros((d2.shape[0]-d1.shape[0], d1.shape[1]))))
                elif d1.shape[0]>d2.shape[0]:
                    d2 = np.vstack((d2, np.zeros((d1.shape[0]-d2.shape[0], d2.shape[1]))))
                temp_value = np.abs(d1-d2)
                distance = np.mean(temp_value)
                img_matrix[i, j] = distance
    np.save('/home/aurora/hdd/workspace/PycharmProjects/data/dsift_distance_2015_1211_with_constraint',img_matrix)
    return img_matrix


def get_dsift_distance_matrix_without_constraint(img_vectors):
    img_vector = img_vectors.astype(dtype=np.int32)
    img_matrix = np.zeros((img_vector.shape[0], img_vector.shape[0]))
    for i in range(0, img_vector.shape[0]):
        for j in range(0, img_vector.shape[0]):
            l1, d1 = sift.read_feature_from_file('dsiftaurora'+str(i)+'.sift')
            l2, d2 = sift.read_feature_from_file('dsiftaurora'+str(j)+'.sift')
            if d1.shape[0]<d2.shape[0]:
                d1 = np.vstack((d1, np.zeros((d2.shape[0]-d1.shape[0], d1.shape[1]))))
            elif d1.shape[0]>d2.shape[0]:
                d2 = np.vstack((d2, np.zeros((d1.shape[0]-d2.shape[0], d2.shape[1]))))

            temp_value = np.abs(d1-d2)
            distance = np.mean(temp_value)
            img_matrix[i, j] = distance
    np.save('/home/aurora/hdd/workspace/PycharmProjects/data/dsift_distance_2015_1211_without_constraint',img_matrix)
    return img_matrix


if __name__=='__main__':
    # # img = getImg('/home/aurora/workspace/PycharmProjects/data/N20040103G/N20040103G030001.bmp')
    # # showImg(img)
    # hist_matrixs = generate_hist(getFiles())
    # print hist_matrixs
    # # adjacent_matrix = hist_adjacent_matrix(hist_matrixs)
    # adjacent_matrix = hist_matrix_add_distance_constraint(hist_matrixs)
    # print adjacent_matrix
    # # data = generate_vector(getFiles())
    # # print '----------------------------------------'
    # # start = clock()
    # # print start
    # # # res_matrix = gen_matrix(data)
    # # res_matrix = gen_matrix_gausses(data)
    # # end = clock()
    # # print end
    # # print end - start
    # #
    # # print res_matrix.shape

    # 2015-12-11
    # vectors = generate_vector(getFiles())
    # sub_distance_matrix = gen_matrix_gausses(vectors)
    # print sub_distance_matrix

    # generator sift descriptor
    # sfit_desc_generator(getFiles())
    sift_matrix()

    # generator dsift descriptor
    # dsfit_desc_generator(getFiles())
    # sub_distance_matrix = get_dsift_distance_matrix_with_constraint(vectors)
    # print sub_distance_matrix
