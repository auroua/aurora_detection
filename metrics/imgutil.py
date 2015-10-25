#encoding:UTF-8
__author__ = 'auroua'

import os
import cv2
import numpy as np
from time import clock

# original file path  /home/auroua/workspace/PycharmProjects/data/N20040103G/
# test file path  /home/auroua/workspace/PycharmProjects/data/test/

def getFiles(path='/home/auroua/workspace/PycharmProjects/data/N20040103G/'):
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
    # img = avg_channel(img)
    hists, bins = np.histogram(img.ravel(), 256, [0, 256])
    return hists

def bgr2hsv(img):
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
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

# def gen_matrix(img_vector):
#     # memory space is not enough  memory-error exception
#     # img_vector1 = img_vector[np.newaxis, :, :, :]
#     # img_vector2 = img_vector[:, np.newaxis, :, :]
#     # temp = img_vector1-img_vector2
#     # return np.abs(img_vector1-img_vector2).sum(axis=3).sum(axis=2)
#     # the running time of this programm nearly three hours
#
#     img_vector = img_vector.astype(dtype=np.int32)
#     gpu_img_vector = cm.CUDAMatrix(img_vector)
#
#     temp_vector = np.zeros((img_vector.shape))
#     result_vector = np.zeros((img_vector.shape[0], img_vector.shape[1]))
#     gpu_result_vector = cm.CUDAMatrix(result_vector)
#     final_result_vector = np.zeros((img_vector.shape[0], img_vector.shape[0]))
#     gpu_final_result_vector = cm.CUDAMatrix(final_result_vector)
#
#     img_matrix = np.zeros((img_vector.shape[0], img_vector.shape[0]))
#
#     gpu_temp_vector = cm.CUDAMatrix(temp_vector)
#     for i in range(0, img_vector.shape[0]):
#         base_img = img_vector[i, :, :]
#         base_img_3d = base_img[np.newaxis, :, :]
#         temp_vector = np.tile(base_img_3d, (temp_vector.shape[0], 1, 1))
#         print temp_vector.shape
#         gpu_base_img = cm.CUDAMatrix(temp_vector)
#         print gpu_base_img.shape
#
#         # img_matrix[:, i:i+1] = np.abs(gpu_img_vector - base_img[np.newaxis, :, :]).sum(axis=2).sum(axis=1,keepdims=True)
#         gpu_img_vector.subtract(gpu_base_img,target = gpu_base_img)
#         cm.abs(gpu_base_img, target=gpu_base_img)
#         print gpu_base_img.shape
#         gpu_base_img.sum(axis=2, target=gpu_result_vector)
#         gpu_result_vector.sum(axis=1, target=gpu_final_result_vector)
#         gpu_final_result_vector.copy_to_host()
#         print gpu_final_result_vector.shape
#         print gpu_final_result_vector.max(), gpu_final_result_vector.min()
#
#         img_matrix[:, i:i+1] = np.abs(gpu_img_vector - base_img_3d).sum(axis=2).sum(axis=1,keepdims=True)
#     np.save('/home/auroua/workspace/PycharmProjects/data/similary',img_matrix)
#     return img_matrix

# old version
# def gen_matrix(img_vector):
#     img_vector = img_vector.astype(dtype=np.int32)
#     img_matrix = np.zeros((img_vector.shape[0], img_vector.shape[0]))
#     for i in range(0, img_vector.shape[0]):
#         img_matrix[:, i:i+1] = np.abs(img_vector - img_vector[i, :, :,][np.newaxis, :, :]).sum(axis=2).sum(axis=1,keepdims=True)
#     np.save('/home/auroua/workspace/PycharmProjects/data/similary',img_matrix)
#     return img_matrix

def gen_matrix(img_vector):
    img_vector = img_vector.astype(dtype=np.int32)
    img_matrix = np.zeros((img_vector.shape[0], img_vector.shape[0]))
    for i in range(0, img_vector.shape[0]):
        for j in range(0,img_vector.shape[0]):
            img_matrix[i, j] = np.abs(img_vector[i, :, :] - img_vector[j, :, :]).sum()
            # print img_matrix[i, j]
    np.save('/home/auroua/workspace/PycharmProjects/data/similary',img_matrix)
    return img_matrix


if __name__=='__main__':
    # /home/auroua/workspace/lena.png
    # img = getImg('/home/auroua/workspace/PycharmProjects/data/N20040103G/N20040103G030001.bmp')
    # img2 =  img.sum(axis=2)
    # # cv2.imshow('original',img)
    # cv2.imshow('original2',img2)
    # cv2.waitKey(0)
    # img = getImg('/home/auroua/workspace/lena.png')
    # # showImg(img)
    # hsv_img = bgr2hsv(img)
    # hsv_hist_16bins(hsv_img)

    data = generate_vector(getFiles())
    print '----------------------------------------'
    start = clock()
    print start
    res_matrix = gen_matrix(data)
    end = clock()
    print end
    print end - start

    print res_matrix
    # print data.shape
    # a = data[0, :, :] - data[1, :, :]
    # # print np.abs(a).sum(axis=1).sum(axis=0)
    # print data[0, 34, 101] - data[1, 34, 101]
    # b = data[1, :, :] - data[0, :, :]
    # # print (data[1, :, :] == data[1, :, :]).all()
    # # print np.abs(b).sum(axis=1).sum(axis=0)
    # print data[1, 34, 101] - data[0, :, :]
    # flag = np.abs(a) == np.abs(b)
    # a = np.abs(a)
    # b = np.abs(b)
    # # 34 101 255 1
    # # 35 99 1 255
    # # 35 100 255 1
    # # 35 101 254 2
    # # 36 98 2 254
    # # 36 99 1 255
    #
    # print a[34, 101],b[34, 101]
    # print data[0, 34, 101], data[1, 34, 101]

    # for i in range(a.shape[0]):
    #     for j in range(a.shape[1]):
    #         if a[i,j]!=b[i,j]:
    #             print i,j,a[i,j],b[i,j]
    #         if j>100:
    #             break
    #
    #     if i>100:
    #         break
    #
    # np.save('/home/auroua/workspace/PycharmProjects/data/flags',flag)
    # print a[np.logical_not(flag)]


    # print end - start
    # print res_matrix.shape
    # print res_matrix
    # cv2.imshow('img0',data[0, :, :])
    # cv2.imshow('img1',data[1, :, :])
    # cv2.imshow('img2',data[2, :, :])
    # cv2.imshow('img3',data[4311, :, :])
    # cv2.waitKey(0)


