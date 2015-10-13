#encoding:UTF-8
__author__ = 'auroua'
__version__ = 0.1

import numpy as np
import cv2
import os
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# /home/auroua/workspace/PycharmProjects/data/N20040103G/
# test url /home/auroua/workspace/PycharmProjects/data/test/

def showImg(url):
    '''显示制定url中的图片'''
    img = cv2.imread(url)
    cv2.namedWindow('aurora')
    cv2.imshow('aurora img',img)
    cv2.waitKey(0)

def getImg(url):
    '''返回图像数据'''
    img = cv2.imread(url)
    return img

def avg_channel(img):
    b,g,r = cv2.split(img)
    return (b+r+g)/3

def getFiles(path):
    '''获取制定目录下的文件的绝对路径,带文件名'''
    filelist = []
    FileNames=os.listdir(path)
    if (len(FileNames)>0):
       for fn in FileNames:
            fullfilename=os.path.join(path,fn)
            filelist.append(fullfilename)

    #对文件名排序
    if (len(filelist)>0):
        filelist.sort()
    return filelist

if __name__=='__main__':
    filenames = getFiles('/home/auroua/workspace/PycharmProjects/data/N20040103G')
    # filenames = getFiles('/home/auroua/workspace/PycharmProjects/data/N20040103G')
    distance = []
    img1 = getImg(filenames[0])
    for i,fn in enumerate(filenames):
        # print i,fn
        img1 = getImg(filenames[0])
        avg_img = avg_channel(img1)
        try:
            url = filenames[i+1]
            img2 = getImg(url)
            avg_img2 = avg_channel(img2)
        except IndexError as ie:
            print 'end for loop'
            break
        distance.append(np.sum(avg_img-avg_img2))
    # print distance
    np_distance = np.array(distance,dtype=np.double)
    # np_distance = preprocessing.scale(np_distance)
    print np_distance
    length = np_distance.shape[0]
    x_label = np.arange(0,length)
    # line, = plt.bar(x_label, np_distance, '-', linewidth=2)

    sns.set(style="whitegrid")
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(6, 15))

    dic_data = {}
    dic_data['index'] = x_label
    dic_data['samilary'] = np_distance
    sns.set_color_codes("pastel")
    sns.barplot(x="index", y="samilary", data=dic_data,label="Total", color="b")
    sns.plt.show()

