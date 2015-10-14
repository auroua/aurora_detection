#encoding:UTF-8
__author__ = 'auroua'

import os
import cv2
import numpy as np

# original file path  /home/auroua/workspace/PycharmProjects/data/N20040103G/
# test file path  /home/auroua/workspace/PycharmProjects/data/test/

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