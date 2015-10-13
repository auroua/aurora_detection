#encoding:UTF-8
__author__ = 'auroua'

import numpy as np
import cv2
import os

# /home/auroua/workspace/PycharmProjects/data/N20040103G/

def getImg(url):
    pass

def getFiles(path):
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
    print filenames
