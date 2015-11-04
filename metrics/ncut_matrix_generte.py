#encoding:UTF-8
__author__ = 'auroua'

import numpy as np
import cv2
import math
from time import clock

def distance(x1, y1, x2, y2):
    dis = math.sqrt((float(x1)-x2)**2+(float(y1)-y2)**2)
    return dis

def weights(images, x1, y1, x2, y2, p_dist):
    # print images[x1, y1], images[x2, y2]
    w_brightness = math.exp(-1*((images[x1, y1]-images[x2, y2])**2/0.01))
    w_distance = math.exp(-1*(p_dist/16.0))
    return w_brightness*w_distance

if __name__ == '__main__':
    # method1 using embedded loop
    # sigmai = 0.1 sigmaix=4.0 r=5    image size:196*284
    img = cv2.imread('/home/auroua/workspace/PycharmProjects/data/imgs/ncuts.png')
    b, g, r = cv2.split(img)
    print b.shape
    temp = b.flatten()
    temp = temp[:, np.newaxis]
    weight = temp - temp.T
    weight = weight - weight
    # print weight.shape
    # print (weight == 0).all()
    print '----------------------------------------'
    start = clock()
    print start

    # weight = np.zeros((b.shape[0]*b.shape[1],b.shape[0]*b.shape[1]))
    x_count = -1
    y_count = -1
    flag = False
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            x_count += 1
            y_count = -1
            flag = False
            for x in range(b.shape[0]):
                if flag:
                    break
                for y in range(b.shape[1]):
                    if i < x and abs(i-x) > 5 and y-j > 5:
                        flag = True
                        break
                    y_count += 1
                    dist = distance(i, j, x, y)
                    if dist < 5:
                        weight[x_count, y_count] = weights(b, i, j, x, y, dist)
                    # else:
                    #     weight[x_count, y_count] = 0

    end = clock()
    print end
    print end - start

    print weight.shape, weight.max(), weight.min()
    np.save('/home/auroua/workspace/PycharmProjects/data/ncuts_weight', weight)


    # cv2.imshow('ncuts',b)
    # cv2.waitKey(0)
