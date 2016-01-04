# encoding:UTF-8
import cv2
import numpy as np


if __name__=='__main__':
    url = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/empire.jpg'
    im = cv2.imread(url)
    print im.shape
    # 下采样
    im_lowres = cv2.pyrDown(im)
    print im_lowres.shape

    gray = cv2.cvtColor(im_lowres, cv2.COLOR_RGB2GRAY)

    s = cv2.SURF()
    mask = np.uint8(np.ones(gray.shape))
    keypoints = s.detect(gray, mask)

    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for k in keypoints[::10]:
        cv2.circle(vis, (int(k.pt[0]), int(k.pt[1])), 2, (0, 255, 0), -1)
        cv2.circle(vis, (int(k.pt[0]), int(k.pt[1])), int(k.size), (0, 255, 0), 2)

    cv2.imshow('local descriptors', vis)
    cv2.waitKey()