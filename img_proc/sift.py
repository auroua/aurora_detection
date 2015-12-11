#encoding:UTF-8
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from imgutil import getFiles

def process_image(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):
    """处理一副图像， 然后将结果保存在文件中"""
    if imagename[-3:]!='pgm':
        # 创建一个pgm文件
        im = Image.open(imagename).convert('L')
        im.save('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/temp.pgm')
        imagename = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/temp.pgm'

    cmmd = str('sift '+imagename+' --output='+resultname+"  "+params)
    os.system(cmmd)
    print 'processed', imagename, 'to', resultname


def read_feature_from_file(filename):
    """读取特征属性值，然后将其以矩阵的形式返回"""
    f = np.loadtxt(filename)
    if f.ndim == 1:
        f = f.reshape((1, f.shape[0]))
    return f[:, :4], f[:, 4:]


def write_features_to_file(filename, locs, desc):
    """将特征位置和描述子保存到文件中"""
    np.savetxt(filename, np.hstack((locs, desc)))


def plot_features(im, locs, circle=False):
    def draw_circle(c, r):
        t = np.arange(0, 1.01, .01)*2*np.pi
        x = r*np.cos(t)+c[0]
        y = r*np.sin(t)+c[1]
        plt.plot(x, y, 'b', linewidth=2)
    plt.imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2],p[2])
    else:
        plt.plot(locs[:, 0], locs[:, 1], 'ob')
    plt.axis('off')


def get_sift_size():
    filelists = getFiles()
    datasets = []
    for index, url in enumerate(filelists):
        im = np.array(Image.open(url).convert('L'))
        process_image(url, 'aurora'+str(index)+'.sift')
        l1, d1 = read_feature_from_file('aurora'+str(index)+'.sift')
        if l1.shape[0] == 1 and l1.shape[1] == 0:
            print '@@@@@@@@@@@@@@@@@@@@'
            print url
            datasets.append(0)
            continue
        else:
            datasets.append(l1.shape[0])
    return datasets


def match(desc1, desc2):
    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape
    matchscores = np.zeros((desc1_size[0], 1), 'int')
    desc2t = desc2.T
    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i, :], desc2t)
        dotprods = 0.9999*dotprods
        # indx is a two dim matrix
        indx = np.argsort(np.arccos(dotprods))
        if np.arccos(dotprods)[indx[0]] < dist_ratio*np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])
    return matchscores


def match_twosided(desc1, desc2):
    """两边对称版本的match()"""
    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)
    # return the index address
    ndx_12 = matches_12.nonzero()[0]

    # 去除非对称的匹配
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = 0
    return matches_12


def appendimages(im1, im2):
    """返回将两幅图像并排拼接成的一副新图像"""

    # 选取具有最少行数的图像，然后填充足够的空格
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2-rows1, im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1-rows2, im2.shape[1]))), axis=0)
    return np.concatenate((im1, im2), axis=1)


def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
    """显示一幅带有连接匹配之间连线的图片"""
    im3 = appendimages(im1, im2)
    temp = im3.copy()
    if show_below:
        im3 = np.vstack((im3, im3))
        im3 = np.vstack((im3, temp))

    plt.imshow(im3)

    cols1 = im1.shape[1]
    rows1 = im1.shape[0]
    plot_features(im3[0:rows1, 0:cols1], locs1, circle=True)
    locs2_t = locs2.copy()
    locs2_t[:, 0] += cols1
    plot_features(im3[0:rows1, cols1:2*cols1], locs2_t, circle=True)
    for i, m in enumerate(matchscores):
        # print 'the value of i '+str(i)+' the value of m is '+str(m)
        if m > 0:
            plt.plot([locs1[i][0], locs2[m[0]][0]+cols1], [locs1[i][1]+rows1, locs2[m[0]][1]+rows1], 'c')
            # plt.plot([276.96, 326.19], [373.45, 450.915], 'c')
    plt.axis('off')


if __name__ == '__main__':
    # # url = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/crans_1_small.jpg'
    # # process_image(url, 'img_test')
    # url = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/crans_1_small.jpg'
    # # url = '/home/aurora/workspace/PycharmProjects/data/N20040103G/N20040103G030001.bmp'
    # im1 = np.array(Image.open(url).convert('L'))
    # process_image(url, 'empire.sift')
    # l1, d1 = read_feature_from_file('empire.sift')
    # print im1.shape
    # print l1[:, 0].max(), l1[:, 1].max()
    # # l1, d1 = read_feature_from_file('aurora1887.sift')
    #
    # plt.figure()
    # plt.gray()
    # plot_features(im1, l1, circle=True)
    # plt.show()

    # caculate the sift description counts
    # results = get_sift_size()
    # print results


    # match two sides
    # im1 = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/crans_1_small.jpg'
    # im1 = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/crans_1_small.jpg'
    im1 = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/climbing_1_small.jpg'
    im2 = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/climbing_2_small.jpg'
    # filelists = imgutil.getFiles()
    # for file in filelists:
    #     img = np.array(Image.open(file).convert('L'))
    #     harrisim = compute_harris_response(img)
    #     filtered_coords = get_harris_points(harrisim, 6)
    #     # plot_harris_points(img, filtered_coords)
    #     if len(filtered_coords)>0:
    #         print file

    # harrisim = compute_harris_response(im1)
    # filtered_coords = get_harris_points(harrisim, 6)
    # plot_harris_points(im1, filtered_coords)

    process_image(im1, 'im1.sift')
    l1, d1 = read_feature_from_file('im1.sift')

    process_image(im2, 'im2.sift')
    l2, d2 = read_feature_from_file('im2.sift')

    print 'starting matching'
    matches = match_twosided(d1, d2)

    plt.figure()
    plt.gray()

    img1 = np.array(Image.open('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/climbing_1_small.jpg').convert('L'))
    img2 = np.array(Image.open('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/climbing_2_small.jpg').convert('L'))
    plot_matches(img1, img2, l1, l2, matches)
    plt.show()