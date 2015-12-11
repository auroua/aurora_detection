#encoding: UTF-8
from scipy.ndimage import filters
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imgutil

def compute_harris_response(im, sigma=3):
    """在一副灰度图像中， 对每个像素计算Harris角点检测器响应函数"""
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    # 计算Harris矩阵的分量
    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    # 计算特征值和迹
    wdet = Wxx*Wyy - Wxy**2
    wtr = Wxx+Wyy

    return wdet/wtr


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """从一幅Harris响应图像中返回焦点.min_dist为分割焦点和图像便捷的最少像素数目"""
    # 寻找高于预支的候选角点
    corner_threshold = harrisim.max()*threshold
    harrisim_t = (harrisim > corner_threshold)*1
    # 得到候选点的坐标
    # harrisim_t.nonzero()  tuple  the following code tuple--->array
    coords = np.array(harrisim_t.nonzero()).T
    # 以及它们的Harris响应值
    candidate_values = [harrisim[c[0], c[1]] for c in coords]
    # 对候选点按照Harris响应值进行排序  返回的索引从小到大排列
    index = np.argsort(candidate_values)
    # 将可行点的位置保存到数组中
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # 按照min_distance原则， 选择最佳Harris点
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0]-min_dist):(coords[i, 0]+min_dist), (coords[i, 1]-min_dist):(coords[i, 1]+min_dist)] = 0
    return filtered_coords


def plot_harris_points(image, filtered_coords):
    """绘制图像中检测到的角点"""
    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    plt.axis('off')
    plt.show()


def get_descriptors(image, filtered_coords, wid=5):
    """对于每个返回的点， 返回点周围的2*wid+1个像素值(假设选取的min_distance>wid)"""
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid:coords[0]+wid+1, coords[1]-wid:coords[1]+wid+1].flatten()
        desc.append(patch)
    return desc


def match(desc1, desc2, threshold=0.5):
    """对于第一幅图像中的每个角点描述子，使用归一化互相关操作，选取它在第二幅图像中的匹配角点"""
    n = len(desc1[0])

    # 点对的距离
    d = -np.ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - np.mean(desc1[i]))/np.std(desc1[i])
            d2 = (desc2[j] - np.mean(desc2[j]))/np.std(desc2[j])
            ncc_value = np.sum(d1*d2)/(n-1)
            if ncc_value > threshold:
                d[i, j] = ncc_value

    ndx = np.argsort(-d)
    matchscores = ndx[:, 0]
    return matchscores


def match_twosided(desc1, desc2, threshold=0.5):
    """两边对称版本的match()"""
    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)
    print matches_12.shape
    print matches_21.shape
    # return the index address
    ndx_12 = np.where(matches_12 >= 0)[0]
    # 去除非对称的匹配
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1
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
    if show_below:
        im3 = np.vstack((im3, im3))

    plt.imshow(im3)

    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m>0:
            plt.plot([locs1[i][1],locs2[m][1]+cols1], [locs1[i][0], locs2[m][0]], 'c')
    plt.axis('off')


if __name__ == '__main__':
    im1 = np.array(Image.open('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/crans_1_small.jpg').convert('L'))
    im2 = np.array(Image.open('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/crans_2_small.jpg').convert('L'))
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

    wid = 5
    harrisim = compute_harris_response(im1, 5)
    filtered_coords1 = get_harris_points(harrisim, wid+1)
    d1 = get_descriptors(im1, filtered_coords1, wid)

    harrisim = compute_harris_response(im2, 5)
    filtered_coords2 = get_harris_points(harrisim, wid+1)
    d2 = get_descriptors(im2, filtered_coords2, wid)

    print 'starting matching'
    matches = match_twosided(d1, d2)

    plt.figure()
    plt.gray()
    plot_matches(im1, im2, filtered_coords1, filtered_coords2, matches)
    plt.show()