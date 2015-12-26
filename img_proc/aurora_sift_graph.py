#encoding: UTF-8
import pydot
from PIL import Image
import sift
import numpy as np
from metrics.imgutil import getFiles
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

def sift_pan_desc_generator(path='/home/aurora/hdd/workspace/PycharmProjects/data/N20040103G/'):
    filelists = getFiles(path)
    feature = []
    for index, file in enumerate(filelists):
        sift.process_image(file, 'pan'+str(index)+'.sift')
        feature.append('pan'+str(index)+'.sift')
    return feature


def sift_aurora_desc_generator(path, des):
    filelists = getFiles(path)
    feature = []
    for index, file in enumerate(filelists):
        sift.process_image(file, des+str(index)+'.sift')
        feature.append(des+str(index)+'.sift')
    return feature


def sift_matrix():
    featurelist = sift_pan_desc_generator('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/panoimages/')
    imlist = getFiles('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/panoimages/')
    nbr_images = len(imlist)
    matchscores = np.zeros((nbr_images, nbr_images))
    for i in range(nbr_images):
        for j in range(i, nbr_images):
            print 'comparing ', imlist[i], imlist[j]
            l1, d1 = sift.read_feature_from_file(featurelist[i])
            l2, d2 = sift.read_feature_from_file(featurelist[j])

            matches = sift.match_twosided(d1, d2)
            nbr_matches = sum(matches > 0)
            print 'number of matches = ', nbr_matches
            matchscores[i, j] = nbr_matches
    for i in range(nbr_images):
        for j in range(i + 1, nbr_images):
            matchscores[j, i] = matchscores[i, j]
    np.save('pan_img_matchscore', matchscores)


def generat_graph(matchscores, imlist, nbr_images, path):
    # 创建关联最小匹配数目
    threshold = 2
    # not using the default DAG
    g = pydot.Dot(graph_type='graph')
    for i in range(nbr_images):
        for j in range(i + 1, nbr_images):
            if matchscores[i, j] > threshold:
                # 图像对中的第一幅图像
                im = Image.open(imlist[i])
                im.thumbnail((100, 100))
                filename = str(i) + '.png'
                im.save(filename)  # 需要一定大小的临时文件
                g.add_node(pydot.Node(str(i), fontcolor='transparent', shape='rectangle', image=path + filename))
                # 图像对中的第二副图像
                im = Image.open(imlist[j])
                im.thumbnail((50, 50))
                filename = str(j) + '.png'
                im.save(filename)  # 需要一定大小的临时文件
                g.add_node(pydot.Node(str(j), fontcolor='transparent', shape='rectangle', image=path + filename))

                g.add_edge(pydot.Edge(str(i), str(j)))
    g.write_png('aurora.png')


def compare_scale_histogram():
    """计算图像sift算子 尺度的统计直方图"""
    scale_1 = np.load('scales_1.npy')
    scale_2 = np.load('scales_2.npy')
    scale_3 = np.load('scales_3.npy')
    scale_4 = np.load('scales_4.npy')

    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flat
    plt.title(r'four different kind aurora type histogram')
    ax0.hist(scale_1, 100, normed=1, histtype='bar', facecolor='g', alpha=0.75)
    ax0.set_title('scale_1')
    ax1.hist(scale_2, 100, normed=1, histtype='bar', facecolor='b', alpha=0.75)
    ax1.set_title('scale_2')
    ax2.hist(scale_3, 100, normed=1, histtype='bar', facecolor='r', alpha=0.75)
    ax2.set_title('scale_3')
    ax3.hist(scale_4, 100, normed=1, histtype='bar', facecolor='tan', alpha=0.75)
    ax3.set_title('scale_4')

    plt.show()

def generate_scale_box():
    """计算四种类型的箱线图"""
    scale_1 = np.load('scales_1.npy')
    scale_2 = np.load('scales_2.npy')
    scale_3 = np.load('scales_3.npy')
    scale_4 = np.load('scales_4.npy')
    print scale_1.shape, scale_2.shape, scale_3.shape, scale_4.shape
    data = np.zeros((scale_1.shape[0], 4))
    data[:, 0] = scale_1
    data[0:scale_2.shape[0], 1] = scale_2
    data[0:scale_3.shape[0], 2] = scale_3
    data[0:scale_4.shape[0], 3] = scale_4

    # plt.boxplot(scale_2, notch=True, patch_artist=True)
    # plt.boxplot(scale_3, notch=True, patch_artist=True)
    # plt.boxplot(scale_4, notch=True, patch_artist=True)
    # plt.boxplot(data)
    fig, axes = plt.subplots(nrows=2, ncols=2)
    ax0, ax1, ax2, ax3 = axes.flat
    ax0.boxplot(scale_1, notch=True, patch_artist=True)
    ax0.set_title('scale_1')
    ax1.boxplot(scale_2, notch=True, patch_artist=True)
    ax1.set_title('scale_2')
    ax2.boxplot(scale_3, notch=True, patch_artist=True)
    ax2.set_title('scale_3')
    ax3.boxplot(scale_4, notch=True, patch_artist=True)
    ax3.set_title('scale_4')


    plt.show()

if __name__=='__main__':
    # generate the aurora graph, maybe the graph is too big to generate
    # matchscores = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/aurora_img_matches_matrix_20151212.npy')
    # imlist = getFiles('/home/aurora/hdd/workspace/PycharmProjects/data/N20040103G/')
    # nbr_images = len(imlist)
    # path = '/home/aurora/workspace/PycharmProjects/aurora_detection/img_proc/'
    # generat_graph(matchscores, imlist, nbr_images, path)

    # 2015-12-24  generate the four different kinds image datas
    # url_1 = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora/1/'
    # url_2 = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora/2/'
    # url_3 = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora/3/'
    # url_4 = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora/4/'
    # destinction_1 = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora/sift_1/'
    # destinction_2 = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora/sift_2/'
    # destinction_3 = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora/sift_3/'
    # destinction_4 = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora/sift_4/'
    #
    # sift_aurora_desc_generator(url_1, destinction_1)
    # sift_aurora_desc_generator(url_2, destinction_2)
    # sift_aurora_desc_generator(url_3, destinction_3)
    # sift_aurora_desc_generator(url_4, destinction_4)


    # sift_url_1 = '/home/aurora/hdd/workspace/PycharmProjects/data/aurora/sift_4/'
    # imlist = getFiles(sift_url_1)
    # temp = []
    # for i, sift_img in enumerate(imlist):
    #     l = sift.read_scale_from_file(sift_img)
    #     for j in range(l.shape[0]):
    #         temp.append(l[j])
    # print len(temp)
    # scales = np.array(temp)
    # np.save('scales_4', scales)
    # compare_scale_histogram()
    generate_scale_box()


