#encoding: UTF-8
import pydot
from PIL import Image
import sift
import numpy as np
from metrics.imgutil import getFiles

def sift_pan_desc_generator(path='/home/aurora/hdd/workspace/PycharmProjects/data/N20040103G/'):
    filelists = getFiles(path)
    feature = []
    for index, file in enumerate(filelists):
        sift.process_image(file, 'pan'+str(index)+'.sift')
        feature.append('pan'+str(index)+'.sift')
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
                im.thumbnail((100, 100))
                filename = str(j) + '.png'
                im.save(filename)  # 需要一定大小的临时文件
                g.add_node(pydot.Node(str(j), fontcolor='transparent', shape='rectangle', image=path + filename))

                g.add_edge(pydot.Edge(str(i), str(j)))
    g.write_png('whitehouse.png')


if __name__=='__main__':
    # sift_matrix()
    matchscores = np.load('pan_img_matchscore.npy')
    imlist = getFiles('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/panoimages/')
    nbr_images = len(imlist)
    path = '/home/aurora/workspace/PycharmProjects/aurora_detection/img_proc/'
    generat_graph(matchscores, imlist, nbr_images, path)
