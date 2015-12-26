# encoding: UTF-8
import dsift_test
import sift
from scipy.cluster.vq import *
import numpy as np
import os


def gen_texture(url):
    imlist = dsift_test.dsift_filelist(url)
    nbr_images = len(imlist)
    featurelist = [imlist[i][:-3]+'sift' for i in range(nbr_images)]
    for i in range(nbr_images):
        sift.process_image(imlist[i], featurelist[i])

def get_feature_list(url):
    imlist = [os.path.join(url, f) for f in os.listdir(url) if f.endswith('sift')]
    return imlist


class Vocabulary(object):
    def __init__(self, name):
        self.name = name
        self.voc = []
        self.idf = []
        self.trainingdata = []
        self.nbr_words = 0

    def train(self, featurefiles, k=100, subsampling=10):
        """用含有k个单词的K-means列出在featurefiles中的特征文件训练处一个词汇。对训练数据下采样可以加快训练速度"""
        nbr_images = len(featurefiles)
        # 从文件中读取特征
        descr = []
        descr.append(sift.read_feature_from_file(featurefiles[0])[1])
        # 将所有的特征并在一起，以便后面进行K-means聚类
        descriptors = descr[0]
        for i in np.arange(1, nbr_images):
            descr.append(sift.read_feature_from_file(featurefiles[i])[1])
            descriptors = np.vstack((descriptors, descr[i]))

        self.voc, distortion = kmeans(descriptors[::subsampling, :], k, 1)
        self.nbr_words = self.voc.shape[0]

        # 遍历所有的训练图像，并投影到词汇上
        imwords = np.zeros((nbr_images, self.nbr_words))
        print imwords.shape
        for i in range(nbr_images):
            imwords[i] = self.project(descr[i])
        nbr_occurences = np.sum((imwords>0)*1, axis=0)
        self.idf = np.log((1.0*nbr_images)/(1.0*nbr_occurences+1))
        self.trainingdata = featurefiles

    def project(self, descript):
        """将描述子投影到词汇上，以创建单词直方图"""
        # 图像单词直方图
        imhist = np.zeros((self.nbr_words))
        words, distance = vq(descript, self.voc)
        for w in words:
            imhist[w] += 1
        return imhist

if __name__ == '__main__':
    url = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/first1000/'
    # gen_texture(url)
    imlists = get_feature_list(url)
    voc = Vocabulary('test')
    voc.train(imlists)