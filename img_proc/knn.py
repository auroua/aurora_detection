# encoding:UTF-8
import numpy as np
import math
import pickle
from img_tools import plot_2D_boundary


class KnnClassifier(object):
    def __init__(self, labels, samples):
        """使用训练数据初始化分类器"""
        self.labels = labels
        self.samples = samples

    def classify(self, point, k=3):
        """在训练数据上采用k近邻分类，并返回标记"""
        # 计算所有训练数据点的距离
        dist = np.array([L2dist(point, s) for s in self.samples])
        # 对训练数据排序
        ndx = dist.argsort()
        # 用字典存储k近邻
        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label, 0)
            votes[label]+=1
        return max(votes)


def L2dist(p1, p2):
    return math.sqrt(math.fsum((p1-p2)**2))


def generate_data():
    n = 200
    class_1 = 0.6 * np.random.randn(n, 2)
    class_2 = 1.2 * np.random.randn(n, 2) + np.array([5, 1])
    labels = np.hstack((np.ones(n), -np.ones(n)))
    with open('points_normal.pkl', 'w') as f:
        pickle.dump(class_1, f)
        pickle.dump(class_2, f)
        pickle.dump(labels, f)
    class_1 = 0.6 * np.random.randn(n, 2)
    r = 0.8 * np.random.randn(n, 1) + 5
    angle = 2 * np.pi * np.random.randn(n, 1)
    testt = r * np.cos(angle)
    class_2 = np.hstack((r * np.cos(angle), r * np.sin(angle)))
    labels = np.hstack((np.ones(n), -1 * np.ones(n)))
    with open('points_ring.pkl', 'w') as f:
        pickle.dump(class_1, f)
        pickle.dump(class_2, f)
        pickle.dump(labels, f)




if __name__ == '__main__':
    # generate_data()
    with open('points_normal.pkl', 'r') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)
    model = KnnClassifier(labels, np.vstack((class_1, class_2)))

    with open('points_normal_test.pkl', 'r') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)
    print model.classify(class_1[0])

    def classify(x, y, models=model):
        return np.array([models.classify([xx, yy]) for (xx, yy) in zip(x, y)])

    plot_2D_boundary([-6, 6, -6, 6], [class_1, class_2], classify, [1, -1])