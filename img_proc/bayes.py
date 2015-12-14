import numpy as np
import pickle
import img_tools

class BayesClassifier(object):
    def __init__(self):
        self.labels = []
        self.mean = []
        self.var = []
        self.n = 0

    def train(self, data, labels=None):
        if labels==None:
            labels = range(len(data))
        self.labels = labels
        self.n = len(labels)

        for c in data:
            self.mean.append(np.mean(c, axis=0))
            self.var.append(np.var(c, axis=0))

    def classify(self, points):
        est_prob = np.array([gauss(m, v, points) for m, v in zip(self.mean, self.var)])
        ndx = est_prob.argmax(axis=0)
        est_labels = np.array([self.labels[n] for n in ndx])
        return est_labels, est_prob


def gauss(m, v, x):
    if len(x.shape)==1:
        n, d = 1, x.shape[0]
    else:
        n, d = x.shape

    S = np.diag(1/v)
    x = x-m
    y = np.exp(-0.5*np.diag(np.dot(x, np.dot(S, x.T))))

    return y*(2*np.pi)**(-d/2.0)/(np.sqrt(np.prod(v))+1e-6)

if __name__ == '__main__':
    # with open('points_ring.pkl', 'r') as f:
    with open('points_normal.pkl', 'r') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)
    bc = BayesClassifier()
    bc.train([class_1, class_2], [1, -1])

    with open('points_normal_test.pkl', 'r') as f:
    # with open('points_ring_test.pkl', 'r') as f:
        class_1 = pickle.load(f)
        class_2 = pickle.load(f)
        labels = pickle.load(f)
    print bc.classify(class_2[:10])[0]

    def classify(x, y, bc=bc):
        points = np.vstack((x, y))
        return bc.classify(points.T)[0]

    img_tools.plot_2D_boundary([-6, 6, -6, 6], [class_1, class_2], classify, [1, -1])



