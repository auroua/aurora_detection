import pca
import dsift_test as dsift
import sift
import numpy as np
import bayes


if __name__ == '__main__':
    url1 = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/gesture/train'
    url2 = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/gesture/test'
    featurelist, labels = dsift.read_gesture_features_labels(url1)
    test_featurelist, test_labels = dsift.read_gesture_features_labels(url2)
    V,S,m = pca.pca(featurelist)
    classnames = ['A', 'B', 'C', 'F', 'P', 'V']
    V = V[:50]
    features = np.array([np.dot(V, f-m) for f in featurelist])
    test_features = np.array([np.dot(V, f-m) for f in test_featurelist])
    bc = bayes.BayesClassifier()
    blist = [features[np.where(labels == c)[0]] for c in classnames]
    bc.train(blist, classnames)
    res = bc.classify(test_features)[0]
    acc = np.sum((res==test_labels)*1.0)/len(test_labels)
    print acc
    dsift.print_confusion(res, test_labels, classnames)