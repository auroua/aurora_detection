#encoding:UTF-8
from imgutil import getFiles
import numpy as np
import dsift
import sift
import matplotlib.pyplot as plt
from PIL import Image
import os
from knn import KnnClassifier


def dsift_filelist(url):
    imlists = getFiles(url)
    return imlists

def get_labels(images_names):
    labels = [img.split('/')[-1][0] for img in images_names]
    return labels


def dsift_features_generator(imlist):
    for img in imlist:
        featfile = img[:-3]+'dsift'
        dsift.process_image_dsift(img, featfile, 10, 5, resize=(50, 50))


def read_gesture_features_labels(path):
    featlists = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dsift')]

    features = []
    for featfile in featlists:
        l, d = sift.read_feature_from_file(featfile)
        features.append(d.flatten())
    features = np.array(features, dtype=np.float64)
    labels = [f.split('/')[-1][0] for f in featlists]
    return features, np.array(labels)


def print_confusion(result, labels, classnames):
    n = len(classnames)
    # confusion matrix
    class_ind = dict([(classnames[i], i) for i in range(n)])
    confuse = np.zeros((n, n))
    for i in range(len(result)):
        confuse[class_ind[result[i]], class_ind[labels[i]]]+=1
    print 'Confusion matrix for'
    print classnames
    print confuse

if __name__ == '__main__':
    # generator features
    url1 = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/gesture/train'
    url2 = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/gesture/test'
    # imlists = dsift_filelist(url2)
    # imlabels = get_labels(imlists)
    # dsift_features_generator(imlists)

    # show one gesture image
    # im = Image.open('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/gesture/train/A-uniform02.ppm').convert('L')
    # im = im.resize((50, 50))
    # im1 = np.array(im)
    # l1, d1 = sift.read_feature_from_file('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/gesture/train/A-uniform02.dsift')
    # plt.figure()
    # plt.gray()
    # sift.plot_features(im1, l1, circle=True)
    # plt.show()
    featurelist, labels = read_gesture_features_labels(url1)
    test_featurelist, test_labels = read_gesture_features_labels(url2)
    models = KnnClassifier(labels, featurelist)

    results = [models.classify(test_img, 1) for test_img in test_featurelist]
    results = np.array(results)
    flags = results == test_labels
    flags = flags*1.0
    correct = np.sum(flags)
    accuracy = correct/len(test_labels)
    print accuracy

    classnames = ['A', 'B', 'C', 'F', 'P', 'V']
    print_confusion(results, test_labels, classnames)



