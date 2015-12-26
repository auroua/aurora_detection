import dsift
import dsift_test
import sift
import svmutil as svm
import numpy as np

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
    classnames = ['A', 'B', 'C', 'F', 'P', 'V']
    featurelist, labels = dsift_test.read_gesture_features_labels(url1)
    test_featurelist, test_labels = dsift_test.read_gesture_features_labels(url2)
    featurelist = map(list, featurelist)
    test_featurelist = map(list, test_featurelist)
    transl = {}
    for i, c in enumerate(classnames):
        transl[c], transl[i] = i, c

    labels = [transl[i] for i in labels]
    temp_test_labels = [transl[i] for i in test_labels]
    prob = svm.svm_problem(labels, featurelist)
    param = svm.svm_parameter('-t 0')
    m = svm.svm_train(prob, param)

    res = svm.svm_predict(labels, featurelist, m)

    res = svm.svm_predict(temp_test_labels, test_featurelist, m)[0]
    res = [transl[r] for r in res]
    flags = res == test_labels
    flags = flags*1.0
    correct = np.sum(flags)
    accuracy = correct/len(test_labels)
    print accuracy


    dsift_test.print_confusion(res, test_labels, classnames)