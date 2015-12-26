import pickle
import svmutil as svm
import img_tools
import numpy as np

with open('points_ring.pkl', 'r') as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)

class_1 = map(list, class_1)
class_2 = map(list, class_2)
labels = list(labels)
samples = class_1+class_2
prob = svm.svm_problem(labels, samples)
param = svm.svm_parameter('-t 2')

m = svm.svm_train(prob, param)
res = svm.svm_predict(labels, samples, m)

with open('points_ring_test.pkl', 'r') as f:
    class_1 = pickle.load(f)
    class_2 = pickle.load(f)
    labels = pickle.load(f)

class_1 = map(list, class_1)
class_2 = map(list, class_2)

def predict(x, y, model=m):
    result = zip(x, y)
    result = map(list, result)
    return np.array(svm.svm_predict([0]*len(x), result, model)[0])

img_tools.plot_2D_boundary([-6, 6, -6, 6], [np.array(class_1), np.array(class_2)], predict, [-1, 1])