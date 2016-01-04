#encoding:UTF-8
from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import maximum_flow
import numpy as np
import bayes
import matplotlib.pyplot as plt
from scipy.misc import imresize
from PIL import Image


def build_bayes_grapy(im, labels, sigma=1e2, kappa=1):
    """从像素四邻域建立一个图，前景和背景（前景用1标记，背景用-1标记，其他的用0标记）
       由labels决定，并用朴素贝叶斯分类器建模"""
    m, n = im.shape[:2]
    vim = im.reshape((-1, 3))
    print vim.shape

    foreground = im[labels == 1].reshape((-1, 3))
    background = im[labels == -1].reshape((-1, 3))
    train_data = [foreground, background]

    bc = bayes.BayesClassifier()
    bc.train(train_data)

    bc_labels, prob = bc.classify(vim)
    prob_fg = prob[0]
    prob_bg = prob[1]
    gr = digraph()
    gr.add_nodes(range(m*n+2))
    source = m*n
    sink = m*n+1

    vim = vim.astype(dtype=np.float64)
    vim += 1e-6
    for i in range(vim.shape[0]):
        vim[i] = vim[i]/np.linalg.norm(vim[i])
    for i in range(m*n):
        # add edge from source
        gr.add_edge((source, i), wt=(prob_fg[i]/(prob_fg[i]+prob_bg[i])))

        # add edge to sink
        gr.add_edge((i, sink), wt=(prob_bg[i]/(prob_fg[i]+prob_bg[i])))

        # add edges to neighbors
        if i % n != 0:    # left exists
            edge_wt = kappa*np.exp(-1.0*sum((vim[i]-vim[i-1])**2)/sigma)
            gr.add_edge((i, i-1), wt=edge_wt)
        if (i+1) % n != 0:  # right exists
            edge_wt = kappa*np.exp(-1.0*sum((vim[i]-vim[i+1])**2)/sigma)
            gr.add_edge((i, i+1), wt=edge_wt)
        if i//n != 0:  # up exists
            edge_wt = kappa*np.exp(-1.0*sum((vim[i]-vim[i-n])**2)/sigma)
            gr.add_edge((i, i-n), wt=edge_wt)
        if i//n != m-1:  # down exists
            edge_wt = kappa*np.exp(-1.0*sum((vim[i]-vim[i+n])**2)/sigma)
            gr.add_edge((i, i+n), wt=edge_wt)

        return gr


def cut_graph(gr, imsize):
    """    Solve max flow of graph gr and return binary
        labels of the resulting segmentation."""

    m, n = imsize
    source = m*n  # second to last is source
    sink = m*n+1  # last is sink

    # cut the graph
    flows, cuts = maximum_flow(gr, source, sink)
    # print cuts
    # convert graph to image with labels
    res = np.zeros(m*n)
    for pos, label in cuts.items()[:-2]: #don't add source/sink
        res[pos] = label

    return res.reshape((m, n))


def show_labeling(im, labels):
    """    Show image with foreground and background areas.
        labels = 1 for foreground, -1 for background, 0 otherwise."""

    plt.imshow(im)
    plt.contour(labels, [-0.5, 0.5])
    plt.contourf(labels, [-1, -0.5], colors='b', alpha=0.25)
    plt.contourf(labels, [0.5, 1], colors='r', alpha=0.25)
    plt.xticks([])
    plt.yticks([])


# def save_as_pdf(gr, filename, show_weights=False):
#     from pygraph.readwrite.dot import write
#     import gv
#     dot = write(gr, weighted=show_weights)
#     gvv = gv.readstring(dot)
#     gv.layout(gvv, 'fdp')
#     gv.render(gvv, 'pdf', filename)

if __name__ == '__main__':
    url = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/empire.jpg'
    im = np.array(Image.open(url))
    im = imresize(im, 0.07, interp='bilinear')
    print im.shape
    size = im.shape[:2]

    # size 56*39
    labels = np.zeros(size)
    labels[3:18, 3:18] = -1
    labels[-18:-3, -18:-3] = 1
    g = build_bayes_grapy(im, labels, kappa=1)

    res = cut_graph(g, size)
    print res

    plt.figure()
    show_labeling(im, labels)
    plt.figure()
    plt.imshow(res)
    plt.gray()
    plt.axis('off')
    plt.show()