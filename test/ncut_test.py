__author__ = 'auroua'
# /home/auroua/workspace/PycharmProjects/data/coffee.png
from skimage import data, io, segmentation, color, draw
from skimage.future import graph
import matplotlib.pyplot as plt
import numpy as np

def show_img(img):
    print img.shape
    width = img.shape[1]/75.0
    height = img.shape[0]*width/img.shape[1]
    f = plt.figure(figsize=(width,height))
    plt.gray()
    plt.imshow(img)
    plt.show()

def display_edges(image, g):
    max_weight = max([d['weight'] for x, y, d in g.edges_iter(data=True)])
    min_weight = min([d['weight'] for x, y, d in g.edges_iter(data=True)])

    print max_weight, min_weight

    for edge in g.edges_iter():
        n1, n2 = edge
        r1, c1 = map(int, g.node[n1]['centroid'])
        r2, c2 = map(int, g.node[n2]['centroid'])

        n_green = np.array([0, 1, 0])
        n_red = np.array([1, 0, 0])

        line = draw.line(r1, c1, r2, c2)
        circle = draw.circle(r1, c1, 2)
        norm_weight = (g[n1][n2]['weight']-min_weight)/(max_weight-min_weight)

        image[line] = norm_weight*n_red + (1-norm_weight)*n_green
        image[circle] = 1, 1, 0   #the center of the node
    return image


if __name__ == '__main__':
    img = data.coffee()
    # img = data.camera()

    labels1 = segmentation.slic(img, compactness=30, n_segments=120000)
    out1 = color.label2rgb(labels1, img, kind='avg')
    show_img(out1)

    g = graph.rag_mean_color(img, labels1, mode='similarity')
    labels2 = graph.cut_normalized(labels1, g)
    print labels2
    out2 = color.label2rgb(labels2, img, kind='avg')

    show_img(out2)