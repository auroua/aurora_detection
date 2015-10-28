#encoding:UTF-8
__author__ = 'auroua'
from skimage import data, io, segmentation, color, draw
from skimage.future import graph
import matplotlib.pyplot as plt
from skimage.measure import regionprops
import numpy as np

def show_img(img):
    print img.shape
    width = img.shape[1]/75.0
    height = img.shape[0]*width/img.shape[1]
    f = plt.figure(figsize=(width,height))
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

if __name__=='__main__':
    # '/home/auroua/workspace/PycharmProjects/data/label_rgb.png'
    demo_image = io.imread('/home/auroua/workspace/PycharmProjects/data/label_rgb.png')
    # show_img(demo_image)

    #使用k-means对图像进行聚类   segments 100类
    labels = segmentation.slic(demo_image, compactness=30, n_segments=100)
    print labels
    labels = labels + 1
    #通过regionprops 生成图像区域的信息
    regions = regionprops(labels)

    label_rgb = color.label2rgb(labels, demo_image, kind='avg')
    #(0, 1, 1) RGB 决定了边界的颜色
    label_rgb = segmentation.mark_boundaries(label_rgb, labels, (0, 1, 1))

    #计算临接节点的相似度  similarity不仅衡量 色彩上的相似度  也衡量距离上的相似度  region adjacency graph
    rag = graph.rag_mean_color(demo_image, labels, mode='similarity')
    for region in regions:
        rag.node[region['label']]['centroid'] = region['centroid']

    label_rgb = display_edges(label_rgb, rag)
    show_img(label_rgb)