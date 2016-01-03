import hcluster
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

if __name__=='__main__':
    url = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/sunsets/flickr-sunsets-small/'
    imlist = [os.path.join(url, f) for f in os.listdir(url) if f.endswith('.jpg')]

    features = np.zeros((len(imlist), 512))
    for i,f in enumerate(imlist):
        im = np.array(Image.open(f))
        h, edges = np.histogramdd(im.reshape(-1, 3), 8, normed=True, range=[(0, 255), (0, 255), (0, 255)])
        features[i] = h.flatten()

    tree = hcluster.hcluster(features)
    # hcluster.draw_dendrogram(tree, imlist, filename='sunset.pdf')
    clusters = tree.extract_cluster(0.23*tree.distance)

    for c in clusters:
        elements = c.get_cluster_elements()
        nbr_elements = len(elements)
        if nbr_elements>3:
            plt.figure()
            for p in range(np.minimum(nbr_elements, 20)):
                plt.subplot(4, 5, p+1)
                im = np.array(Image.open(imlist[elements[p]]))
                plt.imshow(im)
                plt.axis('off')
    plt.show()