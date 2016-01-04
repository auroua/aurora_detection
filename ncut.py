import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import *
from scipy.cluster.vq import *
from PIL import Image
from scipy.misc import imresize


def ncut_graph_matrix(im,sigma_d=1e2,sigma_g=1e-2):
    """ Create matrix for normalized cut. The parameters are
        the weights for pixel distance and pixel similarity. """

    m,n = im.shape[:2]
    N = m*n

    # normalize and create feature vector of RGB or grayscale
    if len(im.shape)==3:
        for i in range(3):
            im[:,:,i] = im[:,:,i] / im[:,:,i].max()
        vim = im.reshape((-1,3))
    else:
        im = im / im.max()
        vim = im.flatten()

    # x,y coordinates for distance computation
    xx,yy = np.meshgrid(range(n),range(m))
    x,y = xx.flatten(),yy.flatten()

    # create matrix with edge weights
    W = np.zeros((N,N),'f')
    for i in range(N):
        for j in range(i,N):
            d = (x[i]-x[j])**2 + (y[i]-y[j])**2
            W[i,j] = W[j,i] = np.exp(-1.0*sum((vim[i]-vim[j])**2)/sigma_g) * np.exp(-d/sigma_d)
    return W


def cluster(S,k,ndim):
    """ Spectral clustering from a similarity matrix."""

    # check for symmetry
    if sum(abs(S-S.T)) > 1e-10:
        print 'not symmetric'

    # create Laplacian matrix
    rowsum = sum(abs(S),axis=0)
    D = np.diag(1 / np.sqrt(rowsum + 1e-6))
    L = np.dot(D, np.dot(S,D))

    # compute eigenvectors of L
    U,sigma,V = np.linalg.svd(L,full_matrices=False)

    # create feature vector from ndim first eigenvectors
    # by stacking eigenvectors as columns
    features = np.array(V[:ndim]).T

    # k-means
    features = whiten(features)
    centroids,distortion = kmeans(features,k)
    code,distance = vq(features,centroids)

    return code,V

if __name__=='__main__':
    url = '/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/C-uniform03.ppm'
    im = np.array(Image.open(url))
    m, n = im.shape[:2]
    # resize image to (wid,wid)
    wid = 50
    rim = imresize(im, (wid, wid), interp='bilinear')
    rim = np.array(rim, 'f')
    # create normalized cut matrix
    A = ncut_graph_matrix(rim, sigma_d=1, sigma_g=1e-2)
    # cluster
    code, V = cluster(A, k=3, ndim=3)

    plt.imshow(imresize(code.reshape(wid,wid),(m,n),interp='bilinear'))