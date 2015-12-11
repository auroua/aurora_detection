#encoding:UTF-8
import os
from PIL import Image
import numpy as np
from matplotlib import pylab
import matplotlib.pyplot as plt

def get_imlist(path):
    '''return the absolute path of the image files in the path folder'''
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


def imresize(im, sz):
    pil_im = Image.fromarray(np.uint8(im))
    return pylab.array(pil_im.resize(sz))


def histeq(im, nbr_bins=256):
    """对一副图像进行直方图均衡化"""
    imhist, bins = pylab.histogram(im.flatten(), nbr_bins, normed=True)
    cdf = imhist.cumsum()
    cdf = 255.0*cdf/cdf[-1]
    im2 = pylab.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf

def compute_average(imlist):
    """计算图像列表的平均图像"""
    averageim = np.array(Image.open(imlist[0]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += np.array(Image.open(imname))
        except:
            print imname + '...skipped'
    averageim /= len(imlist)

    return np.array(averageim, 'uint8')


def plot_2D_boundary(plot_range,points,decisionfcn,labels,values=[0]):
    """    Plot_range is (xmin,xmax,ymin,ymax), points is a list
        of class points, decisionfcn is a funtion to evaluate,
        labels is a list of labels that decisionfcn returns for each class,
        values is a list of decision contours to show. """

    clist = ['b','r','g','k','m','y'] # colors for the classes

    # evaluate on a grid and plot contour of decision function
    x = np.arange(plot_range[0],plot_range[1],.1)
    y = np.arange(plot_range[2],plot_range[3],.1)
    xx,yy = np.meshgrid(x,y)
    xxx,yyy = xx.flatten(),yy.flatten() # lists of x,y in grid
    zz = np.array(decisionfcn(xxx,yyy))
    zz = zz.reshape(xx.shape)
    # plot contour(s) at values
    plt.contour(xx,yy,zz,values)

    # for each class, plot the points with '*' for correct, 'o' for incorrect
    for i in range(len(points)):
        d = decisionfcn(points[i][:,0],points[i][:,1])
        correct_ndx = labels[i]==d
        incorrect_ndx = labels[i]!=d
        plt.plot(points[i][correct_ndx,0],points[i][correct_ndx,1],'*',color=clist[i])
        plt.plot(points[i][incorrect_ndx,0],points[i][incorrect_ndx,1],'o',color=clist[i])

    plt.axis('equal')



if __name__ == '__main__':
    empire_im = pylab.array(Image.open('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/empire.jpg').convert('L'))
    pylab.gray()
    pylab.imshow(empire_im)
    pylab.figure()
    im22, cdfo = histeq(empire_im)
    im2_img = Image.fromarray(np.uint8(im22))
    pylab.imshow(im2_img)
    pylab.figure()
    array_img2 = pylab.array(np.uint8(im2_img))
    # pylab.hist(array_img2.flatten(), 256)
    pylab.hist(cdfo, 256)
    pylab.show()