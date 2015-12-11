import sift
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

def process_image_dsift(imagename, resultname, size=20, steps=10, force_orientation=False, resize=None):
    im = Image.open(imagename).convert('L')
    if resize!=None:
        im = im.resize(resize)
    m, n = im.size

    if imagename[-3:]!='pgm':
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    scale = size/3.0
    x, y = np.meshgrid(range(steps, m, steps), range(steps, n, steps))
    xx, yy = x.flatten(), y.flatten()
    print xx.shape
    frame = np.array([xx, yy, scale*np.ones(xx.shape[0]), np.zeros(xx.shape[0])])
    print frame.shape
    np.savetxt('tmp.frame', frame.T, fmt='%03.3f')

    if force_orientation:
        cmmd = str('sift '+imagename+' --output='+resultname+" --read-frames=tmp.frame --orientations")
    else:
        cmmd = str('sift '+imagename+' --output='+resultname+" --read-frames=tmp.frame")
    os.system(cmmd)
    print 'processed', imagename, 'to', resultname


if __name__=='__main__':
    """可以使用欧几里得距离来计算dsift之间的距离"""
    process_image_dsift('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/climbing_1_small.jpg', 'climbing_1.sift', 90, 40, True)
    process_image_dsift('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/climbing_2_small.jpg', 'climbing_2.sift', 90, 40, True)
    l1, d1 = sift.read_feature_from_file('climbing_1.sift')
    l2, d2 = sift.read_feature_from_file('climbing_2.sift')

    print 'starting matching'
    matches = sift.match_twosided(d1, d2)

    plt.figure()
    plt.gray()

    img1 = np.array(Image.open('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/climbing_1_small.jpg').convert('L'))
    img2 = np.array(Image.open('/home/aurora/hdd/workspace/PycharmProjects/data/pcv_img/climbing_2_small.jpg').convert('L'))
    sift.plot_matches(img1, img2, l1, l2, matches)
    plt.show()