#encoding: UTF-8
__author__ = 'auroua'
import os
from PIL import Image
import img_tools as tools
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

def covert_img(files):
    for infile in files:
        # print os.path.splitext(infile)
        outfile = os.path.splitext(infile)[0]+'.jpg'
        if infile != outfile:
            try:
                Image.open(infile).save(outfile)
            except IOError :
                print 'cannot covert', infile

if __name__ == '__main__':
    files = tools.get_imlist('/home/auroua/workspace/PycharmProjects/data/pcv_img/avg/')
    # covert_img(files)
    img = Image.open(files[0])

    #图像缩放
    # img.thumbnail((128,128))

    #裁剪区域,并将区域翻转后放置在指定区域
    # box = (100, 100, 400, 400)
    # region = img.crop(box)
    # region = region.transpose(Image.ROTATE_180)
    # img.paste(region, box)
    # pylab.imshow(img)

    #image resize and rotate
    out = img.resize((128,128))
    out = out.rotate(45)

    pylab.imshow(out)
    plt.show()
