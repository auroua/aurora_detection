#encoding:UTF-8
__author__ = 'auroua'
__version__ = 0.1

import numpy as np
import seaborn as sns
import imgutil as img

if __name__=='__main__':
    filenames = img.getFiles('/home/auroua/workspace/PycharmProjects/data/N20040103G')
    # filenames = img.getFiles('/home/auroua/workspace/PycharmProjects/data/picture11')
    # filenames = getFiles('/home/auroua/workspace/PycharmProjects/data/N20040103G')
    distance = []
    # img1 = img.getImg(filenames[0])
    for i,fn in enumerate(filenames):
        # print i,fn
        img1 = img.getImg(filenames[0])
        hist_img = img.hist(img1)
        try:
            url = filenames[i+1]
            img2 = img.getImg(url)
            hist_img2 = img.hist(img2)
        except IndexError as ie:
            print 'end for loop'
            break
        distance.append(np.sum(np.abs(hist_img-hist_img2)))
    # print distance
    np_distance = np.array(distance,dtype=np.double)
    # np_distance = preprocessing.scale(np_distance)
    print np_distance

    length = np_distance.shape[0]
    x_label = np.arange(0,length)

    sns.set(style="whitegrid")
    dic_data = {}
    dic_data['index'] = x_label
    dic_data['samilary'] = np_distance
    sns.set_color_codes("pastel")
    sns.barplot(x="index", y="samilary", data=dic_data,label="Total", color="b")
    sns.plt.show()

