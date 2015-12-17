import numpy as np
import cv2
import python_ncut_lib as ncut
import matplotlib.pyplot as plt
from time import clock
__author__ = 'auroua'


def show_weights_img(originsl_images):
    temp_weights = originsl_images.copy()
    temp_weights = temp_weights/temp_weights.max()
    temp_weights = 1 - temp_weights
    img_resize = cv2.resize(temp_weights, (1024, 1024))
    cv2.imshow("weights_matrix", img_resize)
    cv2.waitKey(0)


def plot_color_gradients(nrows, data, cmplists, names):
    fig, axes = plt.subplots(nrows+1)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title('weights color_maps', fontsize=14)
    i3 = -1
    for name, color in zip(names, cmplists):
        if name != 'ColorMap':
            i3 += 1
            for j in xrange(5):
                temp = np.vstack((data[i3, j, :], data[i3, j, :]))
                # temp = data[i3, j, :]
                axes[i3*4+j].imshow(temp, aspect='auto', cmap=plt.get_cmap(color))
                pos = list(axes[i3*4+j].get_position().bounds)
                x_text = pos[0] - 0.01
                y_text = pos[1] + pos[3]/2.
                fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
        else:
            indexes = np.linspace(0, 1, 256)
            gradient = np.vstack((indexes, indexes, indexes, indexes))
            axes[-1].imshow(gradient, aspect='auto', cmap=plt.get_cmap(color))
            pos = list(axes[-1].get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3]/2.
            fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
    for ax in axes:
        ax.set_axis_off()


def generate_sift_match_graph():
    #TODO
    pass

if __name__ == '__main__':
    img_weights_3 = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/similary_gausses_3.npy')
    img_weights_9 = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/similary_gausses_9.npy')
    img_weights_13 = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/similary_gausses_13.npy')
    img_weights_33 = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/similary_gausses_33.npy')
    hist_weights_33 = np.load('/home/aurora/workspace/PycharmProjects/data/hist_adjacent_matrix.npy')
    # hist_weights_constraint_33 = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/sub_matrix_distance_2015_1211.npy')
    hist_weights_constraint_33 = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/aurora_img_matches_matrix_20151212.npy')
    # show_weights_img(img_weights)
    # datas = [img_weights_3, img_weights_9, img_weights_13, img_weights_33]
    datas = [hist_weights_constraint_33]
    start = clock()
    category = [100, 200, 300, 400, 500]
    # category = [500, 600, 700, 800]
    # category = [4, 6, 8, 10]
    results = np.zeros((4, len(category), hist_weights_constraint_33.shape[0]))
    for idx, data in enumerate(datas):
        for k in category:
            eigval, eigvec = ncut.ncut(data, k)
            discret_eigvec = ncut.discretisation(eigvec)
            group_img = discret_eigvec[:, 0]
            for i in range(1, k):
                group_img += (i+1)*discret_eigvec[:, i]
                # print results[category.index(k)].shape
            results[idx, category.index(k)] = group_img.todense().T
            # results[0, category.index(k)] = (results[0, category.index(k)]/k)*256
    print results.shape
    np.save('/home/aurora/hdd/workspace/PycharmProjects/data/img_sub_ncuts_matrix_distance_2015_1211_m_400', results)
    # print np.unique(results[0][0])
    # print np.unique(results[0][1])
    # print np.unique(results[0][2])
    # print np.unique(results[0][3])
    # print np.unique(results[0][4])
    # print np.unique(results[0][5])
    # print np.unique(results[0][6])

    end = clock()
    print 'total run time is: '
    print end - start
    # cmp_list = ['CMRmap', 'CMRmap', 'CMRmap', 'CMRmap', 'CMRmap']
    cmp_list = ['CMRmap', 'CMRmap']
    # name_list = ['sigma-3', 'sigma-9', 'sigma-13', 'sigma-33', 'ColorMap']
    name_list = ['sigma-33', 'ColorMap']
    # plot_color_gradients(16, results, cmp_list, name_list)
    plot_color_gradients(5, results, cmp_list, name_list)
    plt.show()
