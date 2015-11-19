import numpy as np
import cv2
import python_ncut_lib as ncut
import matplotlib.pyplot as plt
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
    i = -1
    for ax, name, color in zip(axes, names, cmplists):
        if name!='ColorMap':
            i += 1
            ax.imshow(data[i], aspect='auto', cmap=plt.get_cmap(color))
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3]/2.
            fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
        else:
            indexes = np.linspace(0, 1, 256)
            gradient = np.vstack((indexes, indexes, indexes, indexes))
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(color))
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3]/2.
            fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)


    for ax in axes:
        ax.set_axis_off()


if __name__ == '__main__':
    img_weights_3 = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/similary_gausses_3.npy')
    img_weights_9 = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/similary_gausses_9.npy')
    img_weights_13 = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/similary_gausses_13.npy')
    img_weights_33 = np.load('/home/aurora/hdd/workspace/PycharmProjects/data/similary_gausses_33.npy')
    # show_weights_img(img_weights)

    category = [100, 200, 300, 400]
    index = np.linspace(0, 1, 4312)
    results = np.zeros((4, len(category), img_weights_3.shape[0]))
    for k in category:
        eigval, eigvec = ncut.ncut(img_weights_3, k)
        discret_eigvec = ncut.discretisation(eigvec)
        group_img = discret_eigvec[:, 0]
        for i in range(1, k):
            group_img += (i+1)*discret_eigvec[:, i]
            # print results[category.index(k)].shape
        results[0, category.index(k)] = group_img.todense().T
        results[0, category.index(k)] = (results[0, category.index(k)]/k)*256

    for k in category:
        eigval, eigvec = ncut.ncut(img_weights_9, k)
        discret_eigvec = ncut.discretisation(eigvec)
        group_img_9 = discret_eigvec[:, 0]
        for i in range(1, k):
            group_img_9 += (i+1)*discret_eigvec[:, i]
            # print results[category.index(k)].shape
        results[1, category.index(k)] = group_img_9.todense().T
        results[0, category.index(k)] = (results[0, category.index(k)]/k)*256

    for k in category:
        eigval, eigvec = ncut.ncut(img_weights_13, k)
        discret_eigvec = ncut.discretisation(eigvec)
        group_img_13 = discret_eigvec[:, 0]
        for i in range(1, k):
            group_img_13 += (i+1)*discret_eigvec[:, i]
            # print results[category.index(k)].shape
        results[2, category.index(k)] = group_img_13.todense().T
        results[0, category.index(k)] = (results[0, category.index(k)]/k)*256

    for k in category:
        eigval, eigvec = ncut.ncut(img_weights_33, k)
        discret_eigvec = ncut.discretisation(eigvec)
        group_img_33 = discret_eigvec[:, 0]
        for i in range(1, k):
            group_img_33 += (i+1)*discret_eigvec[:, i]
            # print results[category.index(k)].shape
        results[3, category.index(k)] = group_img_33.todense().T
        results[0, category.index(k)] = (results[0, category.index(k)]/k)*256

    print results.shape

    cmp_list = ['CMRmap', 'CMRmap', 'CMRmap', 'CMRmap', 'CMRmap']
    name_list = ['sigma-3', 'sigma-9', 'sigma-13', 'sigma-33', 'ColorMap']
    plot_color_gradients(4, results, cmp_list, name_list)
    plt.show()
