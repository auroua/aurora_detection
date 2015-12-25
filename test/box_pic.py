# """
# Demo of the new boxplot functionality
# """
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # fake data
# np.random.seed(937)
# data = np.random.lognormal(size=(37, 4), mean=1.5, sigma=1.75)
# labels = list('ABCD')
# fs = 10  # fontsize
#
# # demonstrate how to toggle the display of different elements:
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))
# axes[0, 0].boxplot(data, labels=labels)
# axes[0, 0].set_title('Default', fontsize=fs)
#
# axes[0, 1].boxplot(data, labels=labels, showmeans=True)
# axes[0, 1].set_title('showmeans=True', fontsize=fs)
#
# axes[0, 2].boxplot(data, labels=labels, showmeans=True, meanline=True)
# axes[0, 2].set_title('showmeans=True,\nmeanline=True', fontsize=fs)
#
# axes[1, 0].boxplot(data, labels=labels, showbox=False, showcaps=False)
# axes[1, 0].set_title('Tufte Style \n(showbox=False,\nshowcaps=False)', fontsize=fs)
#
# axes[1, 1].boxplot(data, labels=labels, notch=True, bootstrap=10000)
# axes[1, 1].set_title('notch=True,\nbootstrap=10000', fontsize=fs)
#
# axes[1, 2].boxplot(data, labels=labels, showfliers=False)
# axes[1, 2].set_title('showfliers=False', fontsize=fs)
#
# for ax in axes.flatten():
#     ax.set_yscale('log')
#     ax.set_yticklabels([])
#
# fig.subplots_adjust(hspace=0.4)
# plt.show()
#
#
# # demonstrate how to customize the display different elements:
# boxprops = dict(linestyle='--', linewidth=3, color='darkgoldenrod')
# flierprops = dict(marker='o', markerfacecolor='green', markersize=12,
#                   linestyle='none')
# medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
# meanpointprops = dict(marker='D', markeredgecolor='black',
#                       markerfacecolor='firebrick')
# meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
#
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(6, 6))
# axes[0, 0].boxplot(data, boxprops=boxprops)
# axes[0, 0].set_title('Custom boxprops', fontsize=fs)
#
# axes[0, 1].boxplot(data, flierprops=flierprops, medianprops=medianprops)
# axes[0, 1].set_title('Custom medianprops\nand flierprops', fontsize=fs)
#
# axes[0, 2].boxplot(data, whis='range')
# axes[0, 2].set_title('whis="range"', fontsize=fs)
#
# axes[1, 0].boxplot(data, meanprops=meanpointprops, meanline=False,
#                    showmeans=True)
# axes[1, 0].set_title('Custom mean\nas point', fontsize=fs)
#
# axes[1, 1].boxplot(data, meanprops=meanlineprops, meanline=True, showmeans=True)
# axes[1, 1].set_title('Custom mean\nas line', fontsize=fs)
#
# axes[1, 2].boxplot(data, whis=[15, 85])
# axes[1, 2].set_title('whis=[15, 85]\n#percentiles', fontsize=fs)
#
# for ax in axes.flatten():
#     ax.set_yscale('log')
#     ax.set_yticklabels([])
#
# fig.suptitle("I never said they'd be pretty")
# fig.subplots_adjust(hspace=0.4)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# generate some random test data
all_data = [np.random.normal(0, std, 100) for std in range(6, 10)]

# plot violin plot
axes[0].violinplot(all_data,
                   showmeans=False,
                   showmedians=True)
axes[0].set_title('violin plot')

# plot box plot
axes[1].boxplot(all_data)
axes[1].set_title('box plot')

# adding horizontal grid lines
for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(all_data))])
    ax.set_xlabel('xlabel')
    ax.set_ylabel('ylabel')

# add x-tick labels
plt.setp(axes, xticks=[y+1 for y in range(len(all_data))],
         xticklabels=['x1', 'x2', 'x3', 'x4'])
plt.show()