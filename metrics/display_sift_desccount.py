import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":
    data = np.load('sift_desc_data.npy')
    adata = np.array(data)
    indexs = np.arange(adata.shape[0])

    # line, = plt.plot(indexs, adata, '-', linewidth=2)
    plt.plot(indexs, adata, 'ro')
    plt.show()