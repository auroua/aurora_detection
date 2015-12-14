import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

url = '/home/aurora/hdd/workspace/PycharmProjects/data/N20040103G_LBP_R2P8U2/N20040103G030011.mat'

data = sio.loadmat(url)
print data
lbpmap = data['lbpMap']
print lbpmap.shape