import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

N = 430
T = 48

# preparing measurements data
m = np.loadtxt('data/2011/2011080100_measurements_GI_2623.txt')
mh = np.zeros((N, T))
for i in range(N):
    for j in range(T):
        tt = i*6-j
        if tt < 0:
            mh[i, j] = np.nan
        else:
            mh[i, j] = m[tt]

# ensembles loading
with open('data/2011/2011080100_ens_dist_names.txt') as t_file:
    ens_names = t_file.read().split(sep='\n')
ens_fc = ()
ens_err = ()
for i in range(len(ens_names)):
    ens_fc += (np.loadtxt('data/2011/2011080100_ens_'+ens_names[i]+'_GI_48x430.txt'),)
    ens_err += (np.loadtxt('data/2011/2011080100_ens_'+ens_names[i]+'_GI_48x430_err.txt'),)
    print('Loaded', ens_names[i], np.shape(ens_fc[i]), np.shape(ens_err[i]))
ens_dist_dtw = np.loadtxt('data/2011/2011080100_ens_dist_dtw.txt').transpose()
ens_dist_mae = np.loadtxt('data/2011/2011080100_ens_dist_mae.txt').transpose()
ens_dist_index = np.loadtxt('data/2011/2011080100_ens_dist_index.txt').astype(int)
ens_dist_index_matrix = np.full((len(ens_names), len(ens_names)), -1)
for i in range(len(ens_dist_index)):
    ens_dist_index_matrix[ens_dist_index[i, 0], ens_dist_index[i, 1]] = i
    ens_dist_index_matrix[ens_dist_index[i, 1], ens_dist_index[i, 0]] = i

