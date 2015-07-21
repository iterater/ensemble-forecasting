import numpy as np
from sklearn import tree
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

N = 430
T = 48

# preparing measurements forecast
m = np.loadtxt('data/2011/2011080100_measurements_GI_2623.txt')

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
ens_dist_index = np.loadtxt('data/2011/2011080100_ens_dist_index.txt')
ens_dist_index_matrix = np.full((len(ens_names), len(ens_names)), -1)
for i in range(len(ens_dist_index)):
    ens_dist_index_matrix[ens_dist_index[i, 0], ens_dist_index[i, 1]] = i
    ens_dist_index_matrix[ens_dist_index[i, 1], ens_dist_index[i, 0]] = i

# best ensemble classes
best_class = []
for t in range(N):
    errs = list(map(lambda x: x[t, 1], ens_err))
    best_class += [np.argmin(errs), ]
best_class = np.array(best_class).astype(int)
print(best_class)

# classification try
train_ratio = 0.6
for i in range(10):
    shuffled_index = np.arange(N)
    np.random.shuffle(shuffled_index)
    train_index = shuffled_index[0:int(train_ratio*N)]
    test_index = shuffled_index[int(train_ratio*N):]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(ens_dist_dtw[train_index], best_class[train_index])
    prediction = clf.predict(ens_dist_dtw[test_index])
    full_err = np.mean(ens_err[-1][test_index, 1])
    predicted_error = np.mean([ens_err[prediction[j]][test_index[j], 1] for j in range(len(prediction))])
    print('DIFF:', full_err - predicted_error)

# pca on distance vector
pca = PCA(n_components=2)
pca_res = pca.fit_transform(ens_dist_dtw)
pca = PCA(n_components=2)
pca_res = pca.fit_transform(ens_dist_dtw)
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
plt.xlabel('All-distance PC#1')
plt.ylabel('All-distance PC#2')
for i in range(7):
    plt.plot(pca_res[best_class == i, 0], pca_res[best_class == i, 1], 'o'+colors[i])
plt.savefig('pics\\2011\\all-distance-pca.png')
plt.close()
