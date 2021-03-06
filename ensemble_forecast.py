import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

N = 430
T = 48
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# preparing measurements data
m = np.loadtxt('data/2011/2011080100_measurements_GI_2623.txt')
m_s1 = np.loadtxt('data/2011/2011080100_measurements_S1_2623.txt')
MHT = 70  # maximum history length
mh = np.zeros((N, MHT))
mh_s1 = np.zeros((N, MHT))
for i in range(N):
    for j in range(MHT):
        tt = i*6-j
        if tt < 0:
            mh[i, j] = np.nan
            mh_s1[i, j] = np.nan
        else:
            mh[i, j] = m[tt]
            mh_s1[i, j] = m_s1[tt]

# preparing measurements forecast
m_fc = np.zeros((N, T+1))
for i in range(N):
    for j in range(T+1):
        m_fc[i, j] = m[i*6+j]
m_fc = m_fc[0:N, 1:T+1]

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

# best ensemble classes
best_class = []
for t in range(N):
    errs = list(map(lambda x: x[t, 1], ens_err))
    best_class += [np.argmin(errs), ]
best_class = np.array(best_class).astype(int)
print(best_class)

# error pdf
# for i in range(len(ens_names)):
#     plt.figure(i)
#     err0 = np.array(np.abs(ens_fc[i][:, 0] - m_fc[:, 0])).flatten()
#     x = np.linspace(min(err0), max(err0), 100)
#     gkde = gaussian_kde(err0[best_class == i])
#     pdfTrue = gkde(x)
#     gkde = gaussian_kde(err0[best_class != i])
#     pdfFalse = gkde(x)
#     plt.xlim((min(err0), max(err0)))
#     plt.plot(x, pdfTrue)
#     plt.plot(x, pdfFalse)
#     plt.xlabel('MAE(t=0), cm')
#     plt.ylabel('PDF')
#     plt.legend(['Best is '+ens_names[i], 'Best is not '+ens_names[i]])
#     plt.savefig('pics\\2011\\best-not-best-'+str(i).zfill(2)+'.png')
#     plt.close()

# error dtw vs mae
# plt.figure(44)
# plt.xlabel('MAE, cm')
# plt.ylabel('DTW, cm')
# for i in range(len(ens_names)):
#     plt.plot(ens_err[i][:, 0], ens_err[i][:, 1], colors[i]+'o')
# plt.legend(ens_names)
# plt.savefig('pics\\2011\\mae-vs-dtw.png')
# plt.close()

# autocorrelation
# ac = []
# max_dt = 60
# for dt in range(max_dt + 1):
#     t_ac = []
#     for i in range(len(ens_names)):
#         x = ens_err[i][0:N-dt, 1]
#         y = ens_err[i][dt:N, 1]
#         cc = np.corrcoef(x, y)
#         t_ac += [cc[0, 1], ]
#     ac += [t_ac, ]
# plt.figure(44)
# plt.xlabel('Time shift, h')
# plt.ylabel('Correlation')
# plt.plot(ac)
# plt.legend(ens_names)
# plt.savefig('pics\\2011\\ac-dtw.png')
# plt.close()



# error prediction
train_ratio = 0.6
history_view_window = 48
available_start = int(np.ceil(history_view_window/6))
for i in range(10):
    shuffled_index = np.arange(N - available_start) + available_start
    np.random.shuffle(shuffled_index)
    train_index = shuffled_index[0:int(train_ratio*N)]
    test_index = shuffled_index[int(train_ratio*N):]
    predicted_err = []
    for ens_i in range(len(ens_names)):
        ens_idx_selection = (ens_dist_index[:, 0].flatten() == ens_i) | (ens_dist_index[:, 1].flatten() == ens_i)
        train_arg = np.hstack((mh[train_index, 0:history_view_window],
                               mh_s1[train_index, 0:history_view_window],
                               ens_dist_dtw[train_index][:, ens_idx_selection]))
        test_arg = np.hstack((mh[test_index, 0:history_view_window],
                              mh_s1[test_index, 0:history_view_window],
                              ens_dist_dtw[test_index][:, ens_idx_selection]))
        # train_arg = mh[train_index, 0:history_view_window]
        # test_arg = mh[test_index, 0:history_view_window]
        train_val = ens_err[ens_i][train_index, 1]
        # regr = svm.SVR(kernel='rbf', gamma=0.3)
        regr = svm.SVR()
        regr.fit(train_arg, train_val)
        predicted_err += [regr.predict(test_arg), ]
    predicted_err = np.array(predicted_err).transpose()
    ens_selection = np.argmin(predicted_err, axis=1)
    err_selection = [ens_err[ens_selection[j]][test_index[j], 1] for j in range(len(ens_selection))]
    print('DIFF: ', np.mean(ens_err[-1][test_index, 1]) - np.mean(err_selection))



# classification try
# train_ratio = 0.6
# for i in range(10):
#     shuffled_index = np.arange(N)
#     np.random.shuffle(shuffled_index)
#     train_index = shuffled_index[0:int(train_ratio*N)]
#     test_index = shuffled_index[int(train_ratio*N):]
#     clf = tree.DecisionTreeClassifier()
#     clf = clf.fit(ens_dist_dtw[train_index], best_class[train_index])
#     prediction = clf.predict(ens_dist_dtw[test_index])
#     full_err = np.mean(ens_err[-1][test_index, 1])
#     predicted_error = np.mean([ens_err[prediction[j]][test_index[j], 1] for j in range(len(prediction))])
#     print('DIFF:', full_err - predicted_error)

# pca on distance vector
# pca = PCA(n_components=2)
# pca_res = pca.fit_transform(ens_dist_dtw)
# plt.xlabel('All-distance PC#1')
# plt.ylabel('All-distance PC#2')
# for i in range(7):
#     plt.plot(pca_res[best_class == i, 0], pca_res[best_class == i, 1], 'o'+colors[i])
# plt.savefig('pics\\2011\\all-distance-pca.png')
# plt.close()
