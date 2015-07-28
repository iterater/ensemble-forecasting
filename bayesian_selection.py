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
mh = np.zeros((N, T))
for i in range(N):
    for j in range(T):
        tt = i*6+j + 1
        if tt >= len(m):
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


# best ensemble classes
best_class = []
for t in range(N):
    errs = list(map(lambda x: x[t, 1], ens_err))
    best_class += [np.argmin(errs), ]
best_class = np.array(best_class).astype(int)
print(best_class)


def mae_prob(err_set, de, max_e):
    t_err = np.array(err_set).flatten()
    res = []
    for x in np.arange(0, max_e, de):
        res += [sum((t_err >= x) & (t_err < x + de))/len(t_err), ]
    res[-1] += sum(t_err >= max_e)/len(t_err)
    return res


def select_by_mae_prob(test_err, err_set, de, max_e):
    p_err = mae_prob(err_set, de, max_e)
    t_idx = int(np.floor(test_err/de))
    if t_idx >= len(p_err):
        return p_err[-1]
    return p_err[t_idx]

# selection definition time
# selection = np.loadtxt('data\\2011\\bayesian_selection.txt')
# t_selection = np.zeros(N)
# for k in range(N):
#     for t in range(len(selection[k])):
#         if selection[k, t_selection[k]] != selection[k, t]:
#             t_selection[k] = t
# plt.figure(i)
# x = np.linspace(0, 48, 500)
# for i in range(len(ens_names)):
#     gkde = gaussian_kde(t_selection[best_class == i])
#     pdf = gkde(x)
#     plt.plot(x, pdf, colors[i])
# plt.xlim((0, 48))
# plt.xlabel('Final selection time, h')
# plt.ylabel('PDF')
# plt.legend(ens_names)
# plt.savefig('pics\\2011\\bayesian-selection-time.png')
# plt.close()

apriori_prob = [sum(best_class == i) / N for i in range(len(ens_names))]
print(apriori_prob)
selection = []
for k in range(N):
    p = apriori_prob
    print(k, ens_names[best_class[k]])
    res = [p, ]
    for t in range(T):
        new_p = []
        for en in range(len(ens_names)):
            t_err = np.abs(ens_fc[en] - mh)
            e = t_err[k, t]
            p_e = select_by_mae_prob(e, t_err[:, t], 1, 15)
            p_e_h = select_by_mae_prob(e, t_err[best_class == en][:, t], 1, 15)
            new_p += [p_e_h * p[en] / p_e, ]
        p = new_p
        norm = sum(p)
        res += [p / norm, ]
    # print(res)
    selection += [np.array(np.argmax(res, axis=1)).flatten(), ]
    # plotting
    plt.figure(k)
    plt.text(1, 0.9, 'Best: '+ens_names[best_class[k]])
    plt.plot(res)
    plt.legend(ens_names)
    plt.xlabel('Forecast time, h')
    plt.ylabel('Best guess probability')
    plt.savefig('pics\\2011\\bayesian_test\\fc'+str(k).zfill(3)+'.png')
    plt.close()
np.savetxt('data\\2011\\bayesian_selection.txt', selection)