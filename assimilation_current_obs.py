# Assimilation of observations within the forecast

import numpy as np
import matplotlib.pyplot as plt
import forecast_dists as dist
import scipy.optimize as opt
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import svm

import peak_plot_procedures as ppp


h = np.loadtxt('data/2011/2011080100_hiromb_GI_60x434.txt') - 37.356
s = np.loadtxt('data/2011/2011080100_swan_GI_48x434.txt')
n = np.loadtxt('data/2011/2011080100_noswan_GI_48x434.txt')
m = np.loadtxt('data/2011/2011080100_measurements_GI_2623.txt')

N = 430
T = 48

# preparing measurements forecast
m_fc = np.zeros((N, T+1))
for i in range(N):
    for j in range(T+1):
        m_fc[i, j] = m[i*6+j]

# data cut and shifting
shift_const = 0
m_average = np.average(m_fc)
s = s[0:N, 1:T+1] - np.average(s) + shift_const
n = n[0:N, 1:T+1] - np.average(n) + shift_const
h = h[0:N, 1:T+1] - np.average(h) + shift_const
m_fc = m_fc[0:N, 1:T+1] - m_average


def std_error(idx, t_lim, k):
    """
    Estimation of ensemble error for h*k[0]+s*k[1] with forecasts [idx, 0:t_lim]
    :param idx: Indices
    :param t_lim: Forecast size
    :param k: Ensemble weights
    :return: Random error
    """
    ens = h[idx, 0:t_lim] * k[0] + s[idx, 0:t_lim] * k[1]
    return np.std(ens - m_fc[idx, :t_lim])


def std_error_w(idx_start, idx_end, t_lim, k, masked):
    ens = h[idx_start:(idx_end + 1), 0:t_lim] * k[0] + s[idx_start:(idx_end + 1), 0:t_lim] * k[1]
    if not masked:
        return np.std(ens - m_fc[idx_start:(idx_end + 1), :t_lim])
    t_errs = np.array([])
    m_sel = m_fc[idx_start:(idx_end + 1), 0:t_lim]
    for ii in range(idx_end -idx_start + 1):
        mask_lim = (idx_end - ii + 1) * 6
        t_fc_erss = np.array(ens[ii, :min(t_lim, mask_lim)] - m_sel[ii, :min(t_lim, mask_lim)])
        t_errs = np.hstack((t_errs, t_fc_erss))
    return np.std(t_errs)

k0 = np.array([0.5, 0.5])
k_opt_glob = opt.minimize(lambda k: np.mean(dist.forecast_stdev_err(h * k[0] + s * k[1], m_fc)), k0).x
print('Global: ', k_opt_glob)

all_k1 = np.loadtxt('data\\information_assimilation\\all_k1.csv')
all_k2 = np.loadtxt('data\\information_assimilation\\all_k2.csv')

# Observation coverage
start_cov = 2

# Compute LSE historical ensembles
h_window = 30
h_k = np.loadtxt('data\\information_assimilation\\historical_ensemble.csv')
print('Historical ensemble', h_k.shape)

# Predictors for correlation estimate
xy = np.vstack((m_fc[h_window:, 0], m_fc[h_window:, 0] - h[h_window:, 0], m_fc[h_window:, 0] - s[h_window:, 0]))
final_state = np.vstack((all_k1[h_window:, -1], all_k2[h_window:, -1]))
opt_shift = h_k[h_window:].transpose() - final_state

# pca for optimal shift
# pca = PCA(n_components=2)
# pca_res = pca.fit_transform(opt_shift.transpose())
# plt.xlabel('PC#1')
# plt.ylabel('PC#2')
# plt.plot(pca_res[:, 0], pca_res[:, 1], 'o')
# plt.show()

# Prediction with liner regression
# predictor = linear_model.LinearRegression()
# predictor.fit(xy.transpose(), opt_shift.transpose())
# predicted_shift = predictor.predict(xy.transpose())

# Prediction with SVM
predictor = svm.SVR()
predictor.fit(xy.transpose(), opt_shift[0])
predicted_shift_0 = predictor.predict(xy.transpose())
predictor.fit(xy.transpose(), opt_shift[1])
predicted_shift_1 = predictor.predict(xy.transpose())
predicted_shift = np.vstack((predicted_shift_0, predicted_shift_1)).transpose()

# Testing results
cc = np.corrcoef(xy, opt_shift)
print('Corr(x/y, final state):\n', cc[0:len(xy), len(xy):])
h_err = np.array([std_error(i, T, h_k[i]) for i in range(h_window, N)])
g_err = np.array([std_error(i, T, k_opt_glob) for i in range(h_window, N)])
p_err = np.array([std_error(i, T, h_k[i] - predicted_shift[i - h_window]) for i in range(h_window, N)])
ppp.plot_biplot(h_err, p_err, 'H-ensemble error, cm', 'H-ensemble + predicted shift error, cm',
                'pics\\information_assimilation\\predicted_shift.png')
print('Historical to global:', np.mean(g_err-h_err))
print('Average improve (shifted to historical):', np.mean(h_err-p_err))
print('Comparing to global (shifted to historical):', np.mean(g_err-p_err))

#       LR     SVR
# G-H -0.097 -0.097
# P-H  0.104  0.830
# G-P  0.007  0.733

# Count autocorrelation
# MAX_AC_SHIFT = 20
# ac = np.ones((2, MAX_AC_SHIFT + 1))
# for i in range(1, MAX_AC_SHIFT + 1):
#     ac[0, i] = np.corrcoef(all_k1[h_window + MAX_AC_SHIFT:N, -1].flatten(),
#                            all_k1[h_window + MAX_AC_SHIFT - i:N - i, -1].flatten())[0, 1]
#     ac[1, i] = np.corrcoef(all_k2[h_window + MAX_AC_SHIFT:N, -1].flatten(),
#                            all_k2[h_window + MAX_AC_SHIFT - i:N - i, -1].flatten())[0, 1]
# print(ac)
# plt.figure(44, figsize=(8, 6))
# plt.grid()
# plt.plot(np.arange(0, MAX_AC_SHIFT + 1), ac[0], 'ro-')
# plt.plot(np.arange(0, MAX_AC_SHIFT + 1), ac[1], 'bo-')
# plt.legend(('k(h)', 'k(s)'))
# plt.xlabel('Forecast index shift')
# plt.ylabel('Autocorrelation')
# plt.savefig('pics\\information_assimilation\\state_ac.png')
# plt.close()

# Cross-validation (K-fold)
K = 10
h_err = np.array([])
p_err = np.array([])
ALL_N = xy.shape[1]
K_W = int(ALL_N / K)
predictor = svm.SVR(kernel='linear', C=1)
for kk in range(K):
    print('K-fold:', kk * K_W, '-', (kk + 1) * K_W - 1)
    test_index = (np.arange(ALL_N) >= kk * K_W) & (np.arange(ALL_N) < (kk + 1) * K_W)
    predictor.fit(xy.transpose()[~test_index], opt_shift[0][~test_index])
    predicted_shift_0 = predictor.predict(xy.transpose()[test_index])
    predictor.fit(xy.transpose()[~test_index], opt_shift[1][~test_index])
    predicted_shift_1 = predictor.predict(xy.transpose()[test_index])
    predicted_shift = np.vstack((predicted_shift_0, predicted_shift_1)).transpose()
    h_err_t = np.array([std_error(h_window + kk * K_W + i, T, h_k[h_window + kk * K_W + i]) for i in range(K_W)])
    p_err_t = np.array([std_error(h_window + kk * K_W + i, T, h_k[h_window + kk * K_W + i] - predicted_shift[i])
                        for i in range(K_W)])
    h_err = np.hstack((h_err, h_err_t))
    p_err = np.hstack((p_err, p_err_t))
ppp.plot_biplot(h_err, p_err, 'H-ensemble error, cm', 'H-ensemble + predicted shift error, cm',
                'pics\\information_assimilation\\predicted_shift_crossvalidation.png')
print('Cross-validation. Average improve (shifted to historical):', np.mean(h_err-p_err))
plt.figure(44, figsize=(8, 6))
plt.hist(h_err-p_err, bins=30, normed=True)
plt.xlabel('Error improve, cm')
plt.savefig('pics\\information_assimilation\\predicted_shift_crossvalidation_improve.png')
plt.close()

# SVM (linear kernel, C=1): +0.027
# SVM (linear kernel, C=1e2): +0.010
# SVM (linear kernel, C=1e-2): +0.027
# SVM (rbf kernel, C=1e-4): +0.019
