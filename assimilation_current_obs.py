# Assimilation of observations within the forecast

import numpy as np
import matplotlib.pyplot as plt
import forecast_dists as dist
import scipy.optimize as opt
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
shift_const = 24
s = s[0:N, 1:T+1] - np.average(s) + shift_const
n = n[0:N, 1:T+1] - np.average(n) + shift_const
h = h[0:N, 1:T+1] - np.average(h) + shift_const
m_fc = m_fc[0:N, 1:T+1]


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
k_opt_glob = opt.minimize(lambda k: np.mean(dist.forecast_stdev_err(h * k[0] + s * k[1], m_fc)), k0)
print('Global: ', k_opt_glob.x)

all_k1 = np.loadtxt('data\\information_assimilation\\all_k1.csv')
all_k2 = np.loadtxt('data\\information_assimilation\\all_k2.csv')

# Observation coverage
start_cov = 2

# Compute LSE historical ensembles
h_window = 30
if False:
    h_k = np.zeros((h_window, 2))
    for i in range(h_window, N):
        print(i)
        k0 = np.array([0.5, 0.5])
        k_opt = opt.minimize(lambda k: std_error_w(i - h_window, i - 1, T, k, True), k0)
        h_k = np.vstack((h_k, k_opt.x))
    np.savetxt('data\\information_assimilation\\historical_ensemble.csv', h_k)
else:
    h_k = np.loadtxt('data\\information_assimilation\\historical_ensemble.csv')

# Assimilation of current data
