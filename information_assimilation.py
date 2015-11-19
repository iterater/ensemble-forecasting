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
    h_k = np.zeros((w, 2))
    for i in range(w, N):
        print(i)
        k0 = np.array([0.5, 0.5])
        k_opt = opt.minimize(lambda k: std_error_w(i - h_window, i - 1, T, k, True), k0)
        h_k = np.vstack((h_k, k_opt.x))
    np.savetxt('data\\information_assimilation\\historical_ensemble.csv', h_k)
else:
    h_k = np.loadtxt('data\\information_assimilation\\historical_ensemble.csv')

# Kalman filter
print('cov(X)')
P = np.matrix(np.cov(np.vstack((all_k1[:, -1].transpose(), all_k2[:, -1].transpose()))))
print(P)
obs_error = np.vstack(((all_k1[7:(N - 1), 6-start_cov] - all_k1[8:, -1]).transpose(),
                       (all_k2[7:(N - 1), 6-start_cov] - all_k2[8:, -1]).transpose()))
for tt in range(2, 8):
    obs_error = np.vstack((obs_error,
                           (all_k1[(8 - tt):(N - tt), tt * 6 - start_cov] - all_k1[8:, -1]).transpose(),
                           (all_k2[(8 - tt):(N - tt), tt * 6 - start_cov] - all_k2[8:, -1]).transpose()))
R = np.matrix(np.cov(obs_error))
print('cov(E(Y))')
print(R)
C = np.matrix(np.zeros((14, 2)))
for i in range(14):
    C[i, i & 1] = 1
print(C)
# C P* C' + R
K = np.dot(np.dot(P, C.transpose()), (np.dot(np.dot(C, P), C.transpose()) + R).getI())
print(K)
assimilated_k = np.zeros((8, 2))
errs = np.zeros((8, 3))
for i in range(8, N):
    # X = np.matrix([all_k1[i - 8, -1], all_k2[i - 8, -1]]).transpose()
    # X = np.matrix(k_opt_glob.x).transpose()
    X = np.matrix(h_k[i]).transpose()
    Y = []
    for di in range(1, 8):
        Y += [all_k1[i - di, di * 6 - start_cov], all_k2[i - di, di * 6 - start_cov]]
    Ym = np.matrix(Y).transpose()
    X1 = X + np.dot(K, Ym - np.dot(C, X))
    assimilated_k += [X1[0, 0], X1[1, 0]]
    errs = np.vstack((errs,
                      [std_error(i, T, [X1[0, 0], X1[1, 0]]), std_error(i, T, k_opt_glob.x), std_error(i, T, h_k[i])]))
errs = np.array(errs)
print('[ASSIM, GLOBAL, HISTORY]: ', np.mean(errs[h_window:, :], axis=0))
ppp.plot_biplot(errs[h_window:, 0].flatten(), errs[h_window:, 2].flatten(), 'Assimilated ensemble error',
                'Historical ensemble error', 'pics\\information_assimilation\\ensemble_assimilation_biplot.png')

# COMPARE TO HISORICALLY-TRAINED !!!!

# Distance plot
if False:
    start_plot = 5
    all_d = []
    for i in range(N):
        d_1 = all_k1[i, (start_plot - start_cov):] - all_k1[i, -1]
        d_2 = all_k2[i, (start_plot - start_cov):] - all_k2[i, -1]
        d = np.sqrt(d_1 * d_1 + d_2 * d_2)
        all_d += [d]
    cov_arg = np.arange(start_plot, T + 1)
    # plt.figure(44, figsize=(8, 6))
    # plt.plot(cov_arg, np.mean(all_d, axis=0), 'b-', label='Mean')
    # plt.plot(cov_arg[(6 - start_plot)::6], np.mean(all_d, axis=0)[(6 - start_plot)::6], 'bo')
    # plt.plot(cov_arg, np.std(all_d, axis=0), 'r-', label='StDev')
    # plt.plot(cov_arg[(6 - start_plot)::6], np.std(all_d, axis=0)[(6 - start_plot)::6], 'ro')
    # plt.xlim((0, T))
    # plt.legend()
    # plt.xlabel('Observation coverage')
    # plt.ylabel('Distance')
    # plt.savefig('pics\\information_assimilation\\state_dist.png')
    # plt.close()
    # plt.figure(44, figsize=(18, 6))
    # plt.boxplot(np.transpose(all_d).tolist(), positions=cov_arg, whis=[0, 100])
    # plt.xlim((0, T))
    # plt.xlabel('Observation coverage')
    # plt.ylabel('Distance')
    # plt.savefig('pics\\information_assimilation\\state_dist_boxplot.png')
    # plt.close()


# Coverage testing
if False:
    all_final = []
    all_k1 = []
    all_k2 = []
    all_err = []
    all_err_tot = []

    for id in range(N):
        print(id)
        res = []
        errs = []
        errs_tot = []
        for tt in range(start_cov, T + 1):
            k0 = np.array([0.5, 0.5])
            # print(k0, std_error(id, T, k0))
            k_opt = opt.minimize(lambda k : std_error(id, tt, k), k0)
            res += [k_opt.x]
            errs += [std_error(id, tt, k_opt.x)]
            errs_tot += [std_error(id, T, k_opt.x)]
            # print(k_opt.x, std_error(id, T, k_opt.x))
            # plt.plot(h[id] + np.mean(m_fc[id]) - np.mean(h[id]), ':')
            # plt.plot(s[id] + np.mean(m_fc[id]) - np.mean(s[id]), ':')
            # ens = h[id] * k_opt.x[0] + s[id] * k_opt.x[1]
            # plt.plot(ens + np.mean(m_fc[id]) - np.mean(ens))
            # plt.plot(m_fc[id], 'o')
            # plt.legend(('h', 's', 'Ens', 'M'))
            # plt.show()
        k_all = np.array(res)
        all_k1 += [k_all.transpose()[0]]
        all_k2 += [k_all.transpose()[1]]
        all_err += [np.array(errs)]
        all_err_tot += [np.array(errs_tot)]
        all_final += [k_all[-1]]
        # print(k_all)
        # plt.figure(44, figsize=(6, 6))
        # plt.plot(k_all[:, 0].flatten(), k_all[:, 1].flatten())
        # plt.plot(k_all[:, 0].flatten(), k_all[:, 1].flatten(), 'o', markersize=3)
        # plt.plot(k_all[0, 0], k_all[0, 1], '^')
        # plt.plot(k_all[-1, 0], k_all[-1, 1], 's')
        # plt.plot(k_opt_glob.x[0], k_opt_glob.x[1], '*')
        # plt.xlabel(r'$K_H$')
        # plt.ylabel(r'$K_S$')
        # plt.legend(('Track', 'Steps', 'Start', 'Finish', 'Global ensemble'))
        # plt.savefig('pics\\information_assimilation\\all_tracks\\case_' + str(id).zfill(3) + '_track.png')
        # plt.close()
        # plt.figure(44, figsize=(8, 6))
        # plt.plot(np.arange(start_cov, T + 1), errs)
        # plt.plot(np.arange(start_cov, T + 1), errs_tot)
        # plt.legend(('Coverage', 'Whole forecast'))
        # plt.xlim((0, T))
        # plt.xlabel('Observation coverage')
        # plt.ylabel('StDev(E)')
        # plt.savefig('pics\\information_assimilation\\all_tracks\\case_' + str(id).zfill(3) + '_error.png')
        # plt.close()
    np.savetxt('data\\information_assimilation\\all_k1.csv', all_k1)
    np.savetxt('data\\information_assimilation\\all_k2.csv', all_k2)
    np.savetxt('data\\information_assimilation\\all_err.csv', all_err)
    np.savetxt('data\\information_assimilation\\all_err_tot.csv', all_err_tot)
    # all_final_np = np.array(all_final)
    # plt.figure(44, figsize=(6, 6))
    # lims = (np.min(all_final_np), np.max(all_final_np))
    # d = lims[1] - lims[0]
    # lims = (lims[0] - 0.1 * d, lims[1] + 0.1 * d)
    # plt.plot(all_final_np[:, 0].flatten(), all_final_np[:, 1].flatten(), 'o')
    # plt.plot(lims, [1 - lims[0], 1 - lims[1]], lw=2)
    # plt.plot(k_opt_glob.x[0], k_opt_glob.x[1], '*', markersize=12)
    # plt.xlim(lims)
    # plt.ylim(lims)
    # plt.savefig('pics\\information_assimilation\\ensemble_h_s_opt_coeffs.png')
    # plt.close()
