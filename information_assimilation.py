import numpy as np
import matplotlib.pyplot as plt
import forecast_dists as dist
import scipy.optimize as opt


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

k0 = np.array([0.5, 0.5])
k_opt_glob = opt.minimize(lambda k : np.mean(dist.forecast_stdev_err(h * k[0] + s * k[1], m_fc)), k0)
print('Global: ', k_opt_glob.x)

# Prediction using previous ensemble states
# [-1, 6], [-2, 12], [-3, 18], [-4, 24], [-5, 30] [-6, 36] [-7, 42] [-8, 48]


# Coverage testing
start_cov = 2
all_final = []
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
    save_array = np.vstack((np.arange(start_cov, T + 1), k_all.transpose(), np.array(errs), np.array(errs_tot)))
    np.savetxt('data\\information_assimilation\\case_' + str(id).zfill(3) + '_track.csv', save_array.transpose())
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

all_final_np = np.array(all_final)
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
