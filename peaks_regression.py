import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math as m


def print_errors(e_arr, name):
    print(name, 'BIAS:', np.mean(e_arr), 'MAE:', np.mean(np.abs(e_arr)), 'STDEV:', np.std(e_arr))


# linear combination of sources
def lin_combination(x, k):
    t_n = len(x)
    k_n = len(k) - 1
    y1 = np.full(t_n, k[-1])
    for i in range(k_n):
        y1 += k[i] * x[:, i]
    return y1.flatten()


# Meshalkin loss function
def rho_meshalkin(x, y, k):
    lmbd = 0.1
    d = lin_combination(x, k) - y
    return sum([(1 - m.exp(-lmbd * d[i] * d[i] / 2)) / lmbd for i in range(len(d))])


# abs loss function
def rho_abs(x, y, k):
    d = lin_combination(x, k) - y
    return sum(np.abs(d))


# square loss function
def rho_square(x, y, k):
    d = lin_combination(x, k) - y
    return sum(d * d)


# get coefficients using optimization
def regr_coeff_opt(x, y, param_name):
    N = x.shape[1]
    mask = ~np.isnan(y)
    for i in range(N):
        mask &= ~np.isnan(x[:, i])
    x_flt = x[mask]
    y_flt = y[mask]
    for i in range(N):
        print_errors(y_flt - x_flt[:, i], str(i))
    c0 = np.concatenate((np.full(N, 1.0 / N), [0, ]))
    f = lambda k: rho_meshalkin(x_flt, y_flt, k)
    opt_res = opt.minimize(f, c0)
    c = opt_res.x
    test = lin_combination(x_flt, c)
    np.savetxt('data\\pre_parameter_estimation_'+param_name+'.txt', test)
    print_errors(y_flt - test, 'ENS')
    return c


# get coefficients using lse equations solve
def regr_coeff(x, y, param_name):
    N = x.shape[1]
    mask = ~np.isnan(y)
    for i in range(N):
        mask &= ~np.isnan(x[:, i])
    x_flt = x[mask]
    y_flt = y[mask]
    for i in range(N):
        print_errors(y_flt - x_flt[:, i], str(i))
    a = np.zeros((N + 1, N + 1))
    b = np.zeros(N + 1)
    for i in range(N):
        a[i, N] = np.sum(x_flt[:, i])
        a[N, i] = a[i, N]
        b[i] = np.sum(y_flt * x_flt[:, i])
        for j in range(N):
            a[i, j] = np.sum(x_flt[:, i] * x_flt[:, j])
    b[N] = np.sum(y_flt)
    a[N, N] = x_flt.shape[0]
    c = np.linalg.solve(a, b)
    test = lin_combination(x_flt, c)
    # lim = [min(min(test), min(y_flt)), max(max(test), max(y_flt))]
    # plt.figure(1, figsize=(5, 5))
    # plt.plot(y_flt, test, 'o')
    # plt.plot(lim, lim, 'k')
    # plt.xlabel('Original parameter '+param_name)
    # plt.ylabel('Ensemble parameter '+param_name)
    # plt.savefig('pics\\parameter_regression_'+param_name+'.png')
    # plt.close()
    # np.savetxt('data\\pre_parameter_estimation_'+param_name+'.txt', test)
    print_errors(y_flt - test, 'ENS')
    return c


def regr_coeff_all(pLevel):
    if pLevel == 80:
        bad_list = [19, 21, 23, 25, 35, 40, 41, 42, 43, 46, 48, 73, 74]  # for 80 cm
    else:
        bad_list = []
    p = np.genfromtxt('data\\PEAK_PARAMS_S1_'+str(pLevel).zfill(3)+'.csv', delimiter=',')
    p1 = np.zeros(p.shape)
    for i in range(4):
        p1[:, 4*i + 1] = p[:, 4*i + 2]
        p1[:, 4*i + 2] = p[:, 4*i + 1]
        p1[:, 4*i + 3] = p[:, 4*i + 3] + p[:, 4*i + 4]
        p1[:, 4*i + 4] = p[:, 4*i + 3] / (p[:, 4*i + 3] + p[:, 4*i + 4])
    # mask processing
    mask = ~np.isnan(p1[:, 0])
    for i in range(p1.shape[1]):
        mask &= ~np.isnan(p1[:, i])
    good_mask = np.ones(sum(mask), dtype=bool)
    good_mask[bad_list] = 0
    p1 = p1[mask][good_mask]
    src_names = ['BSM-WOWC-HIRLAM', 'BALTP-90M-GFS', 'HIROMB', 'M']
    param_names = ['H', 'T', 'W', 'D']
    res_coeff = []
    p_cnt = len(param_names)
    s_cnt = len(src_names) - 1
    for p_idx in range(p_cnt):
        print('Optimizing '+param_names[p_idx])
        src_data = p1[:, (1 + p_idx):(1 + p_cnt * s_cnt):p_cnt]
        m_data = p1[:, 1 + p_idx + p_cnt * s_cnt]
        cc = regr_coeff(src_data, m_data, param_names[p_idx])
        res_coeff += [cc, ]
    return np.array(res_coeff)


def revert_index(idx, N):
    i1 = int(m.floor(idx / N))
    i2 = idx % N
    return i2 * N + i1


# all peak levels
# prange = np.arange(60, 161, 20)
# for i in range(len(prange)):
#     cf = regr_coeff_all(prange[i])
#     np.savetxt('data\\regr_coeff_'+str(prange[i]).zfill(3)+'.txt', cf)
#     print(cf)

# only fixed peak level
p_level = 80
cf = regr_coeff_all(p_level)
np.savetxt('data\\regr_coeff_'+str(p_level).zfill(3)+'.txt', cf)
print(cf)
