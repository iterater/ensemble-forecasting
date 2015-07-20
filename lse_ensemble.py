# Prepare ensemble forecasts with simple LSE training
import numpy as np
import math as mt

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
h = h[0:N, 1:T+1] + shift_const
s = s[0:N, 1:T+1] + shift_const
n = n[0:N, 1:T+1] + shift_const
m_fc = m_fc[0:N, 1:T+1] + shift_const


def forecast_dist(fc1, fc2):
    fc_diff = np.absolute(fc1 - fc2)
    return np.mean(fc_diff, axis=1)


def create_w_mask(n, t, k, skip_extended):
    """
    Create exponential mask
    :param n: Row count
    :param t: Col count
    :param k: Exponential scale
    :param skip_extended: Skip parts after last forecast start
    :return: Mask array with dim [n, t]
    """
    w_res = np.zeros((n, t))
    for t_i in range(n):
        for t_j in range(t):
            if (t_i*6+t_j >= (n - 1)*6) and skip_extended:
                w_res[t_i, t_j] = 0
            else:
                w_res[t_i, t_j] = mt.exp(k*(t_i*6+t_j-(n-1)*6))
    return w_res


def lse_coeff(fcs, m_fc, w):
    """
    Training ensemble coefficients using less-square errors
    :param fcs: Forecasts sequence of 2-d arrays
    :param m_fc: Measurements as forecasts
    :param w: Weight mask
    :return: Array of coefficients [a1, ... an, c]
    """
    a = np.zeros((len(fcs) + 1, len(fcs) + 1))
    b = np.zeros(len(fcs) + 1)
    for i in range(len(fcs)):
        b[i] = np.sum(w*fcs[i]*m_fc)
        a[len(fcs), i] = np.sum(w*fcs[i])
        a[i, len(fcs)] = a[len(fcs), i]
        for j in range(len(fcs)):
            a[i, j] = np.sum(w*fcs[i]*fcs[j])
    b[len(fcs)] = np.sum(w*m_fc)
    a[len(fcs), len(fcs)] = np.sum(w)
    return np.linalg.solve(a, b)


src_names = ['h', 's', 'n']
w_combination = []
fc_set_all = (h, s, n)
for b_flag in range(1, 1 << len(fc_set_all)):
    fc_set = ()
    ens_name = ''
    for q in range(len(fc_set_all)):
        if (b_flag >> q) & 1 == 1:
            fc_set += (fc_set_all[q],)
            ens_name += src_names[q]
    current_coeff = lse_coeff(fc_set, m_fc, 1)
    e_fc = np.full(np.shape(fc_set[0]), current_coeff[len(fc_set)] - shift_const)
    for k in range(len(fc_set)):
        e_fc += fc_set[k] * current_coeff[k]
    print(ens_name, current_coeff, 'ERR', np.mean(forecast_dist(e_fc, m_fc - shift_const)))
    np.savetxt('data/2011/2011080100_ens_'+ens_name+'_GI_'+str(T)+'x'+str(N)+'.txt', e_fc)
