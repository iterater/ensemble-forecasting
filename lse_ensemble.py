# Prepare ensemble forecasts with simple LSE training
import numpy as np
import math as mt
import forecast_dists as dist

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

src_names = ['h', 's', 'n']
fc_set_all = (h, s, n)

def compute_and_save_errors(fcs, name, file_name):
    err_mae = dist.forecast_dist_mae(fcs, m_fc - shift_const)
    err_dtw = dist.forecast_dist_dtw(fcs, m_fc - shift_const)
    err = np.vstack((err_mae, err_dtw)).transpose()
    print(name, 'DTW', np.mean(err_dtw), 'MAE', np.mean(err_mae))
    np.savetxt(file_name, err)


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


# original errors
for i in range(len(fc_set_all)):
    compute_and_save_errors(fc_set_all[i] - shift_const, src_names[i],
                            'data/2011/2011080100_original_'+src_names[i]+'_GI_'+str(T)+'x'+str(N)+'_err.txt')

# building all ensembles
ens_set = ()
ens_names = ()
for b_flag in range(1, 1 << len(fc_set_all)):
    fc_set = ()
    ens_name = ''
    for q in range(len(fc_set_all)):
        if (b_flag >> q) & 1 == 1:
            fc_set += (fc_set_all[q],)
            ens_name += src_names[q]
    current_coeff = lse_coeff(fc_set, m_fc, 1)
    e_fc = np.full(np.shape(fc_set[0]), current_coeff[len(fc_set)] - shift_const)
    ens_set += (e_fc, )
    ens_names += (ens_name, )
    for k in range(len(fc_set)):
        e_fc += fc_set[k] * current_coeff[k]
    print(ens_name, current_coeff)
    compute_and_save_errors(e_fc, ens_name, 'data/2011/2011080100_ens_'+ens_name+'_GI_'+str(T)+'x'+str(N)+'_err.txt')
    np.savetxt('data/2011/2011080100_ens_'+ens_name+'_GI_'+str(T)+'x'+str(N)+'.txt', e_fc)

dist_all_index = []
dist_all_mae = []
dist_all_dtw = []
for i in range(len(ens_names)):
    for j in range(i):
        dist_all_dtw += [dist.forecast_dist_dtw(ens_set[i], ens_set[j]), ]
        dist_all_mae += [dist.forecast_dist_mae(ens_set[i], ens_set[j]), ]
        dist_all_index += [[i, j], ]
np.savetxt('data/2011/2011080100_ens_dist_mae.txt', dist_all_mae)
np.savetxt('data/2011/2011080100_ens_dist_dtw.txt', dist_all_dtw)
np.savetxt('data/2011/2011080100_ens_dist_index.txt', dist_all_index, fmt='%i')
with open('data/2011/2011080100_ens_dist_names.txt', 'w') as t_file:
    t_file.write('\n'.join(ens_names))