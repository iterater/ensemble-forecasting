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
for b_flag in range(1, 1 << len(fc_set_all)):
    fc_set = ()
    ens_name = ''
    for q in range(len(fc_set_all)):
        if (b_flag >> q) & 1 == 1:
            fc_set += (fc_set_all[q],)
            ens_name += src_names[q]
    w_combination += [ens_name]
    # Processing window
    for w_i in range(len(w_length_set)):
        a_err = np.zeros(N-test_period_start)
        a_coeff = np.zeros((N-test_period_start, len(fc_set)+1))
        w = create_w_mask(w_length_set[w_i]+1, 0)
        for t in range(test_period_start, N):
            current_coeff = lse_coeff(fc_set, t-w_length_set[w_i], t, w)
            a_coeff[t-test_period_start] = current_coeff
            fc = np.full((1, T+1), current_coeff[len(fc_set)])
            for k in range(len(fc_set)):
                fc += fc_set[k][t, 0:T+1] * current_coeff[k]
            current_err = fc - m_fc[t]
            for fct in range(1, T+1):
                a_err[t-test_period_start] += abs(current_err[0, fct])/T
        w_errors[w_i, b_flag-1] = np.average(a_err)
        print(ens_name, w_length_set[w_i], w_errors[w_i, b_flag-1])
min_index = w_errors.argmin(axis=0)
for k in range(len(w_combination)):
    print(w_combination[k], min_index[k], w_length_set[min_index[k]], w_errors[min_index[k], k])
# h 62 126 6.76751004054
# s 61 124 7.42153549765
# hs 60 122 6.35916196227
# n 28 58 7.9138419639
# hn 60 122 6.1638420028
# sn 62 126 7.33471965393
# hsn 60 122 6.23706191128
plt.figure(2, figsize=(8, 5))
plt.plot(w_length_set*6, w_errors)
plt.legend(w_combination)
plt.title('MAE by window')
plt.xlabel('Window length, h')
plt.ylabel('MAE, cm')
plt.savefig('windows-error-2.png')
plt.close()


cc = lse_coeff((s[0:N, 1:T+1], n[0:N, 1:T+1], h[0:N, 1:T+1]), m_fc[0:N, 1:T+1], 1)
print(cc)
