import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
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


def forecastError(fc):
    sum = np.zeros(N)
    for i in range(T+1):
        sum += abs(fc[0:N:,i]-m[i:N*6+i:6]) / T
    return sum

indexRange = np.arange(N)
plt.figure(1, figsize=(12, 5))
plt.plot(indexRange, forecastError(h), indexRange, forecastError(s), indexRange, forecastError(n))
plt.legend(['Hiromb', 'Swan', 'No-Swan'])
plt.title('Sources MAE')
plt.xlabel('Forecast index')
plt.ylabel('Forecast MAE, cm')
plt.savefig('sources-mae.png')
plt.close()


def create_w_mask(w_len, w_k):
    w_res = np.zeros((w_len, T))
    for t_i in range(w_len):
        for t_j in range(T):
            if t_i*6+t_j >= (w_len - 1)*6:
                w_res[t_i, t_j] = 0
            else:
                w_res[t_i, t_j] = mt.exp(w_k*(t_i*6+t_j-(w_len-1)*6))
    return w_res




def run_test_forecast(fcs, fc_index, window, w):
    c = lse_coeff(fcs, fc_index-window, fc_index, w)
    t_fc = np.full((1, T+1), c[len(fcs)])
    for k in range(len(fcs)):
        t_fc += fcs[k][t, 0:T+1] * c[k]
    t_err = t_fc - m_fc[t]
    res_err = 0.0
    for fct in range(1, T+1):
        res_err += abs(t_err[0, fct])/T
    return res_err


test_period_start = 150
fc_set_all = (h, s, n)


# Process weighting function
def process_k():
    # Processing k
    # 122 err(0)= 6.23706191128
    w_k_w = np.arange(0, 0.2, 0.0001)
    w_errors = np.zeros(len(w_k_w))
    w_l = 122
    for w_i in range(len(w_k_w)):
        a_err = np.zeros(N-test_period_start)
        w = create_w_mask(w_l+1, w_k_w[w_i])
        for t in range(test_period_start, N):
            a_err[t-test_period_start] = run_test_forecast(fc_set, t, w_l, w)
        w_errors[w_i] = np.average(a_err)
        print(w_k_w[w_i], w_errors[w_i])


def process_all_variants():
    src_names = ['h', 's', 'n']
    w_combination = []
    w_length_set = np.arange(2, test_period_start + 1, 2)
    w_errors = np.zeros((len(w_length_set), (1 << len(fc_set_all)) - 1))
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


def process_distance():
    src_names = ['h', 's', 'n']
    w_combination = []
    fc_combination = ()
    for b_flag in range(1, 1 << len(fc_set_all)):
        fc_set = ()
        ens_name = ''
        for q in range(len(fc_set_all)):
            if (b_flag >> q) & 1 == 1:
                fc_set += (fc_set_all[q],)
                ens_name += src_names[q]
        w_combination += [ens_name]
        fc_combination += (fc_set,)

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


process_all_variants()
