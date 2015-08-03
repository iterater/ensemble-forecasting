import os
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import scaling_forecast_procedures as sfp
import peak_plot_procedures as ppp

s1 = np.loadtxt('data\\ensemble-forecasts\\2013102012-BSM-WOWC-HIRLAM-S1.txt')
s2 = np.loadtxt('data\\ensemble-forecasts\\2013102012-BALTP-90M-GFS-S1.txt')[:, 0:61]
s3 = np.loadtxt('data\\ensemble-forecasts\\2013102012-HIROMB-S1.txt') - 37.4
m = np.loadtxt('data\\ensemble-forecasts\\2013102012-M-GI_C1NB_C1FG_SHEP_restored.txt')[:, 2]
print(np.shape(s1))
print(np.shape(s2))
print(np.shape(s3))
print(np.shape(m))

N = 281
T = 60

m_fc = np.zeros((N, T+1))
for i in range(N):
    for j in range(T+1):
        m_fc[i, j] = m[i*6+j]
print(np.shape(m_fc))


def create_w_mask(w_len, w_k):
    w_res = np.zeros((w_len, T))
    for t_i in range(w_len):
        for t_j in range(T):
            if t_i*6+t_j >= (w_len - 1)*6:
                w_res[t_i, t_j] = 0
            else:
                w_res[t_i, t_j] = mt.exp(w_k*(t_i*6+t_j-(w_len-1)*6))
    return w_res

def create_w_mask_from_level(w_len, w_scale):
    w_res = np.zeros((w_len, T))
    for t_i in range(w_len):
        for t_j in range(T):
            if t_i*6+t_j >= (w_len - 1)*6:
                w_res[t_i, t_j] = 0
            else:
                w_res[t_i, t_j] = mt.exp(m_fc[t_i, t_j] * w_scale)
    return w_res


def lse_coeff(fcs, start_index, stop_index, w):
    a = np.zeros((len(fcs) + 1, len(fcs) + 1))
    b = np.zeros(len(fcs) + 1)
    for i in range(len(fcs)):
        b[i] = np.sum(w*fcs[i][start_index:stop_index+1, 1:T+1]*m_fc[start_index:stop_index+1, 1:T+1])
        a[len(fcs), i] = np.sum(w*fcs[i][start_index:stop_index+1, 1:T+1])
        a[i, len(fcs)] = a[len(fcs), i]
        for j in range(len(fcs)):
            a[i, j] = np.sum(w*fcs[i][start_index:stop_index+1, 1:T+1]*fcs[j][start_index:stop_index+1, 1:T+1])
    b[len(fcs)] = np.sum(w*m_fc[start_index:stop_index+1, 1:T+1])
    a[len(fcs), len(fcs)] = np.sum(w)
    return np.linalg.solve(a, b)


def forecast_peak_error(test_fc, base_fc, base_peak_pos):
    peak_pos_test = np.zeros(len(test_fc))
    peak_pos_base = np.zeros(len(base_fc))
    peak_level_test = np.zeros(len(base_fc))
    peak_level_base = np.zeros(len(base_fc))
    for i in range(len(test_fc)):
        for j in range(len(test_fc[i])):
            if (j > base_peak_pos[i] - 20) and (j < base_peak_pos[i] + 20):
                if test_fc[i, j] > test_fc[i, peak_pos_test[i]]:
                    peak_pos_test[i] = j
                    peak_level_test[i] = test_fc[i, j]
                if base_fc[i, j] > base_fc[i, peak_pos_base[i]]:
                    peak_pos_base[i] = j
                    peak_level_base[i] = base_fc[i, j]
    l_err = peak_level_test - peak_level_base
    t_err = (peak_pos_test - peak_pos_base).astype(float)
    return np.vstack((t_err, l_err))


def forecast_wmae(fc, pl):
    ae = np.abs(fc - m_fc)
    msk_l = m_fc < pl
    ae[msk_l] = np.nan
    res = np.nanmean(ae, axis=1)
    return res


def forecast_error(fc):
    ae = np.abs(fc - m_fc)
    mae = np.mean(ae, axis=1)
    return mae


pLevel = 80
source_v_scale_mode = sfp.ScaleType.no_scale
ensemble_v_scale_mode = sfp.ScaleType.add_all_scale
level_scale_flag_string = ''
w = create_w_mask(N, 0)
# w = create_w_mask_from_level(N, 0.05)
src_set = (s1, s2, s3)
c = lse_coeff(src_set, 0, N - 1, w)
print(c)
fc_def = np.full((N, T + 1), c[3])
e_def = []
for src_i in range(3):
    fc_def += src_set[src_i] * c[src_i]
indexRange = np.arange(N)

p = np.genfromtxt('data\\PEAK_PARAMS_S1_'+str(pLevel).zfill(3)+'.csv', delimiter=',')
pc = np.loadtxt('data\\regr_coeff_'+str(pLevel).zfill(3)+'.txt')
mask = ~np.isnan(p[:, 0])
for i in range(13):
    mask &= ~np.isnan(p[:, i])
p_flt = p[mask]

# removing bad peaks
bad_list = [19, 21, 23, 25, 35, 40, 41, 42, 43, 46, 48, 73, 74]  # for 80 cm
good_mask = np.ones(len(p_flt), dtype=bool)
good_mask[bad_list] = 0
p_flt = p_flt[good_mask]

t_idx = np.arange(0, 61)
colors = ('b', 'g', 'r')
peak_l = [pLevel, pLevel, pLevel]
dir_name = 'pics\\all-forecasts-' + level_scale_flag_string + 't-scale-pl' + str(pLevel).zfill(3)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
res_params_array = []
for i in range(len(p_flt)):
    print('FC #' + str(i))
    # plt.figure(i, figsize=(10, 7))
    params = []
    res_params = pc[:, 3].flatten()
    for src_i in range(3):
        p1 = p_flt[i, (src_i * 4 + 1):(src_i * 4 + 5)]
        p2 = [p1[1], p1[0], p1[2] + p1[3], p1[2] / (p1[2] + p1[3])]
        params += [p2]
        res_params = res_params + np.array(p2) * pc[:, src_i].flatten()
        # plt.plot(t_idx, src_set[src_i][p_flt[i, 0]], colors[src_i] + '-')
    # plt.plot(t_idx, m_fc[p_flt[i, 0]], 'o', color='k')
    peak_t = [res_params[1] - res_params[2] * res_params[3],
              res_params[1],
              res_params[1] + res_params[2] * (1.0 - res_params[3])]
    peak_l[1] = res_params[0]
    # plt.plot(peak_t, peak_l, '^', markersize=10)
    # plt.plot(t_idx, fc_def[p_flt[i, 0]], 'k-')
    e_fc = np.full(T + 1, c[3])
    for src_i in range(3):
        src_set[src_i][p_flt[i, 0]] = sfp.transform_forecast(src_set[src_i][p_flt[i, 0]], params[src_i],
                                                             res_params, T, source_v_scale_mode)
        # plt.plot(t_idx, src_set[src_i][p_flt[i, 0]], colors[src_i] + '--')
        e_fc += src_set[src_i][p_flt[i, 0]] * c[src_i]
    # plt.plot(t_idx, e_fc, 'k--')
    e_fc_v = sfp.scale_peak_vertically(e_fc, res_params, T, ensemble_v_scale_mode)
    res_params_array = np.concatenate((res_params_array, res_params))
    # plt.plot(t_idx, e_fc_v, 'k:')
    # plt.xlim([0, T])
    # plt.xlabel('Forecast time, h')
    # plt.ylabel('Level, cm')
    # plt.savefig(dir_name + '\\forecast-' + str(i).zfill(3) + '.png')
    # plt.close()
res_params_array = res_params_array.reshape((len(p_flt), 4))

peak_index = p_flt[:, 0].flatten().astype(int)
peak_mask = np.zeros(len(m_fc), dtype=bool)
peak_mask[peak_index] = 1

# default peak errror
fc_def_err = forecast_peak_error(fc_def[peak_mask], m_fc[peak_mask], res_params_array[:, 1].flatten())

# basic ensemble on tuned sources
c = lse_coeff(src_set, 0, N - 1, w)
fc_1 = np.full((N, T + 1), c[3])
for src_i in range(3):
    fc_1 += src_set[src_i] * c[src_i]
fc_0_err = forecast_peak_error(fc_1[peak_mask], m_fc[peak_mask], res_params_array[:, 1].flatten())

# additional peak forcing
for i in range(len(p_flt)):
    print("Forcing FC", p_flt[i, 0])
    fc_1[p_flt[i, 0]] = sfp.scale_peak_vertically(fc_1[p_flt[i, 0]], res_params_array[i],
                                                  T, ensemble_v_scale_mode)
fc_1_err = forecast_peak_error(fc_1[peak_mask], m_fc[peak_mask], res_params_array[:, 1].flatten())

print('BIAS E. default (T, L):', np.mean(fc_def_err, axis=1))
print('BIAS E0 (T, L):', np.mean(fc_0_err, axis=1))
print('BIAS E1 (T, L):', np.mean(fc_1_err, axis=1))
print('STDEV E. default (T, L):', np.std(fc_def_err, axis=1))
print('STDEV E0 (T, L):', np.std(fc_0_err, axis=1))
print('STDEV E1 (T, L):', np.std(fc_1_err, axis=1))
print('Average improve  E0 (T, L):', np.mean(np.abs(fc_def_err) - np.abs(fc_0_err), axis=1))
print('Average improve E1 (T, L):', np.mean(np.abs(fc_def_err) - np.abs(fc_1_err), axis=1))

ppp.plot_biplot(np.abs(fc_def_err[1]), np.abs(fc_1_err[1]), 'Default ensemble, AE(H), cm',
                'E. with shifted sources and peak forcing, AE(H), cm', 'pics\\bp_l_def_vs_e1.png')
ppp.plot_biplot(np.abs(fc_def_err[1]), np.abs(fc_0_err[1]), 'Default ensemble, AE(H), cm',
                'E. with shifted sources, AE(H), cm', 'pics\\bp_l_def_vs_e0.png')
ppp.plot_biplot(np.abs(fc_def_err[0]), np.abs(fc_0_err[0]), 'Default ensemble, AE(T), h',
                'E. with shifted sources, AE(T), h', 'pics\\bp_t_def_vs_e0.png')
