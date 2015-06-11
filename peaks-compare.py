from scipy.stats import gaussian_kde
import os
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import math as mt
from sklearn.neighbors.kde import KernelDensity

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


def forecast_peak_error(fc, msk_l):
    msk_b = np.ones(len(fc), dtype=bool)
    msk_b[msk_l] = 0
    max_l_err = np.amax(fc, axis=1) - np.amax(m_fc, axis=1)
    max_t_err = (np.argmax(fc, axis=1) - np.argmax(m_fc, axis=1)).astype(float)
    msk_add = (max_l_err > 20) | (max_l_err < -20) | (max_t_err > 10) | (max_t_err < -10)
    max_l_err[msk_b | msk_add] = np.nan
    max_t_err[msk_b | msk_add] = np.nan
    return np.vstack((max_t_err, max_l_err))


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


pLevel = 60
level_scale_mode = 1 # 0 - mult, 1 - add-peak, 2 - add-all
level_scale_flag = True
level_scale_flag_string = ''
if level_scale_flag:
    level_scale_flag_string = 'l' + str(level_scale_mode)
# w = create_w_mask(N, 0)
w = create_w_mask_from_level(N, 0.05)
src_set = (s1, s2, s3)
c = lse_coeff(src_set, 0, N - 1, w)
print(c)
fc_def = np.full((N, T + 1), c[3])
e_def = []
for src_i in range(3):
    fc_def += src_set[src_i] * c[src_i]
indexRange = np.arange(N)
plt.figure(1, figsize=(12, 5))
plt.plot(indexRange, forecast_error(fc_def), indexRange, forecast_error(s1), indexRange, forecast_error(s2), indexRange, forecast_error(s3))
plt.legend(['Ensemble', 'Hiromb', 'Swan', 'No-Swan'])
plt.title('Simple ensemble MAE')
plt.xlabel('Forecast index')
plt.ylabel('Forecast MAE, cm')
plt.xlim([0, N])
plt.savefig('pics\\simple-ensemble-mae-' + level_scale_flag_string + 't-scale-pl' + str(pLevel).zfill(3) + '.png')
plt.close()

plt.figure(1, figsize=(12, 5))
plt.plot(indexRange, forecast_wmae(fc_def, pLevel), indexRange, forecast_wmae(s1, pLevel), indexRange, forecast_wmae(s2, pLevel), indexRange, forecast_wmae(s3, pLevel))
plt.legend(['Ensemble', 'Hiromb', 'Swan', 'No-Swan'])
plt.title('Ensemble WMAE')
plt.xlabel('Forecast index')
plt.ylabel('WMAE (' + str(pLevel) + ' cm), cm')
plt.xlim([0, N])
plt.savefig('pics\\simple-ensemble-wmae-' + level_scale_flag_string + 't-scale-pl' + str(pLevel).zfill(3) + '.png')
plt.close()


def transform_forecast(fc, original_peak, target_peak, scale_vertical):
    # Full scale
    # t_src_nodes = [-1, np.round(original_peak[1] - original_peak[2] * original_peak[3]), np.round(original_peak[1]), np.round(original_peak[1] + original_peak[2] * (1.0 - original_peak[3])), T + 1]
    # t_dst_nodes = [-1, int(target_peak[1] - target_peak[2] * target_peak[3]), int(target_peak[1]), int(target_peak[1] + target_peak[2] * (1.0 - target_peak[3])), T + 1]
    # Peak scale
    p_start = min(np.round(original_peak[1] - original_peak[2] * original_peak[3]), np.round(target_peak[1] - target_peak[2] * target_peak[3]))
    p_start = max(p_start, 0)
    p_end = max(np.round(original_peak[1] + original_peak[2] * (1.0 - original_peak[3])), np.round(target_peak[1] + target_peak[2] * (1.0 - target_peak[3])))
    p_end = min(p_end, T)
    t_src_nodes = [-1, p_start, np.round(original_peak[1]), p_end, T + 1]
    t_dst_nodes = [-1, p_start, np.round(target_peak[1]), p_end, T + 1]
    # multiplication scale
    if level_scale_mode == 0:
        scale = np.full(fc.shape, 1)
        mult = target_peak[0] / original_peak[0]
        if scale_vertical:
            for t_fc in range(T + 1):
                if t_fc == t_src_nodes[2]:
                    scale[t_fc] = mult
                if (t_fc > t_src_nodes[1]) and (t_fc < t_src_nodes[2]):
                    scale[t_fc] = mult * (t_fc - t_src_nodes[1]) / (t_src_nodes[2] - t_src_nodes[1])
                if (t_fc > t_src_nodes[2]) and (t_fc < t_src_nodes[3]):
                    scale[t_fc] = mult * (t_src_nodes[3] - t_fc) / (t_src_nodes[3] - t_src_nodes[2])
        l_res = fc * scale
    # additive scale
    if level_scale_mode == 1:
        scale = np.full(fc.shape, 0)
        add = target_peak[0] - original_peak[0]
        if scale_vertical:
            for t_fc in range(T + 1):
                if t_fc == t_src_nodes[2]:
                    scale[t_fc] = add
                if (t_fc > t_src_nodes[1]) and (t_fc < t_src_nodes[2]):
                    scale[t_fc] = add * (t_fc - t_src_nodes[1]) / (t_src_nodes[2] - t_src_nodes[1])
                if (t_fc > t_src_nodes[2]) and (t_fc < t_src_nodes[3]):
                    scale[t_fc] = add * (t_src_nodes[3] - t_fc) / (t_src_nodes[3] - t_src_nodes[2])
        l_res = fc + scale
    # additive all scale
    if level_scale_mode == 2:
        scale = target_peak[0] - original_peak[0]
        l_res = fc + scale



    l_res = np.concatenate(([l_res[0]], l_res, [l_res[-1]]))
    t_res = np.array([])
    for node_i in range(4):
        idxs = np.arange(t_src_nodes[node_i], t_src_nodes[node_i + 1] + 1) - t_src_nodes[node_i]
        if t_src_nodes[node_i + 1] != t_src_nodes[node_i]:
            idxs *= (t_dst_nodes[node_i + 1] - t_dst_nodes[node_i]) / (t_src_nodes[node_i + 1] - t_src_nodes[node_i])
        idxs += t_dst_nodes[node_i]
        if node_i != 0:
            idxs = idxs[1:]
        t_res = np.concatenate((t_res, idxs))
    if len(t_res) != len(l_res):
        print('ACHTUNG!!!')
        return [0]
    fi = interp.interp1d(t_res, l_res)
    res = fi(np.arange(0, T+1))
    return res


p = np.genfromtxt('data\\PEAK_PARAMS_S1_'+str(pLevel).zfill(3)+'.csv', delimiter=',')
pc = np.loadtxt('data\\regr_coeff_'+str(pLevel).zfill(3)+'.txt')
mask = ~np.isnan(p[:, 0])
for i in range(13):
    mask &= ~np.isnan(p[:, i])
p_flt = p[mask]
t_idx = np.arange(0, 61)
colors = ('b', 'g', 'r')
peak_l = [pLevel, pLevel, pLevel]
dir_name = 'pics\\all-forecasts-' + level_scale_flag_string + 't-scale-pl' + str(pLevel).zfill(3)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
for i in range(len(p_flt)):
    print 'FC #' + str(i)
    plt.figure(i, figsize=(12, 9))
    params = []
    res_params = pc[:, 3].flatten()
    for src_i in range(3):
        p1 = p_flt[i, (src_i * 4 + 1):(src_i * 4 + 5)]
        p2 = [p1[1], p1[0], p1[2] + p1[3], p1[2] / (p1[2] + p1[3])]
        params += [p2]
        res_params = res_params + np.array(p2) * pc[:, src_i].flatten()
        plt.plot(t_idx, src_set[src_i][p_flt[i, 0]], colors[src_i] + '-')
    plt.plot(t_idx, m_fc[p_flt[i, 0]], 'o', color='k')
    peak_t = [res_params[1] - res_params[2] * res_params[3], res_params[1], res_params[1] + res_params[2] * (1.0 - res_params[3])]
    peak_l[1] = res_params[0]
    plt.plot(peak_t, peak_l, '^', markersize=10)
    plt.plot(t_idx, fc_def[p_flt[i, 0]], 'k-')
    e_fc = np.full(T + 1, c[3])
    for src_i in range(3):
        src_set[src_i][p_flt[i, 0]] = transform_forecast(src_set[src_i][p_flt[i, 0]], params[src_i], res_params, level_scale_flag)
        plt.plot(t_idx, src_set[src_i][p_flt[i, 0]], colors[src_i] + '--')
        e_fc += src_set[src_i][p_flt[i, 0]] * c[src_i]
    plt.plot(t_idx, e_fc, 'k--')
    plt.xlim([0, T])
    plt.savefig(dir_name + '\\forecast-' + str(i).zfill(3) + '.png')
    plt.close()


c = lse_coeff(src_set, 0, N - 1, w)
fc_1 = np.full((N, T + 1), c[3])
for src_i in range(3):
    fc_1 += src_set[src_i] * c[src_i]
indexRange = np.arange(N)

plt.figure(1, figsize=(12, 5))
plt.plot(indexRange, forecast_error(fc_def), indexRange, forecast_error(fc_1))
plt.legend(['Ensemble def', 'Ensemble #1'])
plt.title('Simple ensemble MAE')
plt.xlabel('Forecast index')
plt.ylabel('Forecast MAE, cm')
plt.xlim([0, N - 1])
plt.savefig('pics\\ensemble-1-mae-' + level_scale_flag_string + 't-scale-pl' + str(pLevel).zfill(3) + '.png')
plt.close()

plt.figure(1, figsize=(12, 5))
plt.plot(indexRange, forecast_wmae(fc_def, pLevel), indexRange, forecast_wmae(fc_1, pLevel))
plt.legend(['Ensemble def', 'Ensemble #1'])
plt.title('Ensemble WMAE')
plt.xlabel('Forecast index')
plt.ylabel('WMAE (' + str(pLevel) + ' cm), cm')
plt.xlim([0, N - 1])
plt.savefig('pics\\ensemble-1-wmae-' + level_scale_flag_string + 't-scale-pl' + str(pLevel).zfill(3) + '.png')
plt.close()

plt.figure(1, figsize=(12, 5))
fc_def_err = forecast_peak_error(fc_def, np.array(p_flt[:, 0]).flatten().astype(int))
fc_1_err = forecast_peak_error(fc_1, np.array(p_flt[:, 0]).flatten().astype(int))
plt.plot(indexRange, fc_def_err[0], 'o-')
plt.plot(indexRange, fc_1_err[0], 'o-')
plt.legend(['Ensemble def', 'Ensemble #1'])
plt.xlabel('Forecast index')
plt.ylabel('Peak time error, h')
plt.xlim([0, N - 1])
plt.savefig('pics\\ensemble-1-pet-' + level_scale_flag_string + 't-scale-pl' + str(pLevel).zfill(3) + '.png')
plt.close()

plt.figure(1, figsize=(12, 5))
plt.plot(indexRange, fc_def_err[1], 'o-')
plt.plot(indexRange, fc_1_err[1], 'o-')
plt.legend(['Ensemble def', 'Ensemble #1'])
plt.xlabel('Forecast index')
plt.ylabel('Peak level error, cm')
plt.xlim([0, N - 1])
plt.savefig('pics\\ensemble-1-pel-' + level_scale_flag_string + 't-scale-pl' + str(pLevel).zfill(3) + '.png')
plt.close()

plt.figure(1, figsize=(8, 6))
nan_filter = ~np.isnan(fc_def_err[1]) & ~np.isnan(fc_1_err[1])
lim = [-30, 30]
# gkde = gaussian_kde(fc_def_err[1][nan_filter])
x = np.linspace(lim[0], lim[1], 100)
# pdf = gkde(x)
# plt.plot(x, pdf, linewidth=2, color='b')
plt.hist(fc_def_err[1][nan_filter], histtype='step', normed=True, color='b')
kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(np.array(fc_def_err[1][nan_filter]).reshape(-1, 1))
plt.plot(x, np.exp(kde.score_samples(x.reshape(-1, 1))), linewidth=2, color='b')
# gkde = gaussian_kde(fc_1_err[1][nan_filter])
# pdf = gkde(x)
# plt.plot(x, pdf, linewidth=2, color='r')
plt.hist(fc_1_err[1][nan_filter], histtype='step', normed=True, color='r')
kde = KernelDensity(kernel='gaussian', bandwidth=2).fit(np.array(fc_1_err[1][nan_filter]).reshape(-1, 1))
plt.plot(x, np.exp(kde.score_samples(x.reshape(-1, 1))), linewidth=2, color='r')
plt.legend(['Ensemble def', 'Ensemble #1'])
plt.xlabel('Peak level error, cm')
plt.ylabel('PDF')
plt.savefig('pics\\ensemble-1-pel-pdf-' + level_scale_flag_string + 't-scale-pl' + str(pLevel).zfill(3) + '.png')
plt.close()

plt.figure(1, figsize=(8, 6))
nan_filter = ~np.isnan(fc_def_err[0]) & ~np.isnan(fc_1_err[0])
print('Error def (T, L): ')
print(np.mean(fc_def_err[:, nan_filter], axis=1))
print('Error 1 (T, L): ')
print(np.mean(fc_1_err[:, nan_filter], axis=1))
lim = [-10, 10]
# gkde = gaussian_kde(fc_def_err[0][nan_filter])
x = np.linspace(lim[0], lim[1], 100)
# pdf = gkde(x)
# plt.plot(x, pdf, linewidth=2, color='b')
plt.hist(fc_def_err[0][nan_filter], histtype='step', normed=True, color='b')
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(fc_def_err[0][nan_filter]).reshape(-1, 1))
plt.plot(x, np.exp(kde.score_samples(x.reshape(-1, 1))), linewidth=2, color='b')
# gkde = gaussian_kde(fc_1_err[0][nan_filter])
# pdf = gkde(x)
# plt.plot(x, pdf, linewidth=2, color='r')
plt.hist(fc_1_err[0][nan_filter], histtype='step', normed=True, color='r')
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.array(fc_1_err[0][nan_filter]).reshape(-1, 1))
plt.plot(x, np.exp(kde.score_samples(x.reshape(-1, 1))), linewidth=2, color='r')
plt.legend(['Ensemble def', 'Ensemble #1'])
plt.xlabel('Peak time error, h')
plt.ylabel('PDF')
plt.savefig('pics\\ensemble-1-pet-pdf-' + level_scale_flag_string + 't-scale-pl' + str(pLevel).zfill(3) + '.png')
plt.close()