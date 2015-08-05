import numpy as np

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

all_src = [s1, s2, s3, m_fc]

p_levels = np.arange(50, 161, 10)
for pl in p_levels:
    p = np.genfromtxt('data\\PEAK_PARAMS_S1_'+str(pl).zfill(3)+'.csv', delimiter=',')
    new_params = []
    for i in range(0, len(p)):
        p_param = p[i]
        if not np.isnan(p_param[0]):
            for s_idx in range(len(all_src)):
                pos = p_param[s_idx*4 + 1]
                while (pos >= 0) and (all_src[s_idx][p_param[0], pos] > pl):
                    pos -= 1
                if pos < 0:
                    pos = np.nan
                p_param[s_idx*4 + 3] = p_param[s_idx*4 + 1] - pos
                pos = p_param[s_idx*4 + 1]
                while (pos <= T) and (all_src[s_idx][p_param[0], pos] > pl):
                    pos += 1
                if pos > T:
                    pos = np.nan
                p_param[s_idx*4 + 4] = pos - p_param[s_idx*4 + 1]
        new_params += [p_param, ]
    np.savetxt('data\\PEAK_PARAMS_S1_LCUT_'+str(pl).zfill(3)+'.csv', new_params, fmt='%0g', delimiter=',')



