import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math as m

def average_by_time(p_time, data):
    t_range = np.arange(0, 61)
    res = []
    w = 8
    for tfc in range(len(t_range)):
        mask = (p_time > t_range[tfc] - w) & (p_time < t_range[tfc] + w)
        selected = data[mask]
        if len(selected) > 0:
            res += [[tfc, np.average(selected)]]
    return np.array(res)


def plot_errors(pLevel):
    plt.figure(1, figsize=(9, 12))
    p = np.genfromtxt('d:\\PEAK_PARAMS_S1_'+str(pLevel).zfill(3)+'.csv', delimiter=',')
    N_SRC = 3
    tRange = np.arange(0, 61)
    plt.subplots_adjust(wspace=0, hspace=0)
    for src_i in range(N_SRC):
        # T orig
        args_t = p[:, -4].flatten()
        # H
        ax = plt.subplot(4, 3, src_i + 1)
        src_vals = p[:, src_i*4 + 2].flatten()
        m_vals = p[:, -3].flatten()
        mask = ~np.isnan(src_vals) & ~np.isnan(m_vals)
        plt.plot(args_t[mask], src_vals[mask] - m_vals[mask], 'ro')
        avg_e = average_by_time(args_t[mask], src_vals[mask] - m_vals[mask])
        plt.plot(avg_e[:, 0], avg_e[:, 1], color='k', linewidth=2)
        plt.xlim([0, 60])
        plt.ylim([-100, 100])
        plt.setp(ax.get_xticklabels(), visible=False)
        if src_i != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            plt.ylabel('H error, cm')
        # T
        ax = plt.subplot(4, 3, src_i + 1 + N_SRC)
        src_vals = p[:, src_i*4 + 1].flatten()
        m_vals = p[:, -4].flatten()
        mask = ~np.isnan(src_vals) & ~np.isnan(m_vals)
        plt.plot(args_t[mask], src_vals[mask] - m_vals[mask], 'ro')
        avg_e = average_by_time(args_t[mask], src_vals[mask] - m_vals[mask])
        plt.plot(avg_e[:, 0], avg_e[:, 1], color='k', linewidth=2)
        plt.xlim([0, 60])
        plt.ylim([-10, 10])
        plt.setp(ax.get_xticklabels(), visible=False)
        if src_i != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            plt.ylabel('T error, hours')
        # W
        ax = plt.subplot(4, 3, src_i + 1 + N_SRC * 2)
        src_vals = (p[:, src_i*4 + 4] + p[:, src_i*4 + 3]).flatten()
        m_vals = (p[:, -1] + p[:, -2]).flatten()
        mask = ~np.isnan(src_vals) & ~np.isnan(m_vals)
        plt.plot(args_t[mask], src_vals[mask] - m_vals[mask], 'ro')
        avg_e = average_by_time(args_t[mask], src_vals[mask] - m_vals[mask])
        plt.plot(avg_e[:, 0], avg_e[:, 1], color='k', linewidth=2)
        plt.xlim([0, 60])
        plt.ylim([-50, 50])
        plt.setp(ax.get_xticklabels(), visible=False)
        if src_i != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            plt.ylabel('W error, hours')
        # D
        ax = plt.subplot(4, 3, src_i + 1 + N_SRC * 3)
        src_vals = (p[:, src_i*4 + 4] / (p[:, src_i*4 + 4] + p[:, src_i*4 + 3])).flatten()
        m_vals = (p[:, -1] / (p[:, -1] + p[:, -2])).flatten()
        mask = ~np.isnan(src_vals) & ~np.isnan(m_vals)
        plt.plot(args_t[mask], src_vals[mask] - m_vals[mask], 'ro')
        avg_e = average_by_time(args_t[mask], src_vals[mask] - m_vals[mask])
        plt.plot(avg_e[:, 0], avg_e[:, 1], color='k', linewidth=2)
        plt.xlim([0, 60])
        plt.ylim([-0.8, 0.8])
        if src_i != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            plt.ylabel('D error')
        plt.xlabel('Forecast time, hours')
    # plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig('d:\\PEAK_ERRS_S1_'+str(pLevel).zfill(3)+'.png')
    plt.close()


def plot_stats(pLevel):
    plt.figure(1, figsize=(14, 24))
    p = np.genfromtxt('d:\\PEAK_PARAMS_S1_'+str(pLevel).zfill(3)+'.csv', delimiter=',')
    N_SRC = 3
    for src_i in range(N_SRC):
        # H label
        plt.subplot2grid((12, 7), (0, 0))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.axis('off')
        plt.text(0.5, 0.5, 'H', fontsize=40, horizontalalignment='center', verticalalignment='center')
        # H
        lim = [50, 250]
        ax = plt.subplot2grid((12, 7), (1, src_i * 2 + 1), colspan=2, rowspan=2)
        src_vals = p[:, src_i*4 + 2].flatten()
        m_vals = p[:, -3].flatten()
        mask = ~np.isnan(src_vals) & ~np.isnan(m_vals)
        src_vals = src_vals[mask]
        m_vals = m_vals[mask]
        plt.plot(lim, lim, color='k')
        plt.plot(src_vals, m_vals, 'ro')
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.xlim(lim)
        plt.ylim(lim)
        # H pdf
        ax = plt.subplot2grid((12, 7), (0, src_i * 2 + 1), colspan=2)
        gkde = gaussian_kde(src_vals)
        x = np.linspace(lim[0], lim[1], 100)
        pdf = gkde(x)
        plt.xlim(lim)
        plt.ylim([0, 0.025])
        ax.xaxis.tick_top()
        plt.plot(x, pdf, linewidth=2)
        plt.hist(src_vals, histtype='step', normed=True)
        if src_i != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        # H-M pdf
        if src_i == 0:
            ax = plt.subplot2grid((12, 7), (1, 0), rowspan=2)
            gkde = gaussian_kde(m_vals)
            x = np.linspace(lim[0], lim[1], 100)
            pdf = gkde(x)
            plt.ylim(lim)
            plt.xlim([0, 0.025])
            plt.xticks(rotation='vertical')
            plt.plot(pdf, x, linewidth=2)
            plt.hist(m_vals, histtype='step', normed=True, orientation="horizontal")
        # T label
        plt.subplot2grid((12, 7), (3, 0))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.axis('off')
        plt.text(0.5, 0.5, 'T', fontsize=40, horizontalalignment='center', verticalalignment='center')
        # T
        lim = [0, 60]
        ax = plt.subplot2grid((12, 7), (4, src_i * 2 + 1), colspan=2, rowspan=2)
        src_vals = p[:, src_i*4 + 1].flatten()
        m_vals = p[:, -4].flatten()
        mask = ~np.isnan(src_vals) & ~np.isnan(m_vals)
        src_vals = src_vals[mask]
        m_vals = m_vals[mask]
        plt.plot(lim, lim, color='k')
        plt.plot(src_vals, m_vals, 'ro')
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.xlim(lim)
        plt.ylim(lim)
        # T pdf
        ax = plt.subplot2grid((12, 7), (3, src_i * 2 + 1), colspan=2)
        gkde = gaussian_kde(src_vals)
        x = np.linspace(lim[0], lim[1], 100)
        pdf = gkde(x)
        plt.xlim(lim)
        plt.ylim([0, 0.035])
        ax.xaxis.tick_top()
        plt.plot(x, pdf, linewidth=2)
        plt.hist(src_vals, histtype='step', normed=True)
        if src_i != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        # T-M pdf
        if src_i == 0:
            ax = plt.subplot2grid((12, 7), (4, 0), rowspan=2)
            gkde = gaussian_kde(m_vals)
            x = np.linspace(lim[0], lim[1], 100)
            pdf = gkde(x)
            plt.ylim(lim)
            plt.xlim([0, 0.035])
            plt.xticks(rotation='vertical')
            plt.plot(pdf, x, linewidth=2)
            plt.hist(m_vals, histtype='step', normed=True, orientation="horizontal")
        # W label
        plt.subplot2grid((12, 7), (6, 0))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.axis('off')
        plt.text(0.5, 0.5, 'W', fontsize=40, horizontalalignment='center', verticalalignment='center')
        # W
        lim = [0, 50]
        ax = plt.subplot2grid((12, 7), (7, src_i * 2 + 1), colspan=2, rowspan=2)
        src_vals = (p[:, src_i*4 + 4] + p[:, src_i*4 + 3]).flatten()
        m_vals = (p[:, -1] + p[:, -2]).flatten()
        mask = ~np.isnan(src_vals) & ~np.isnan(m_vals)
        src_vals = src_vals[mask]
        m_vals = m_vals[mask]
        plt.plot(lim, lim, color='k')
        plt.plot(src_vals, m_vals, 'ro')
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.xlim(lim)
        plt.ylim(lim)
        # W pdf
        ax = plt.subplot2grid((12, 7), (6, src_i * 2 + 1), colspan=2)
        gkde = gaussian_kde(src_vals)
        x = np.linspace(lim[0], lim[1], 100)
        pdf = gkde(x)
        plt.xlim(lim)
        plt.ylim([0, 0.06])
        ax.xaxis.tick_top()
        plt.plot(x, pdf, linewidth=2)
        plt.hist(src_vals, histtype='step', normed=True)
        if src_i != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        # W-M pdf
        if src_i == 0:
            ax = plt.subplot2grid((12, 7), (7, 0), rowspan=2)
            gkde = gaussian_kde(m_vals)
            x = np.linspace(lim[0], lim[1], 100)
            pdf = gkde(x)
            plt.ylim(lim)
            plt.xlim([0, 0.06])
            plt.xticks(rotation='vertical')
            plt.plot(pdf, x, linewidth=2)
            plt.hist(m_vals, histtype='step', normed=True, orientation="horizontal")
        # D label
        plt.subplot2grid((12, 7), (9, 0))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.axis('off')
        plt.text(0.5, 0.5, 'D', fontsize=40, horizontalalignment='center', verticalalignment='center')
        # D
        lim = [0, 1]
        ax = plt.subplot2grid((12, 7), (10, src_i * 2 + 1), colspan=2, rowspan=2)
        src_vals = (p[:, src_i*4 + 4] / (p[:, src_i*4 + 4] + p[:, src_i*4 + 3])).flatten()
        m_vals = (p[:, -1] / (p[:, -1] + p[:, -2])).flatten()
        mask = ~np.isnan(src_vals) & ~np.isnan(m_vals)
        src_vals = src_vals[mask]
        m_vals = m_vals[mask]
        plt.plot(lim, lim, color='k')
        plt.plot(src_vals, m_vals, 'ro')
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.xlim(lim)
        plt.ylim(lim)
        # D pdf
        ax = plt.subplot2grid((12, 7), (9, src_i * 2 + 1), colspan=2)
        gkde = gaussian_kde(src_vals)
        x = np.linspace(lim[0], lim[1], 100)
        pdf = gkde(x)
        plt.xlim(lim)
        plt.ylim([0, 3])
        ax.xaxis.tick_top()
        plt.plot(x, pdf, linewidth=2)
        plt.hist(src_vals, histtype='step', normed=True)
        if src_i != 0:
            plt.setp(ax.get_yticklabels(), visible=False)
        # D-M pdf
        if src_i == 0:
            ax = plt.subplot2grid((12, 7), (10, 0), rowspan=2)
            gkde = gaussian_kde(m_vals)
            x = np.linspace(lim[0], lim[1], 100)
            pdf = gkde(x)
            plt.ylim(lim)
            plt.xticks(rotation='vertical')
            plt.xlim([0, 3])
            plt.plot(pdf, x, linewidth=2)
            plt.hist(m_vals, histtype='step', normed=True, orientation="horizontal")
    plt.tight_layout()
    plt.savefig('d:\\PEAK_STATS_S1_'+str(pLevel).zfill(3)+'.png')
    plt.close()
    # plt.show()


def regr_coeff(x, y):
    N = x.shape[1]
    mask = ~np.isnan(y)
    for i in range(N):
        mask &= ~np.isnan(x[:, i])
    x_flt = x[mask]
    y_flt = y[mask]
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
    return np.linalg.solve(a, b)


def regr_coeff_all(pLevel):
    p = np.genfromtxt('d:\\PEAK_PARAMS_S1_'+str(pLevel).zfill(3)+'.csv', delimiter=',')
    p1 = np.zeros(p.shape)
    for i in range(4):
        p1[:, 4*i + 1] = p[:, 4*i + 2]
        p1[:, 4*i + 2] = p[:, 4*i + 1]
        p1[:, 4*i + 3] = p[:, 4*i + 3] + p[:, 4*i + 4]
        p1[:, 4*i + 4] = p[:, 4*i + 3] / (p[:, 4*i + 3] + p[:, 4*i + 4])
    src_names = ['BSM-WOWC-HIRLAM', 'BALTP-90M-GFS', 'HIROMB', 'M']
    param_names = ['H', 'T', 'W', 'D']
    res_coeff = []
    p_cnt = len(param_names)
    s_cnt = len(src_names) - 1
    for p_idx in range(p_cnt):
        src_data = p1[:, (1 + p_idx):(1 + p_cnt * s_cnt):p_cnt]
        m_data = p1[:, 1 + p_idx + p_cnt * s_cnt]
        res_coeff += [regr_coeff(src_data, m_data)]
    return np.array(res_coeff)


def revert_index(idx, N):
    i1 = int(m.floor(idx / N))
    i2 = idx % N
    return i2 * N + i1


def calc_correlations():
    p = np.genfromtxt('d:\\PEAK_PARAMS_S1_060.csv', delimiter=',')
    p1 = np.zeros(p.shape)
    for i in range(4):
        p1[:, 4*i + 1] = p[:, 4*i + 2]
        p1[:, 4*i + 2] = p[:, 4*i + 1]
        p1[:, 4*i + 3] = p[:, 4*i + 3] + p[:, 4*i + 4]
        p1[:, 4*i + 4] = p[:, 4*i + 3] / (p[:, 4*i + 3] + p[:, 4*i + 4])
    src_names = ['BSM-WOWC-HIRLAM', 'BALTP-90M-GFS', 'HIROMB','M']
    param_names = ['H', 'T', 'W', 'D']
    N = len(src_names) * len(param_names)
    c = np.zeros((N, N))
    params_list = ''
    for i in range(N):
        ri = revert_index(i, 4)
        params_list += src_names[int(m.floor(ri / 4))] + '\t' + param_names[ri % 4] + '\n'
        for j in range(N):
            data1 = p1[:, revert_index(i, 4) + 1].flatten()
            data2 = p1[:, revert_index(j, 4) + 1].flatten()
            msk = ~np.isnan(data1) & ~np.isnan(data2)
            c[i, j] = np.corrcoef(data1[msk], data2[msk])[0, 1]
    np.savetxt('d:\\correlation.txt', c)
    f = open('d:\\correlation_names.txt', 'w')
    f.write(params_list)
    f.close()

prange = np.arange(60, 161, 20)
for i in range(len(prange)):
    cf = regr_coeff_all(prange[i])
    np.savetxt('d:\\regr_coeff_'+str(prange[i]).zfill(3)+'.txt', cf)
    print(cf)

# prange = np.arange(60, 161, 20)
# for i in range(len(prange)):
    # plot_stats(prange[i])
    # plot_errors(prange[i])

#     p = np.genfromtxt('d:\\PEAK_PARAMS_S1_'+str(prange[i]).zfill(3)+'.csv', delimiter=',')
#     vals = p[:, 2].flatten()
#     vals = vals[~np.isnan(vals)]
#     #plt.hist(vals, histtype='step', normed=True)
#     gkde = gaussian_kde(vals)
#     pdf = gkde(x)
#     plt.plot(pdf, x)
#     #d = d + [vals]
# plt.legend(prange)
# #plt.boxplot(d, whis=100)
# #plt.ylim([0, 20])
# plt.show()