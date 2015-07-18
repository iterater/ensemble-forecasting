import numpy as np
import math as mt

def run_cont_forecast(fcs, m_fc, w_len):

    c = lse_coeff(fcs, fc_index-window, fc_index, w)
    t_fc = np.full((1, T+1), c[len(fcs)])
    for k in range(len(fcs)):
        t_fc += fcs[k][t, 0:T+1] * c[k]
    t_err = t_fc - m_fc[t]
    res_err = 0.0
    for fct in range(1, T+1):
        res_err += abs(t_err[0, fct])/T
    return res_err


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
