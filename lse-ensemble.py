import numpy as np

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
        a[len(fcs), i] = np.sum(w*fcs[i][start_index:stop_index+1, 1:T+1])
        a[i, len(fcs)] = a[len(fcs), i]
        for j in range(len(fcs)):
            a[i, j] = np.sum(w*fcs[i][start_index:stop_index+1, 1:T+1]*fcs[j][start_index:stop_index+1, 1:T+1])
    b[len(fcs)] = np.sum(w*m_fc[start_index:stop_index+1, 1:T+1])
    a[len(fcs), len(fcs)] = np.sum(w)
    return np.linalg.solve(a, b)
