import numpy as np


def forecast_dist_mae(fc1, fc2):
    """
    MAE distance
    :param fc1: Forecasts array 1 of shape (N,T)
    :param fc2: Forecasts array 2 of shape (N,T)
    :return: Distance array of length N
    """
    fc_diff = np.absolute(fc1 - fc2)
    return np.mean(fc_diff, axis=1)


def forecast_dist_dtw(fc1, fc2):
    """
    DTW distance
    :param fc1: Forecasts array 1 of shape (N,T)
    :param fc2: Forecasts array 2 of shape (N,T)
    :return: Distance array of length N
    """
    fc_shape = np.shape(fc1)
    res = np.zeros(fc_shape[0])
    di = [0, -1, -1]
    dj = [-1, 0, -1]
    m = np.zeros((fc_shape[1], fc_shape[1]))
    k = np.zeros((fc_shape[1], fc_shape[1]))
    for fc_i in range(fc_shape[0]):
        for i in range(fc_shape[1]):
            for j in range(fc_shape[1]):
                m[i, j] = np.absolute(fc1[fc_i, i] - fc2[fc_i, j])
                k[i, j] = 1
                if (i == 0) and (j > 0):
                    m[i, j] += m[i, j - 1]
                    k[i, j] = j + 1
                elif (i > 0) and (j == 0):
                    m[i, j] += m[i - 1, j]
                    k[i, j] = i + 1
                elif (i > 0) and (j > 0):
                    test = [m[i, j - 1], m[i - 1, j], m[i - 1, j - 1]]
                    min_idx = np.argmin(test)
                    m[i, j] += test[min_idx]
                    k[i, j] = 1 + k[i + di[min_idx], j + dj[min_idx]]
        res[fc_i] = m[fc_shape[1] - 1, fc_shape[1] - 1] / k[fc_shape[1] - 1, fc_shape[1] - 1]
    return res


def forecast_wmae(fc, m_fc, lev_lim):
    """
    WMAE for forecasts
    :param fc: Forecasts array 1 of shape (N,T)
    :param m_fc: Measurements forecasts array 1 of shape (N,T)
    :param lev_lim: Level for cut
    :return: WMAE array of shape (N)
    """
    fc_shape = np.shape(fc)
    res = np.zeros(fc_shape[0])
    for i in range(fc_shape[0]):
        mask = m_fc[i] >= lev_lim
        n = sum(mask)
        if n == 0:
            res[i] = np.nan
        else:
            res[i] = np.average(np.abs((m_fc[i] - fc[i])[mask]))
    return res


def forecast_wbias(fc, m_fc, lev_lim):
    """
    WBIAS for forecasts
    :param fc: Forecasts array 1 of shape (N,T)
    :param m_fc: Measurements forecasts array 1 of shape (N,T)
    :param lev_lim: Level for cut
    :return: WBIAS array of shape (N)
    """
    fc_shape = np.shape(fc)
    res = np.zeros(fc_shape[0])
    for i in range(fc_shape[0]):
        mask = m_fc[i] >= lev_lim
        n = sum(mask)
        if n == 0:
            res[i] = np.nan
        else:
            res[i] = np.average((m_fc[i] - fc[i])[mask])
    return res

def forecast_stdev_err(fc, m_fc):
    """
    Standard deviation for forecast error
    :param fc: Forecasts array 1 of shape (N,T)
    :param m_fc: Measurements forecasts array 1 of shape (N,T)
    :return: STDev array of shape (N)
    """
    fc_diff = np.absolute(fc - m_fc)
    return np.std(fc_diff, axis=1)

