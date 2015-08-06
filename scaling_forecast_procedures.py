import numpy as np
from enum import Enum
import scipy.interpolate as interp

# scale type
class ScaleType(Enum):
    multiplication_peak_scale = 1
    add_peak_scale = 2
    add_all_scale = 3
    no_scale = 4
    multiplication_all_scale = 5


# scale forecast vertically to peak
def scale_peak_vertically(fc, target_peak, t_lim, scale_type, w_fun):
    p_start = np.round(target_peak[1] - target_peak[2] * target_peak[3])
    p_start = max(p_start, 0)
    p_end = np.round(target_peak[1] + target_peak[2] * (1.0 - target_peak[3]))
    p_end = min(p_end, t_lim)
    p_pos = np.round(target_peak[1])
    if scale_type == ScaleType.add_peak_scale:
        scale = np.full(fc.shape, 0)
        add = target_peak[0] - fc[p_pos]
        for t_fc in range(t_lim + 1):
            if t_fc == p_pos:
                scale[t_fc] = add
            if (t_fc > p_start) and (t_fc < p_pos):
                scale[t_fc] = add * w_fun((t_fc - p_start) / (p_pos - p_start))
            if (t_fc > p_pos) and (t_fc < p_end):
                scale[t_fc] = add * w_fun((p_end - t_fc) / (p_end - p_pos))
        l_res = fc + scale
    elif scale_type == ScaleType.add_all_scale:
        add = target_peak[0] - fc[p_pos]
        l_res = fc + add
    elif scale_type == ScaleType.multiplication_peak_scale:
        scale = np.full(fc.shape, 1)
        mul = target_peak[0] / fc[p_pos]
        for t_fc in range(t_lim + 1):
            if t_fc == p_pos:
                scale[t_fc] = mul
            if (t_fc > p_start) and (t_fc < p_pos):
                scale[t_fc] = 1 + (mul - 1) * w_fun((t_fc - p_start) / (p_pos - p_start))
            if (t_fc > p_pos) and (t_fc < p_end):
                scale[t_fc] = 1 + (mul - 1) * w_fun((p_end - t_fc) / (p_end - p_pos))
        l_res = fc * scale
    elif scale_type == ScaleType.multiplication_all_scale:
        mul = target_peak[0] / fc[p_pos]
        l_res = fc * mul
    else:
        l_res = fc
    return l_res

def transform_forecast(fc, original_peak, target_peak, t_lim, scale_type, w_fun):
    # Full scale
    # t_src_nodes = [-1, np.round(original_peak[1] - original_peak[2] * original_peak[3]), np.round(original_peak[1]), np.round(original_peak[1] + original_peak[2] * (1.0 - original_peak[3])), T + 1]
    # t_dst_nodes = [-1, int(target_peak[1] - target_peak[2] * target_peak[3]), int(target_peak[1]), int(target_peak[1] + target_peak[2] * (1.0 - target_peak[3])), T + 1]
    # Peak scale
    p_start = min(np.round(original_peak[1] - original_peak[2] * original_peak[3]), np.round(target_peak[1] - target_peak[2] * target_peak[3]))
    p_start = max(p_start, 0)
    p_end = max(np.round(original_peak[1] + original_peak[2] * (1.0 - original_peak[3])), np.round(target_peak[1] + target_peak[2] * (1.0 - target_peak[3])))
    p_end = min(p_end, t_lim)
    t_src_nodes = [-1, p_start, np.round(original_peak[1]), p_end, t_lim + 1]
    t_dst_nodes = [-1, p_start, np.round(target_peak[1]), p_end, t_lim + 1]
    # multiplication scale
    if scale_type == ScaleType.multiplication_peak_scale:
        scale = np.full(fc.shape, 1)
        mult = target_peak[0] / original_peak[0]
        for t_fc in range(t_lim + 1):
            if t_fc == t_src_nodes[2]:
                scale[t_fc] = mult
            if (t_fc > t_src_nodes[1]) and (t_fc < t_src_nodes[2]):
                scale[t_fc] = 1 + (mult - 1) * w_fun((t_fc - t_src_nodes[1]) / (t_src_nodes[2] - t_src_nodes[1]))
            if (t_fc > t_src_nodes[2]) and (t_fc < t_src_nodes[3]):
                scale[t_fc] = 1 + (mult - 1) * w_fun((t_src_nodes[3] - t_fc) / (t_src_nodes[3] - t_src_nodes[2]))
        l_res = fc * scale
    # additive scale
    elif scale_type == ScaleType.add_peak_scale:
        scale = np.full(fc.shape, 0)
        add = target_peak[0] - original_peak[0]
        for t_fc in range(t_lim + 1):
            if t_fc == t_src_nodes[2]:
                scale[t_fc] = add
            if (t_fc > t_src_nodes[1]) and (t_fc < t_src_nodes[2]):
                scale[t_fc] = add * w_fun((t_fc - t_src_nodes[1]) / (t_src_nodes[2] - t_src_nodes[1]))
            if (t_fc > t_src_nodes[2]) and (t_fc < t_src_nodes[3]):
                scale[t_fc] = add * w_fun((t_src_nodes[3] - t_fc) / (t_src_nodes[3] - t_src_nodes[2]))
        l_res = fc + scale
    # additive all scale
    elif scale_type == ScaleType.add_all_scale:
        scale = target_peak[0] - original_peak[0]
        l_res = fc + scale
    # no scale
    else:
        l_res = fc
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
    res = fi(np.arange(0, t_lim+1))
    return res
