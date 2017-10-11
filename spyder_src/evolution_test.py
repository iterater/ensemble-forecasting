import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import KFold
# import seaborn as sns
# sns.set_style("whitegrid")

base_dir = '/media/iterater/DATA/_SRC_/ensemble-forecasting/data/2011/'
# base_dir = 'd:\\Src\\ensemble-forecasting\\data\\2011\\'

h = np.loadtxt(base_dir + '2011080100_hiromb_GI_60x434.txt') - 37.356
s = np.loadtxt(base_dir + '2011080100_swan_GI_48x434.txt')
n = np.loadtxt(base_dir + '2011080100_noswan_GI_48x434.txt')
m = np.loadtxt(base_dir + '2011080100_measurements_GI_2623.txt')

N = 430
T = 48

# preparing measurements forecast
m_fc = np.zeros((N, T+1))
for i in range(N):
    for j in range(T+1):
        m_fc[i, j] = m[i*6+j]

# data cut and shifting
shift_const = 0
m_average = np.average(m_fc)
s = s[0:N, 1:T+1] - np.average(s) + shift_const
n = n[0:N, 1:T+1] - np.average(n) + shift_const
h = h[0:N, 1:T+1] - np.average(h) + shift_const
m_fc = m_fc[0:N, 1:T+1] - m_average

def ens(fcs, k):    
    res = np.zeros(fcs[0].shape)
    for i in range(len(k)):
        res += fcs[i]*k[i]
    return res

def err(fc):
    return np.average(np.std(m_fc - fc, axis=1))

N = 100
ff = np.zeros((N,N))
lims = (-1,2)
k_arg = np.linspace(lims[0],lims[1],N)
pp = PCA(n_components=2)

def work_on_area(t, do_plot=True):
    if do_plot:
        plt.figure(figsize=(4,4))
    lst = []
    for i in range(N):
        for j in range(N):
            ff[i,j] = np.std(ens([s[t], h[t]], [k_arg[i],k_arg[j]]) - m_fc[t])
            lst.append([k_arg[i],k_arg[j],ff[i,j]])        
    lst = np.array(lst)
    ff_min = np.min(ff)
    ff_sel = lst[lst[:,2].flatten() < ff_min*1.1][:,0:2].transpose()
    min_x = k_arg[np.argmin(ff) // ff.shape[1]]
    min_y = k_arg[np.argmin(ff) % ff.shape[1]]
    pp.fit(ff_sel.transpose())
    if do_plot:
        plt.contour(k_arg, k_arg, ff.transpose() - np.min(ff))
        plt.plot(ff_sel[0], ff_sel[1], 'o')
        plt.plot(min_x, min_y, 'or')    
        plt.quiver([min_x, min_x], [min_y, min_y],
                   pp.components_[:, 0].flatten() * pp.explained_variance_,
                   pp.components_[:, 1].flatten() * pp.explained_variance_,
                   scale=1.0)
        plt.xlim(lims)
        plt.ylim(lims)
    return np.concatenate(([min_x, min_y],pp.components_.flatten(),pp.explained_variance_))

diff = s[0:300]-h[0:300]

res = []
for t in range(300):
    res.append(work_on_area(t, do_plot=False))
res = np.array(res)
fig = plt.figure(figsize=(8,10))
plt.subplot(511)
plt.title('Minimum position')
plt.plot(res[:, 0:2])
plt.legend(['K1', 'K2'])
plt.xlim((0,300))
plt.subplot(512)
plt.title('Explained variance')
plt.plot(res[:, 6:8])
plt.legend(['PCA#1', 'PCA#2'])
plt.xlim((0,300))
plt.subplot(513)
plt.title('PCA#1 tang')
plt.plot(res[:, 2] / res[:, 3])
plt.xlim((0,300))
plt.subplot(514)
plt.title('Average distance between sources')
plt.plot(np.average(np.abs(diff), axis=1))
plt.xlim((0,300))
plt.subplot(515)
plt.title('Observations at t0')
plt.plot(m_fc[0:300,0])
plt.xlim((0,300))
plt.tight_layout()

fig = plt.figure(figsize=(4,4))
plt.plot(res[:, 0], res[:, 1], 'o')
plt.xlabel('K1')
plt.ylabel('K2')

predicted = []
kf = KFold(n_splits=len(diff))
for train, test in kf.split(diff):
    sv = SVR()
    sv.fit(diff[train], res[train, 0].flatten())
    k1_predicted = sv.predict(diff[test])
    sv.fit(diff[train], res[train, 1].flatten())
    k2_predicted = sv.predict(diff[test])
    predicted.append([k1_predicted[0], k2_predicted[0]])
predicted = np.array(predicted)

# TODO: Изменение площади покрытия (пересечение, старая-новая) при переходе
# TODO: Расстояние текущего минимума от глобального
