import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import svm
import seaborn as sns

sns.set_style("whitegrid")

base_dir = '/media/iterater/DATA/_SRC_/ensemble-forecasting/data/2011/'

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

def work_on_area(t):
    plt.figure(figsize=(4,4))
    lst = []
    for i in range(N):
        for j in range(N):
            ff[i,j] = np.std(ens([s[t], h[t]], [k_arg[i],k_arg[j]]) - m_fc[t])
            lst.append([k_arg[i],k_arg[j],ff[i,j]])        
    lst = np.array(lst)
    ff_min = np.min(ff)
    ff_sel = lst[lst[:,2].flatten() < ff_min*1.1][:,0:2].transpose()
    # plt.contour(k_arg, k_arg, ff - np.min(ff))
    plt.plot(ff_sel[0], ff_sel[1], 'o')
    plt.xlim(lims)
    plt.ylim(lims)

work_on_area(2)



# plt.savefig('fig-{0}.png'.format(t))
   

 
# TODO: slope, intercept, squizness
