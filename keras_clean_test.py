# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense
# sns.set_style("whitegrid")
plt.style.use('seaborn')

# base_dir = 'd:/GitHub/python-2-work/' # work
base_dir = 'd:/_SRC_/ensemble-forecasting/'  # home
h = np.loadtxt(base_dir + 'data/2011/2011080100_hiromb_GI_60x434.txt') - 37.356
s = np.loadtxt(base_dir + 'data/2011/2011080100_swan_GI_48x434.txt')
n = np.loadtxt(base_dir + 'data/2011/2011080100_noswan_GI_48x434.txt')
m = np.loadtxt(base_dir + 'data/2011/2011080100_measurements_GI_2623.txt')
# forecast dimensions
N = 100 # 430
T = 48
# preparing measurements forecast
m_fc = np.zeros((N, T+1))
for i in range(N):
    for j in range(T+1):
        m_fc[i, j] = m[i*6+j]

# data cut and shifting
shift_const = 0
h = h[0:N, 1:T+1] + shift_const
s = s[0:N, 1:T+1] + shift_const
n = n[0:N, 1:T+1] + shift_const
m_fc = m_fc[0:N, 1:T+1] + shift_const

a = opt.minimize(lambda a: np.mean(np.abs(a[0] + a[1]*h + a[2]*s - m_fc)), [0, 0.5, 0.5]).x
ens = a[0] + a[1]*h + a[2]*s
def plot_ensemble(q):
    plt.figure()
    plt.plot(h[q], 'r-')
    plt.plot(s[q], 'g-')
    plt.plot(m_fc[q], 'ko')
    plt.plot(ens[q], 'm-', linewidth=2)
    plt.title('h:{0:.2f} s:{1:.2f} ens:{2:.2f}'.format(np.mean(np.abs(h[q] - m_fc[q])),
              np.mean(np.abs(s[q] - m_fc[q])),
              np.mean(np.abs(ens[q] - m_fc[q]))))
    plt.show()

def plot_summary(t_h, t_s, t_ens, t_m_fc, t_predicted, t_q):
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.plot(t_ens[t_q], 'm-', linewidth=2, label='ens')
    plt.plot(t_predicted[t_q], 'b-', linewidth=2, label='ann')
    plt.plot(t_m_fc[t_q], 'ko', label='obs')
    plt.legend()
    plt.title('ens:{0:.2f} ann:{1:.2f}'.format(np.mean(np.abs(t_ens[t_q] - t_m_fc[t_q])),
                                               np.mean(np.abs(t_predicted[t_q] - t_m_fc[t_q]))))
    plt.xlabel('Forecast time, h')
    plt.ylabel('Level, cm')
    plt.subplot(132)
    sns.kdeplot(np.mean(np.abs(t_h - t_m_fc), axis=1), label='h')
    sns.kdeplot(np.mean(np.abs(t_s - t_m_fc), axis=1), label='s')
    sns.kdeplot(np.mean(np.abs(t_ens - t_m_fc), axis=1), label='ens')
    sns.kdeplot(np.mean(np.abs(t_predicted - t_m_fc), axis=1), label='ann')
    plt.xlabel('MAE, cm')
    plt.ylabel('PDF')
    plt.subplot(133)
    plt.plot(np.mean(np.abs(t_ens - t_m_fc), axis=1),
             np.mean(np.abs(t_predicted - t_m_fc), axis=1),
             'ob')
    plt.plot(np.mean(np.abs(t_ens[t_q] - t_m_fc[t_q])),
             np.mean(np.abs(t_predicted[t_q] - t_m_fc[t_q])),
             'or')
    plt.plot([0,10], [0,10], 'k-')
    plt.xlabel('MAE(ens), cm')
    plt.ylabel('MAE(ann), cm')
    plt.tight_layout()

freq = np.absolute(np.fft.fft(m_fc[:,-32:], axis=1)[:,0:16])

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def whole_fc_m_ann_ensemble():
    model = Sequential()
    model.add(Dense(units=T*3+16, kernel_initializer='normal', activation='relu', input_dim=T*3+16))
    model.add(Dense(units=T*3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(units=T, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
X = np.hstack((h[8:], s[8:], m_fc[:-8], freq[:-8]))
Y = m_fc[8:]
seed = 7
np.random.seed(seed)
estimator = KerasRegressor(build_fn=whole_fc_m_ann_ensemble, epochs=14, batch_size=10, verbose=0)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(X, Y)
print(estimator.model.summary())

predicted = estimator.predict(X)
plot_summary(h[8:], s[8:], ens[8:], m_fc[8:], predicted, 50)
plt.savefig('pics\\keras_clean_test_out.png')
