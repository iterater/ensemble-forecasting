import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA

# trying to see clusters of relative shapes
pLevel = 80
src_id = 3  # 3 - for measurements
p_m = np.genfromtxt('data\\PEAK_PARAMS_S1_'+str(pLevel).zfill(3)+'.csv', delimiter=',')
p_m = p_m[:, (src_id*4 + 1):(src_id*4 + 5)]
mask = ~np.any(np.isnan(p_m), axis=1)
p_m = np.vstack((p_m[mask][:, 0], p_m[mask][:, 1],
                 p_m[mask][:, 2] + p_m[mask][:, 3],
                 p_m[mask][:, 2] / (p_m[mask][:, 2] + p_m[mask][:, 3]))).transpose()

pca = PCA(n_components=4)
pca.fit(p_m)
print('Explained variance:', pca.explained_variance_)
print('Components:\n', pca.components_)
pca_res = pca.transform(p_m)

# plt.plot(p_m[:, 2] / (p_m[:, 3] + p_m[:, 2]), p_m[:, 1] / (p_m[:, 3] + p_m[:, 2]), 'o')
plt.figure(1, figsize=(8, 6))
n1 = 0
n2 = 1
plt.plot(pca_res[:, n1], pca_res[:, n2], 'o')
plt.xlabel('PC'+str(n1))
plt.ylabel('PC'+str(n2))
plt.savefig('pics\\peak_params_pc1_pc2.png')
plt.close()

plt.figure(1, figsize=(8, 6))
gkde = gaussian_kde(pca_res[:, 0], bw_method=0.08)
lim = [min(pca_res[:, 0]), max(pca_res[:, 0])]
x = np.linspace(lim[0], lim[1], 200)
pdf = gkde(x)
plt.plot(x, pdf, linewidth=2)
plt.hist(pca_res[:, 0], histtype='step', normed=True, bins=20)
plt.xlabel('PC1')
plt.ylabel('PDF')
plt.savefig('pics\\peak_params_pc1_pdf.png')
plt.close()

