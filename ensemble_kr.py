import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n_start = 70
n_stop = 429

original_errors = np.empty((0, 7))
predicted_errors = np.empty((0, 7))

for i in range(n_start, n_stop + 1):
    e = np.loadtxt('data\\2011\\kr\\'+str(i))
    original_errors = np.vstack((original_errors, e[1:, 0].flatten()))
    predicted_errors = np.vstack((predicted_errors, e[1:, 1].flatten()))

original_best_class = np.argmin(original_errors, axis=1)
print([sum(original_best_class == i) for i in range(7)])
print(sum(original_best_class == 6) / sum(original_best_class))
predicted_best_class = np.argmin(predicted_errors, axis=1)
predicted_selected_error = [original_errors[i, predicted_best_class[i]] for i in range(len(predicted_best_class))]
print(np.mean((original_errors[:, -1] - predicted_selected_error)/original_errors[:, -1]))

colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

# original-selected DTW error
full_err = original_errors[:, -1]
selected_error = [original_errors[i, predicted_best_class[i]] for i in range(len(predicted_best_class))]
plt.figure(1, figsize=(6, 6))
plt.xlabel('Full ensemble DTW error, cm')
plt.ylabel('Selected ensemble DTW error, cm')
plt.xlim((0, 9))
plt.ylim((0, 9))
plt.plot([-1, 10], [-1, 10], 'k')
plt.plot(original_errors[:, -1], selected_error, 'o')
plt.savefig('pics\\2011\\selected-vs-original-errors.png')
plt.close()

# original-predicted DTW biplot
plt.xlabel('Original DTW error, cm')
plt.ylabel('Predicted DTW error, cm')
plt.xlim((-0.5, 9.5))
plt.ylim((-0.5, 9.5))
plt.plot([-1, 10], [-1, 10], 'k')
for i in range(7):
    plt.plot(original_errors[:, i], predicted_errors[:, i], 'o'+colors[i])
plt.savefig('pics\\2011\\predicted-vs-original-errors.png')
plt.close()

# error on error estimation
print("ESTIMATION MEAN:", np.mean(original_errors - predicted_errors, axis=0))
print("ESTIMATION STD:", np.std(original_errors - predicted_errors, axis=0))

# pca on distance vector
pca = PCA(n_components=2)
pca_res = pca.fit_transform(predicted_errors)
plt.figure(10)
# ax = fig.add_subplot(111, projection='3d')
plt.xlabel('PC#1')
plt.ylabel('PC#2')
correct_flag = original_best_class == predicted_best_class
plt.plot(pca_res[correct_flag, 0], pca_res[correct_flag, 1], 'og')
plt.plot(pca_res[~correct_flag, 0], pca_res[~correct_flag, 1], 'or')
plt.show()
# plt.savefig('pics\\2011\\predicted-errors-pca.png')
# plt.close()

