import numpy as np

n_start = 70
n_stop = 429

original_errors = np.empty((0, 7))
predicted_errors = np.empty((0, 7))

for i in range(n_start, n_stop + 1):
    e = np.loadtxt('data\\2011\\kr\\'+str(i))
    original_errors = np.vstack((original_errors, e[1:, 0].flatten()))
    predicted_errors = np.vstack((predicted_errors, e[1:, 1].flatten()))

original_best_class = np.argmin(original_errors, axis=1)
predicted_best_class = np.argmin(original_errors, axis=1)
predicted_selected_error = [original_errors[i, predicted_best_class[i]] for i in range(len(predicted_best_class))]
print(np.mean(original_errors[:, -1] - predicted_selected_error))

# error on error estimation
print("ESTIMATION MEAN:", np.mean(original_errors - predicted_errors, axis=0))
print("ESTIMATION STD:", np.std(original_errors - predicted_errors, axis=0))
