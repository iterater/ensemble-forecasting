import matplotlib.pyplot as plt
import numpy as np


indexRange = np.arange(N)
plt.figure(1, figsize=(12, 5))
plt.plot(indexRange, forecastError(h), indexRange, forecastError(s), indexRange, forecastError(n))
plt.legend(['Hiromb', 'Swan', 'No-Swan'])
plt.title('Sources MAE')
plt.xlabel('Forecast index')
plt.ylabel('Forecast MAE, cm')
plt.savefig('sources-mae.png')
plt.close()
