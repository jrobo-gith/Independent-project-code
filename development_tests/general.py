import numpy as np

data = np.zeros((4, 3, 300))
print(data.shape)
data = data.reshape(12, 300)
print(data.shape)