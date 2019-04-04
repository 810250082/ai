from sklearn.preprocessing import MinMaxScaler
import numpy as np
a = np.array([[1, 2, 3, 4, 5]], dtype=np.float32)

ss = MinMaxScaler()
a = ss.fit_transform(a.T)
print(a)
print(ss.min_)
print(ss.scale_)
b = 1