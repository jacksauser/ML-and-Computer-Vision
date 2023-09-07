import numpy as np

x = np.array([[1,2]])
w1 = np.array([[1,1],[1,0]])
w2 = np.array([[0,1],[1,1]])
b1 = 0
b2 = 0

expA = np.exp(np.tanh(x.dot(w1)).dot(w2))

print(expA / expA.sum(axis=1, keepdims=True))