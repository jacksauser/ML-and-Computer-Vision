import numpy as np
from process import get_data

X, Y, _, _, = get_data()

# randomly initialize weights 
M = 5 # number of hidden units
D = X.shape[1] # number of index features
K = len(set(Y)) # the number of unique categories

W1 = np.random.rand(D, M)
b1 = np.zeros(M)
W2 = np.random.rand(M, K)
b2 = np.zeros(K)

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2)

# P_Y_given_X = forward(X, W1, b1, W2, b2)

# preds = np.argmax(P_Y_given_X, axis=1)


# def classification_rate(Y, P):
#     return np.mean(Y == P)

# print("Score:", classification_rate(Y, preds))