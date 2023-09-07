import numpy as np 
import matplotlib.pyplot as plt

# number of classes
Nclass = 500

# gaussian cloud                \/ changing their centers
X1 = np.random.randn(Nclass,2) + np.array([0,-2])
X2 = np.random.randn(Nclass,2) + np.array([2,2])
X3 = np.random.randn(Nclass,2) + np.array([-2,2])
X = np.vstack([X1,X2,X3])

print(X)

Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)

plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show()

D = 2 
M = 3 # hidden layer count
K = 3 # number of classes 

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def foward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1)) # sigmoid or the value at the hidden layer

    # Softmaze of the next layer
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y  = expA / expA.sum(axis=1, keepdims=True)

    return Y

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0 
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct +=1

    return n_correct / n_total

prob_Y = foward(X, W1, b1, W2, b2)

P = np.argmax(prob_Y, axis = 1)

assert(len(P) == len(Y))
print(Y)
plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5, )
plt.show()
print("Classification rate with random weights", classification_rate(Y, P))