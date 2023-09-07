import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from process import get_data
from predict import softmax

def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2) , Z

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(Y, pY):
    return -np.mean(Y * np.log(pY))

def main():

    Xtrain, Ytrain, Xtest, Ytest = get_data()

    D = Xtrain.shape[1] # num rows
    K = len(set(Ytrain)|set(Ytest)) # num unique outputs
    M = 5 # arbitrary

    Ytrain_ind = y2indicator(Ytrain, K)
    Ytest_ind = y2indicator(Ytest, K)

    # randomly initialize weights
    W1 = np.random.randn(D, M)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K)
    b2 = np.zeros(K)

    # training loop
    train_costs = []
    test_costs = []
    learning_rate = 0.001 # arbitrary 
    epochs = 5000 # arbitrary

    for i in range(epochs):
        pYtrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
        pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)

        ctrain = cross_entropy(Ytrain_ind, pYtrain)
        ctest = cross_entropy(Ytest_ind, pYtest)

        train_costs.append(ctrain)
        test_costs.append(ctest)

        # gradients (optional)
        gW2 = Ztrain.T.dot(pYtrain - Ytrain_ind)
        gb2 = (pYtrain - Ytrain_ind).sum(axis=0)
        dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain * Ztrain) # hold variable
        gW1 = Xtrain.T.dot(dZ)
        gb1 = dZ.sum(axis=0)
        
        # updating weights
        W2 -= learning_rate * gW2
        b2 -= learning_rate * gb2
        W1 -= learning_rate * gW1
        b1 -= learning_rate * gb1

        if i % 1000 == 0:
            print(i, ctrain, ctest)

    pYtrain, _ = forward(Xtrain, W1, b1, W2, b2)
    pYtest, _ = forward(Xtest, W1, b1, W2, b2)

    acc_test = classification_rate(Ytest, predict(pYtest))
    acc_train = classification_rate(Ytrain, predict(pYtrain))
    print('Final train classification rate:', acc_train)    
    print('Final test classification rate:', acc_test)  


    plt.plot(train_costs, label = 'train cost')
    plt.plot(test_costs, label = 'test cost')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()