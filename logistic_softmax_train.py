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


def forward(X, W, b):
    return softmax(X.dot(W) + b)

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(Y, pY):
    # N, _ = Y.shape
    # return -(np.sum(Y * np.log(pY)))/N
    return -np.mean(Y * np.log(pY))

def main():
    Xtrain, Ytrain, Xtest, Ytest = get_data()

    D = Xtrain.shape[1]
    K = len(set(Ytrain) | set(Ytest))

    # convert to indicator
    Ytrain_ind = y2indicator(Ytrain, K)
    Ytest_ind = y2indicator(Ytest, K)

    # randomly initialize weights
    W = np.random.randn(D, K)
    b = np.zeros(K)

    train_costs = []
    test_costs = []
    learning_rate = 0.001
    # run for 10,000 epochs in this example
    for i in range(10000):
        pYtrain = forward(Xtrain, W, b)
        pYtest = forward(Xtest, W, b)

        ctrain = cross_entropy(Ytrain_ind, pYtrain)
        ctest = cross_entropy(Ytest_ind, pYtest)

        train_costs.append(ctrain)
        test_costs.append(ctest)

        # gradient descent
        W -= learning_rate * Xtrain.T.dot(pYtrain - Ytrain_ind)
        b -= learning_rate * (pYtrain - Ytrain_ind).sum(axis=0)

        # if i % 1000 == 0:
        #     print(i, ctrain, ctest)

    acc_train = classification_rate(Ytrain, predict(pYtrain))
    print('Score:', acc_train)    

    acc_test = classification_rate(Ytest, predict(pYtest))
    print('Score - Test:', acc_test)  

    plt.plot(train_costs, label = 'train cost')
    plt.plot(test_costs, label = 'test cost')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()