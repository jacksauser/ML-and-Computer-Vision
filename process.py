import numpy as np 
import pandas as pd

def get_data():
    df = pd.read_csv('data/ecommerce_data.csv')

    ###################### pretty standard stuff
    data = df.to_numpy()

    # shuffle
    np.random.shuffle(data)

    # split features and labels
    X = data[:, :-1]
    Y = data[:, -1].astype(np.int32)

    N, D = X.shape
    ######################
    
    Xfinal = np.zeros((N, D+3)) # 3 extra columns are needed to one-hot encode a feature with 4 catagories
    Xfinal[:, :(D-1)] = X[:, :(D-1)] # non-categorical columns

    # one hot encoding using a for loop
    # for n in range(N):
    #     t = int(X[n, D-1])
    #     Xfinal[n, t+D-1] = 1 

    # one hot encoding using vectorization 
    Z = np.zeros((N,4))  
    Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
    Xfinal[:, -4:] = Z


    # split into test and train
    Xtrain = Xfinal[:-100]
    Ytrain = Y[:-100]
    Xtest = Xfinal[-100:]
    Ytest = Y[-100:]

    # normalizing columns 1 & 2
    for i in (1,2):
        m = Xtrain[:,i].mean()
        s = Xtrain[:,i].std()
        Xtrain[:,i] = (Xtrain[:,i] - m) / s
        Xtest[:,i] = (Xtest[:,i] - m) / s

    return Xtrain, Ytrain, Xtest, Ytest


def get_binary_data():
    Xtrain, Ytrain, Xtest, Ytest = get_data()
    
    # looking for all Y values with a 1 or 0
    X2train = Xtrain[Ytrain <= 1]
    Y2train = Ytrain[Ytrain <= 1]
    X2test = Ytest[Ytest <= 1]
    Y2test = [Ytest <= 1]

    return X2train, Y2train, X2test, Y2test