# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:37:22 2022

@author: dogiv
Multilayer perceptron, with a bias node in each layer.
Mini-batch stochastic gradient descent.
"""

import numpy as np
from numpy.random import rand
from sklearn.datasets import make_circles


def relu(x):
    return np.where(x > 0, x, 0)
def drelu(x): # relu has the property f'(f(x)) = f'(x). So does leaky relu.
    return np.where(x > 0, 1, 0)
def sigma(x):
    return 1./(1. + np.exp(-x))
def dsigma(sig): # dsigma has the property dsigma(sigma(x)) = d/dx sigma(x)
    return sig*(1-sig)
def sqrelu(x):
    return np.where(x > 0, np.sqrt(x), 0)
def dsqrelu(sqr):
    return np.where(sqr > 0, 1.5/sqr, 0)

#g = sigma
#gprime = dsigma
g = relu
gprime = drelu 

#def go(x): return x
#def goprime(x): return 1 # this needs to have the same property as dsigma above,
# namely goprime(go(x)) = d/dx go(x). 
go = sigma
goprime = dsigma

def feedfwd(w, x):
    a = [x]
    o = [x]
    for k in range(len(w)):
        outputkplusbias = np.hstack((o[k], np.ones((len(o[k]), 1))))
        a.append(outputkplusbias @ w[k])
        o.append(g(a[k+1]))
    o[-1] = go(a[-1])
    return o

def train(w, X, Y, steps=10000):
    C = []
    E2 = []
    alpha = 0.06
    B = len(X)
    if len(X) > 200:
        B = 20 # batch size
        alpha = alpha / 5
    for i in range(steps):
        if B < len(X):
            # sample some random data points from X and use those as the batch
            # This method allows using a sample more than once per batch,
            # I should probably fix that at some point.
            randlist = [np.random.randint(len(X)) for i in range(B)]
            x = np.vstack([X[randlist[i]] for i in range(B)])
            y = np.vstack([Y[randlist[i]] for i in range(B)])
            if i % 10 == 0:
                x = X
                y = Y
        else:
            x = X
            y = Y
        # Run network with current weights
        o = feedfwd(w, x)
        # Calculate error
        yhat = o[-1]
        E = yhat - y
        E2.append(E**2) # might have to put a sum here if there are multiple output nodes?
        if B == len(X) or i % 10 == 0:
            C.append(sum(E2[-1]) / len(x))
        else:
            C.append(C[-1])
        # Calculate gradient of C w/r to a[k], working backwards to k=0
        delta = [0 for i in range(len(o))]
        delta[-1] = goprime(o[-1]) * E / len(x) # last layer has a special activation function
        for k in range(len(w)-1, 0, -1):
            ek = w[k][:-1,:] @ delta[k+1].T
            delta[k] = gprime(o[k]) * ek.T 
        # Update weights
        for k in range(len(w)):
            outputkplusbias = np.hstack((o[k], np.ones((len(o[k]), 1))))
            w[k] += -alpha * outputkplusbias.T @ delta[k+1]
    return E2, C

#x = np.array([[0,0,1],
#              [0,1,1],
#              [1,0,0],
#               [1,1,0],
#               [1,0,1],
#               [1,1,1]])

# y = np.array([[0],
#               [1],
#               [1],
#               [0],
#               [1],
#               [0]])
x, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
y = np.vstack(y)

#np.random.seed(0)

# Specify the number of nodes in each hidden layer in this list:
r = [x.shape[1], 11, 13, 15, 11, y.shape[1]]
# Each layer's weight matrix has r[k] + 1 inputs (the +1 is for the bias)
# and r[k+1] outputs. 
w = [rand(r[k]+1,r[k+1])*2-1 for k in range(len(r)-1)]

e2, c = train(w, x, y)
import matplotlib.pyplot as plt
plt.plot(c)
