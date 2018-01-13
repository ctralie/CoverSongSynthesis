import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def jointNMFObjOuter(Xs, Us, Vs, VStar, lambdas):
    Qs = [np.sum(U, 0) for U in Us]
    ret = 0.0
    for i in range(Xs.shape[0]):
        ret += np.sum((Xs[i] - Us[i].dot(Vs[i].T))**2)
        ret += lambdas[i]*np.sum((Vs[i]*Qs[i][None, :] - VStar)**2)
    return ret

def jointNMFObjInner(X, U, V, VStar, lam):
    #Equation 3.7
    Q = np.sum(U, 0)
    ret = np.sum((X-U.dot(V.T))**2)
    ret += lam*np.sum((V*Q[None, :]-VStar)**2)
    return ret

def getImprovement(errs, Verbose = False):
    [a, b] = errs[-2::]
    improvement = (a-b)/a
    if Verbose:
        print("%g (%g %s improvement)"%(b, improvement*100, '%'))
    return improvement

def doJointNMF(pXs, lambdas, K, NIters, tol = 0.05, Verbose = False):
    """
    
    """
    #Normalize each view so that the L1 matrix norm is 1
    Xs = [X/np.sum(X) for X in pXs]
    #Randomly initialize Us and Vs
    Us = [np.random.rand(X.shape[0], K) for X in Xs]
    Vs = [np.random.rand(X.shape[1], K) for X in Xs]
    VStar = np.random.rand(Xs[0].shape[1], K)
    #Objective function
    errsOuter = [jointNMFObjOuter(Xs, Us, Vs, VStar, lambdas)]
    convergedOuter = False
    while not convergedOuter:
        for i, (X, U, V, lam) in enumerate(zip(Xs, Us, Vs, lambdas)):
            errsInner = [jointNMFObjInner(X, U, V, VStar, lam)]
            convergedInner = False
            while not convergedInner:
                print("Iteration %i - %i - %i"%(len(errsOuter), i+1, len(errsInner)))
                #Fixing V* and Vi, update Ui by Equation 3.8
                num = X.dot(V) + lam*np.sum(V*VStar, 0)[None, :]
                VSqrCols = np.sum(V**2, 0)
                denom = U.dot((V.T).dot(V)) + \
                    lam*np.sum(U*VSqrCols[None, :], 0)[None, :]
                U = U*(num/denom)
                #Normalize U and V as in Equation 3.9
                Q = np.sum(U, 0)
                U = U/Q[None, :]
                V = V*Q[None, :]
                #Fixing V* and U, update V by Equation 3.10
                num = (X.T).dot(V) + lam*VStar
                denom = V.dot((U.T).dot(U)) + lam*V
                V = V*(num/denom)
                #Check for convergence
                errsInner.append(jointNMFObjInner(X, U, V, VStar, lam))
                if getImprovement(errsInner, Verbose) < tol:
                    convergedInner = True
        #Fixing Us and Vs, update V* by Equation 3.11
        VStar *= 0
        for U, V, lam in zip(Us, Vs, lambdas):
            VStar += lam*V*np.sum(U, 0)[None, :]
        VStar /= np.sum(lambdas)
        #Check for convergence
        errsOuter.append(jointNMFObjOuter(Xs, Us, Vs, VStar, lambdas))
        if getImprovement(errsOuter, Verbose) < tol:
            convergedOuter = True
    return {'Vs':Vs, 'Us':Us, }

