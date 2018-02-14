"""
Implement the technique in [1] for performing joint nonnegative matrix
factorization on two or more views of the same process that are synchronized
[1] "Multi-View Clustering via Joint Nonnegative Matrix Factorization"
    Jialu Liu, Chi Wang, Ning Gao, Jiawei Han
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def plotJointNMFwGT(Xs, Us, Vs, VStar, UsGT, VsGT, errs):
    """
    :param Xs: List of X matrices that are being factorized
    :param Us: List of U (template) matrices
    :param Vs: List of matrices holding loadings
    :param VStar: Consensus loading matrix
    :param UsGT: Ground truth Us
    :param VsGT: Ground truth Vs
    :param errs: List of convergence errors on iterations so far
    """
    N = len(Xs)
    for i in range(N):
        #Subplots: X, U*V, U, V, UGT, VGT
        plt.subplot(N+1, 6, 6*i + 1)
        plt.imshow(Xs[i], cmap = 'afmhot', aspect = 'auto', interpolation = 'nearest')
        plt.title("X%i"%i)
        plt.colorbar()
        plt.subplot(N+1, 6, 6*i + 2)
        plt.imshow(Us[i].dot(Vs[i].T), cmap = 'afmhot', aspect = 'auto', interpolation = 'nearest')
        plt.title("U%i*V%i^T"%(i, i))
        plt.colorbar()
        plt.subplot(N+1, 6, 6*i + 3)
        plt.imshow(VsGT[i], cmap = 'afmhot', aspect = 'auto', interpolation = 'nearest')
        plt.title("VGT%i"%i)
        plt.colorbar()
        plt.subplot(N+1, 6, 6*i + 4)
        plt.imshow(Vs[i], cmap = 'afmhot', aspect = 'auto', interpolation = 'nearest')
        plt.title("V%i"%i)
        plt.colorbar()
        plt.subplot(N+1, 6, 6*i + 5)
        plt.imshow(Us[i], cmap = 'afmhot', aspect = 'auto', interpolation = 'nearest')
        plt.title("U%i"%i)
        plt.colorbar()
        plt.subplot(N+1, 6, 6*i + 6)
        plt.imshow(UsGT[i], cmap = 'afmhot', aspect = 'auto', interpolation = 'nearest')
        plt.title("UGT%i"%i)
        plt.colorbar()
    plt.subplot(N+1, 6, 6*N + 4)
    plt.imshow(VStar, cmap = 'afmhot', aspect = 'auto', interpolation = 'nearest')
    plt.title("V*")
    plt.colorbar()
    plt.subplot2grid((N+1, 6), (N, 0), rowspan = 1, colspan = 3)
    plt.semilogy(np.arange(len(errs)), errs)
    plt.scatter([len(errs)-1], [errs[-1]])
    plt.title("Convergence Errors")
    plt.xlabel("Iteration Number")
    plt.ylabel("Error")

def plotJointNMFSpectra(Xs, Us, Vs, VStar, errs, hopLength):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H
    :param Xs: List of X matrices that are being factorized
    :param Us: List of U (template) matrices
    :param Vs: List of matrices holding loadings
    :param VStar: Consensus loading matrix
    :param errs: List of convergence errors on iterations so far
    :param hopLength: The hop length (for plotting)
    """
    import librosa
    import librosa.display
    ncols = 4
    N = len(Xs)
    for i in range(N):
        #Subplots: X, U*V, U, V, UGT, VGT
        plt.subplot(N+1, ncols, ncols*i + 1)
        librosa.display.specshow(librosa.amplitude_to_db(Xs[i]), hop_length = hopLength, \
                                y_axis = 'log', x_axis = 'time')
        plt.title("X%i"%i)
        plt.subplot(N+1, ncols, ncols*i + 2)
        librosa.display.specshow(librosa.amplitude_to_db(Us[i].dot(Vs[i].T)), \
                                hop_length = hopLength, y_axis = 'log', x_axis = 'time')
        plt.title("U%i*V%i^T"%(i, i))
        plt.subplot(N+1, ncols, ncols*i + 3)
        librosa.display.specshow(librosa.amplitude_to_db(Us[i]), hop_length = hopLength, \
                                y_axis = 'log', x_axis = 'time')
        plt.title("U%i"%i)
        plt.subplot(N+1, ncols, ncols*i + 4)
        plt.imshow(librosa.amplitude_to_db(Vs[i]), cmap = 'afmhot', \
                interpolation = 'nearest', aspect = 'auto')  
        plt.title("V%i"%i)
    plt.subplot(N+1, ncols, ncols*N + 4)
    plt.imshow(librosa.amplitude_to_db(VStar), cmap = 'afmhot', \
                aspect = 'auto', interpolation = 'nearest')
    plt.title("V*")
    plt.subplot2grid((N+1, ncols), (N, 0), rowspan = 1, colspan = 3)
    plt.semilogy(np.arange(len(errs)), errs)
    plt.scatter([len(errs)-1], [errs[-1]])
    plt.title("Convergence Errors")      

def jointNMFObjOuter(Xs, Us, Vs, VStar, lambdas):
    Qs = [np.sum(U, 0) for U in Us]
    ret = 0.0
    for i in range(len(Xs)):
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

def doJointNMF(pXs, lambdas, K, tol = 0.05, Verbose = False, plotfn = None):
    """
    :param pXs: An array of matrices which each have the same number of columns\
        which are assumed to be in correspondence
    :param lambdas: Weights of each X
    :param K: Rank of the decomposition (number of components)
    :param tol: Fraction of improvement below which to deem convergence
    :param Verbose: Whether to print debugging information
    :param plotfn: A function for plotting matrices during interations\
        expects arguments (Xs, Us, Vs, VStar, errs)
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
    if plotfn:
        res = 4
        plt.figure(figsize=(res*6, res*(len(Xs)+1)))
        plotfn(Xs, Us, Vs, VStar, errsOuter)
        plt.savefig("JointNMF%i.png"%0)
    while not convergedOuter:
        for i, (X, U, V, lam) in enumerate(zip(Xs, Us, Vs, lambdas)):
            errsInner = [jointNMFObjInner(X, U, V, VStar, lam)]
            convergedInner = False
            while not convergedInner:
                if Verbose:
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
                num = (X.T).dot(U) + lam*VStar
                denom = V.dot((U.T).dot(U)) + lam*V
                V = V*(num/denom)
                #Check for convergence
                errsInner.append(jointNMFObjInner(X, U, V, VStar, lam))
                if getImprovement(errsInner, Verbose) < tol:
                    convergedInner = True
            Us[i] = U
            Vs[i] = V
        #Fixing Us and Vs, update V* by Equation 3.11
        VStar *= 0
        for U, V, lam in zip(Us, Vs, lambdas):
            VStar += lam*V*np.sum(U, 0)[None, :]
        VStar /= np.sum(lambdas)
        #Check for convergence
        errsOuter.append(jointNMFObjOuter(Xs, Us, Vs, VStar, lambdas))
        if getImprovement(errsOuter, Verbose) < tol:
            convergedOuter = True
        if plotfn:# and convergedOuter:# (len(errsOuter)%10 == 0 or convergedOuter):
            plt.clf()
            plotfn(Xs, Us, Vs, VStar, errsOuter)
            plt.savefig("JointNMF%i.png"%len(errsOuter[0:-1]))
    return {'Vs':Vs, 'Us':Us, 'VStar':VStar, 'Xs':Xs, 'errsOuter':errsOuter}

