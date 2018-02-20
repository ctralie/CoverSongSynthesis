import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from NMF import *

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
    Implement the technique in [1] for performing joint nonnegative matrix
    factorization on two or more views of the same process that are synchronized
    [1] "Multi-View Clustering via Joint Nonnegative Matrix Factorization"
        Jialu Liu, Chi Wang, Ning Gao, Jiawei Han
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

def getJointEuclideanError(A, Ap, B, W1, W2, H1, H2):
    res = getEuclideanError(A, multiplyConv2D(W1, H1))
    res += getEuclideanError(Ap, multiplyConv2D(W2, H1))
    res += getEuclideanError(B, multiplyConv2D(W1, H2))
    return res

def doNMF2DConvJoint(A, Ap, B, K, T, F, L, plotfn = None):
    """
    Implementing the Euclidean 2D NMF technique described in 
    "Nonnegative Matrix Factor 2-D Deconvolution
        for Blind Single Channel Source Separation"
    :param A: An M x N1 matrix for song A
    :param Ap: An M x N1 matrix for song A'
    :param B: An M x N2 matrix for song B
    :param K: Number of latent factors
    :param T: Time extent of W matrices
    :param F: Frequency extent of H matrices
    :param L: Number of iterations
    :param plotfn: A function used to plot each iteration, which should\
        take the arguments (V, W, H, iter)
    :returns (W1, W2, H1, H2): \
        W1, W2 are an MxKxT matrices of K sources over spatiotemporal spans MxT\
        H1 is a KxN1xF matrix of source activations for the A, A' pair
        H2 is a KxN2xF matrix of source activations for B
    """
    if not (A.shape[1] == Ap.shape[1]):
        print("Error: A and A' should have same number of frames")
        return None
    M = A.shape[0]
    N1 = A.shape[1]
    N2 = B.shape[1]
    W1 = np.random.rand(M, K, T)
    W2 = np.random.rand(M, K, T)
    H1 = np.random.rand(K, N1, F)
    H2 = np.random.rand(K, N2, F)    

    errs = [getJointEuclideanError(A, Ap, B, W1, W2, H1, H2)]
    if plotfn:
        res=4
        plt.figure(figsize=((8+2*K)*res*1.2, 2*res))
        plotfn(A, Ap, B, W1, W2, H1, H2, 0, errs)
        plt.savefig("NMF2DConvJoint_%i.png"%0, bbox_inches = 'tight')
    for l in range(L):
        print("Joint 2DNMF iteration %i of %i"%(l+1, L))
        #Step 1: Update Ws
        Lam11 = multiplyConv2D(W1, H1)
        Lam12 = multiplyConv2D(W1, H2)
        Lam21 = multiplyConv2D(W2, H1)
        W1Nums = np.zeros(W1.shape)
        W1Denoms = np.zeros(W1.shape)
        W2Nums = np.zeros(W2.shape)
        W2Denoms = np.zeros(W2.shape)
        ticouter = time.time()
        for f in range(F):
            tic = time.time()
            thisA = shiftMatLRUD(A, di=-f)
            thisAp = shiftMatLRUD(Ap, di=-f)
            thisB = shiftMatLRUD(B, di=-f)
            thisLam11 = shiftMatLRUD(Lam11, di=-f)
            thisLam12 = shiftMatLRUD(Lam12, di=-f)
            thisLam21 = shiftMatLRUD(Lam21, di=-f)
            for t in range(T):
                thisH1T = (shiftMatLRUD(H1[:, :, f], dj=t)).T
                thisH2T = (shiftMatLRUD(H2[:, :, f], dj=t)).T
                W1Nums[:, :, t] += thisA.dot(thisH1T) + thisB.dot(thisH2T)
                W1Denoms[:, :, t] += thisLam11.dot(thisH1T) + thisLam12.dot(thisH2T)
                W2Nums[:, :, t] += thisAp.dot(thisH1T)
                W2Denoms[:, :, t] += thisLam21.dot(thisH1T)
            print("Elapsed Time Ws Phi=%i: %.3g"%(f, time.time()-tic))
        W1 = W1*(W1Nums/W1Denoms)
        W2 = W2*(W2Nums/W2Denoms)
        print("Elapsed Time All Ws: %.3g"%(time.time()-ticouter))

        #Step 2: Update Hs
        Lam11 = multiplyConv2D(W1, H1)
        Lam12 = multiplyConv2D(W1, H2)
        Lam21 = multiplyConv2D(W2, H1)
        H1Nums = np.zeros(H1.shape)
        H1Denoms = np.zeros(H1.shape)
        H2Nums = np.zeros(H2.shape)
        H2Denoms = np.zeros(H2.shape)
        ticouter=time.time()
        for t in range(T):
            tic = time.time()
            thisA = shiftMatLRUD(A, dj=-t)
            thisAp = shiftMatLRUD(Ap, dj=-t)
            thisB = shiftMatLRUD(B, dj=-t)
            thisLam11 = shiftMatLRUD(Lam11, dj=-t)
            thisLam12 = shiftMatLRUD(Lam12, dj=-t)
            thisLam21 = shiftMatLRUD(Lam21, dj=-t)
            for f in range(F):
                thisW1T = (shiftMatLRUD(W1[:, :, t], di=f)).T
                thisW2T = (shiftMatLRUD(W2[:, :, t], di=f)).T
                H1Nums[:, :, f] += thisW1T.dot(thisA) + thisW2T.dot(thisAp)
                H1Denoms[:, :, f] += thisW1T.dot(thisLam11) + thisW2T.dot(thisLam21)
                H2Nums[:, :, f] += thisW1T.dot(thisB)
                H2Denoms[:, :, f] += thisW1T.dot(thisLam12)
            print("Elapsed time H t=%i, %.3g"%(t, time.time() - tic))
        H1 = H1*(H1Nums/H1Denoms)
        H2 = H2*(H2Nums/H2Denoms)
        print("Elapsed Time All Hs: %.3g"%(time.time()-ticouter))
        errs.append(getJointEuclideanError(A, Ap, B, W1, W2, H1, H2))
        if plotfn and ((l+1) == L or (l+1)%40 == 0):
            plt.clf()
            plotfn(A, Ap, B, W1, W2, H1, H2, l+1, errs)
            plt.savefig("NMF2DConvJoint_%i.png"%(l+1), bbox_inches = 'tight')
    return (W1, W2, H1, H2)


def plotNMF2DConvJointSpectra(A, Ap, B, W1, W2, H1, H2, iter, errs, \
        hopLength = -1, audioParams = None):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H
    :param A: An M x N1 matrix for song A
    :param Ap: An M x N1 matrix for song A'
    :param B: An M x N2 matrix for song B
    :param W1: An M x K x T source/corpus matrix for songs A and B
    :param W2: An M x K x T source/corpus matrix for song A'
    :param H1: A K x N1 x F matrix of activations for A and A'
    :param H2: A K x N2 x F matrix of activations for B
    :param iter: The iteration number
    :param errs: Errors over time
    :param hopLength: The hop length (for plotting)
    :param audioParams: Parameters for inverting CQT
    """
    import librosa
    import librosa.display
    K = W1.shape[1]

    Lam11 = multiplyConv2D(W1, H1)
    Lam12 = multiplyConv2D(W1, H2)
    Lam21 = multiplyConv2D(W2, H1)
    Lam22 = multiplyConv2D(W2, H2)
    for k, (V, Lam, s1, s2) in enumerate(zip([A, Ap, B, None], [Lam11, Lam21, Lam12, Lam22], \
            ["A", "Ap", "B", "Bp"],\
            ["$\Lambda_{W1, H1}$", "$\Lambda_{W2, H1}$", "$\Lambda_{W1, H2}$", "$\Lambda_{W2, H2}$"])):
        if k < 3:
            plt.subplot(2, 8+2*K, k+1)
            if hopLength > -1:
                librosa.display.specshow(librosa.amplitude_to_db(V), hop_length = hopLength, \
                                            y_axis = 'log', x_axis = 'time')
            else:
                plt.imshow(V, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
                plt.colorbar()
            plt.title(s1)

        plt.subplot(2, 8+2*K, 8+2*K+k+1)
        if hopLength > -1:
            librosa.display.specshow(librosa.amplitude_to_db(Lam), hop_length = hopLength, \
                y_axis = 'log', x_axis = 'time')
        else:
            plt.imshow(Lam, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
            plt.colorbar()
        plt.title("%s Iteration %i"%(s2, iter))

        if audioParams:
            from SpectrogramTools import griffinLimCQTInverse
            from scipy.io import wavfile
            [Fs, bins_per_octave] = [audioParams['Fs'], audioParams['bins_per_octave']]
            y_hat = griffinLimCQTInverse(Lam, Fs, hopLength, bins_per_octave, NIters=10)
            y_hat = y_hat/np.max(np.abs(y_hat))
            sio.wavfile.write("%s_Iter%i.wav"%(s1, iter), Fs, y_hat)

    plt.subplot(2, 8+2*K, 4)
    errs = np.array(errs)
    if len(errs) > 1:
        errs = errs[1::]
    plt.semilogy(errs)
    plt.title("Errors")

    for k in range(K):
        plt.subplot(2, 8+2*K, 4+k+1)
        if hopLength > -1:
            librosa.display.specshow(librosa.amplitude_to_db(W1[:, k, :]), \
                hop_length=hopLength, y_axis='log', x_axis='time')
        else:
            plt.imshow(W1[:, k, :], cmap = 'afmhot', \
                    interpolation = 'nearest', aspect = 'auto')  
            plt.colorbar()
        plt.title("W1%i"%k)

        plt.subplot(2, 8+2*K, 8+2*K+4+k+1)
        if hopLength > -1:
            librosa.display.specshow(librosa.amplitude_to_db(W2[:, k, :]), \
                hop_length=hopLength, y_axis='log', x_axis='time')
        else:
            plt.imshow(W2[:, k, :], cmap = 'afmhot', \
                    interpolation = 'nearest', aspect = 'auto')  
            plt.colorbar()
        plt.title("W2%i"%k)

        plt.subplot(2, 8+2*K, K+4+k+1)
        plt.imshow(H1[k, :, :].T, cmap = 'afmhot', \
            interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
        plt.title("H1%i"%k)

        plt.subplot(2, 8+2*K, 8+2*K+4+K+k+1)
        plt.imshow(H2[k, :, :].T, cmap = 'afmhot', \
            interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
        plt.title("H2%i"%k)