import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
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

def doNMF1DConvJoint(V, K, T, L, r = 0, p = -1, W = np.array([]), plotfn = None, \
        plotComponents = True, prefix=""):
    """
    Implementing the technique described in \
    "Non-negative Matrix Factor Deconvolution; Extraction of\
        Multiple Sound Sources from Monophonic Inputs"\
    NOTE: Using update rules from 2DNMF instead of averaging\
    It's understood that V and W are two songs concatenated\    
        on top of each other.  Functionality is the same, but errors are computed\
        separately, and plotting is done differently
    :param V: An N x M target matrix
    :param K: Number of latent factors
    :param T: Time extent of W matrices
    :param L: Number of iterations
    :param r: Width of the repeated activation filter
    :param p: Degree of polyphony
    :param plotfn: A function used to plot each iteration, which should\
        take the arguments (V, W, H, iter)
    :returns (W, H): \
        W is an TxNxK matrix of K sources over spatiotemporal spans NxT
        H is a KxM matrix of source activations for each column of V
    """
    import scipy.ndimage
    N = V.shape[0]
    M = V.shape[1]
    WFixed = False
    if W.size == 0:
        W = np.random.rand(T, N, K)
    else:
        WFixed = True
        K = W.shape[2]
        print("K = ", K)
    H = np.random.rand(K, M)
    WH = multiplyConv1D(W, H)
    N1 = int(N/2)
    errs = [[getKLError(V[0:N1, :], WH[0:N1, :]), \
            getKLError(V[N1::, :], WH[N1::, :])]]
    if plotfn:
        res=4
        pK = K
        if not plotComponents:
            pK = 0
        plt.figure(figsize=((4+pK)*res, 3*res))
    for l in range(L):
        #KL Divergence Version
        print("Joint 1DNMF iteration %i of %i"%(l+1, L))            
        #Step 1: Avoid repeated activations
        iterfac = 1-float(l+1)/L 
        R = np.array(H)
        if r > 0:
            MuH = scipy.ndimage.filters.maximum_filter(H, size=(1, r))
            R[R<MuH] = R[R<MuH]*iterfac
        #Step 2: Restrict number of simultaneous activations
        P = np.array(R)
        if p > -1:
            colCutoff = -np.sort(-R, 0)[p, :]
            P[P < colCutoff[None, :]] = P[P < colCutoff[None, :]]*iterfac
        H = P
        WH = multiplyConv1D(W, H)
        WH[WH == 0] = 1
        VLam = V/WH
        HNum = np.zeros(H.shape)
        HDenom = np.zeros((H.shape[0], 1))
        for t in range(T):
            thisW = W[t, :, :]
            HDenom += np.sum(thisW, 0)[:, None]
            HNum += (thisW.T).dot(shiftMatLRUD(VLam, dj=-t))
        H = H*(HNum/HDenom)
        if not WFixed:
            WH = multiplyConv1D(W, H)
            WH[WH == 0] = 1
            VLam = V/WH
            for t in range(T):
                HShift = shiftMatLRUD(H, dj=t)
                denom = np.sum(H, 1)[None, :]
                denom[denom == 0] = 1
                W[t, :, :] *= (VLam.dot(HShift.T))/denom
        WH = multiplyConv1D(W, H)
        N1 = int(N/2)
        errs.append([getKLError(V[0:N1, :], WH[0:N1, :]), \
                getKLError(V[N1::, :], WH[N1::, :])])
        if plotfn and ((l+1) == L):# or (l+1)%40 == 0):
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            plt.savefig("%s/NMF1DConv_%i.png"%(prefix, l+1), bbox_inches = 'tight')
    return (W, H)

def getJointError(A, Ap, W1, W2, H, errfn, mulfn):
    res = errfn(A, mulfn(W1, H))
    res += errfn(Ap, mulfn(W2, H))
    return res

def doNMF2DConvJoint(A, Ap, K, T, F, L, foldername = ".", doKL = False, plotfn = None):
    """
    Do a joint version of 2DNMF solving for W1, W2, and H, where 
    A ~= W1*H and Ap ~= W2*H
    :param A: An N x M target matrix for dataset 1
    :param Ap: An N x M target matrix for dataset 2
    :param K: Number of latent factors
    :param T: Time extent of W matrices
    :param F: Frequency extent of H matrices
    :param L: Number of iterations
    :param doKL: Whether to do Kullback-Leibler divergence.  If false, do Euclidean
    :param plotfn: A function used to plot each iteration, which should\
        take the arguments (V, W, H, iter)
    :returns (W1, W2, H): \
        W1 is an TxNxK matrix of K A sources over spatiotemporal spans NxT\
        W2 is an TxNxK matrix of K Ap sources over spatiotemporal spans NxT\
        H is a FxKxM matrix of source activations for each submatrix of W\
            over F transpositions over M time
    """
    N = A.shape[0]
    M = A.shape[1]
    H = np.random.rand(F, K, M)
    W1 = np.random.rand(T, N, K)
    W2 = np.random.rand(T, N, K)
    errfn = getEuclideanError
    WGradfn = multiplyConv2DWGrad
    HGradfn = multiplyConv2DHGrad
    if doKL:
        errfn = getKLError
        WGradfn = multiplyConv2DWGradKL
        HGradfn = multiplyConv2DHGradKL
    errs = [[errfn(A, multiplyConv2D(W1, H)), errfn(Ap, multiplyConv2D(W2, H))]]
    if plotfn:
        res=4
        plt.figure(figsize=((2+K)*res, 2*res))
        plotfn(A, Ap, W1, W2, H, 0, errs, foldername)
        plt.savefig("%s/NMF2DConvJoint_%i.png"%(foldername, 0), bbox_inches = 'tight')
    for l in range(L):
        print("Joint 2DNMF iteration %i of %i"%(l+1, L))
        #Step 1: Update Ws
        ALam = multiplyConv2D(W1, H)
        ApLam = multiplyConv2D(W2, H)
        W1 = W1*WGradfn(W1, H, A, ALam)
        W2 = W2*WGradfn(W2, H, Ap, ApLam)

        #Step 2: Update Hs
        ALam = multiplyConv2D(W1, H)
        ApLam = multiplyConv2D(W2, H)
        (HNums1, HDenoms1) = HGradfn(W1, H, A, ALam, doDivision = False)
        (HNums2, HDenoms2) = HGradfn(W2, H, Ap, ApLam, doDivision = False)
        H = H*((HNums1+HNums2)/(HDenoms1+HDenoms2))

        errs.append([errfn(A, multiplyConv2D(W1, H)), errfn(Ap, multiplyConv2D(W2, H))])
        if plotfn and ((l+1) == L or (l+1)%60 == 0):
            plt.clf()
            plotfn(A, Ap, W1, W2, H, l+1, errs, foldername)
            plt.savefig("%s/NMF2DConvJoint_%i.png"%(foldername, l+1), bbox_inches = 'tight')
    return (W1, W2, H)


def getJoint3WayError(A, Ap, B, W1, W2, H1, H2, errfn, mulfn):
    res = errfn(A, mulfn(W1, H1))
    res += errfn(Ap, mulfn(W2, H1))
    res += errfn(B, mulfn(W1, H2))
    return res

def doNMF2DConvJoint3Way(A, Ap, B, K, T, F, L, plotfn = None, foldername = ".", eps = 1e-20):
    """
    A version of 2DNMF that jointly solves for W1, H1, W2, and H2 that
    satisfy W1*H1 ~= A, W2*H1 ~= Ap, and W2*H1 ~= B
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
    W1 = np.random.rand(T, M, K)
    W2 = np.random.rand(T, M, K)
    H1 = np.random.rand(F, K, N1)
    H2 = np.random.rand(F, K, N2)    

    errs = [getJoint3WayError(A, Ap, B, W1, W2, H1, H2, getEuclideanError, multiplyConv2D)]
    if plotfn:
        res=4
        plt.figure(figsize=((8+2*K)*res*1.2, 2*res))
        plotfn(A, Ap, B, W1, W2, H1, H2, 0, errs, foldername)
        plt.savefig("%s/NMF2DConvJoint3Way%i.png"%(foldername, 0), bbox_inches = 'tight')
    for l in range(L):
        print("Joint 3Way 2DNMF iteration %i of %i"%(l+1, L))
        tic = time.time()
        #Step 1: Update Ws
        Lam11 = multiplyConv2D(W1, H1)
        Lam12 = multiplyConv2D(W1, H2)
        Lam21 = multiplyConv2D(W2, H1)

        W1Nums = np.zeros(W1.shape)
        W1Denoms = np.zeros(W1.shape)
        W2Nums = np.zeros(W2.shape)
        W2Denoms = np.zeros(W2.shape)
        for f in range(F):
            thisA = shiftMatLRUD(A, di=-f)
            thisAp = shiftMatLRUD(Ap, di=-f)
            thisB = shiftMatLRUD(B, di=-f)
            thisLam11 = shiftMatLRUD(Lam11, di=-f)
            thisLam12 = shiftMatLRUD(Lam12, di=-f)
            thisLam21 = shiftMatLRUD(Lam21, di=-f)
            for t in range(T):
                thisH1T = (shiftMatLRUD(H1[f, :, :], dj=t)).T
                thisH2T = (shiftMatLRUD(H2[f, :, :], dj=t)).T
                W1Nums[t, :, :] += thisA.dot(thisH1T) + thisB.dot(thisH2T)
                W1Denoms[t, :, :] += thisLam11.dot(thisH1T) + thisLam12.dot(thisH2T)
                W2Nums[t, :, :] += thisAp.dot(thisH1T)
                W2Denoms[t, :, :] += thisLam21.dot(thisH1T)
        W1Nums[W1Denoms < eps] = 1.0
        W2Nums[W2Denoms < eps] = 1.0
        W1Denoms[W1Denoms < eps] = 1.0
        W2Denoms[W2Denoms < eps] = 1.0
        W1 = W1*(W1Nums/W1Denoms)
        W2 = W2*(W2Nums/W2Denoms)

        #Step 2: Update Hs
        Lam11 = multiplyConv2D(W1, H1)
        Lam12 = multiplyConv2D(W1, H2)
        Lam21 = multiplyConv2D(W2, H1)
        H1Nums = np.zeros(H1.shape)
        H1Denoms = np.zeros(H1.shape)
        H2Nums = np.zeros(H2.shape)
        H2Denoms = np.zeros(H2.shape)
        for t in range(T):
            thisA = shiftMatLRUD(A, dj=-t)
            thisAp = shiftMatLRUD(Ap, dj=-t)
            thisB = shiftMatLRUD(B, dj=-t)
            thisLam11 = shiftMatLRUD(Lam11, dj=-t)
            thisLam12 = shiftMatLRUD(Lam12, dj=-t)
            thisLam21 = shiftMatLRUD(Lam21, dj=-t)
            for f in range(F):
                thisW1T = (shiftMatLRUD(W1[t, :, :], di=f)).T
                thisW2T = (shiftMatLRUD(W2[t, :, :], di=f)).T
                H1Nums[f, :, :] += thisW1T.dot(thisA) + thisW2T.dot(thisAp)
                H1Denoms[f, :, :] += thisW1T.dot(thisLam11) + thisW2T.dot(thisLam21)
                H2Nums[f, :, :] += thisW1T.dot(thisB)
                H2Denoms[f, :, :] += thisW1T.dot(thisLam12)
        H1Nums[H1Denoms < eps] = 1.0
        H2Nums[H2Denoms < eps] = 1.0
        H1Denoms[H1Denoms < eps] = 1.0
        H2Denoms[H2Denoms < eps] = 1.0
        H1 = H1*(H1Nums/H1Denoms)
        H2 = H2*(H2Nums/H2Denoms)
        print("Elapsed Time: %.3g"%(time.time()-tic))

        errs.append(getJoint3WayError(A, Ap, B, W1, W2, H1, H2, getEuclideanError, multiplyConv2D))
        if plotfn and ((l+1) == L or (l+1)%30 == 0):
            plt.clf()
            plotfn(A, Ap, B, W1, W2, H1, H2, l+1, errs, foldername)
            plt.savefig("%s/NMF2DConvJoint3Way%i.png"%(foldername, l+1), bbox_inches = 'tight')
    return (W1, W2, H1, H2)



def plotNMF1DConvSpectraJoint(V, W, H, iter, errs, hopLength = -1, plotComponents = True, \
        audioParams = None):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H, with the understanding
    that V and W are two songs concatenated on top of each other
    :param V: An N x M target
    :param W: An T x N x K source/corpus matrix
    :returns H: A KxM matrix of source activations for each column of V
    :param iter: The iteration number
    :param errs: Errors over time
    :param hopLength: The hop length (for plotting)
    """
    import librosa
    import librosa.display
    K = W.shape[2]
    if not plotComponents:
        K = 0

    N = int(W.shape[1]/2)
    W1 = W[:, 0:N, :]
    W2 = W[:, N::, :]
    V1 = V[0:N, :]
    V2 = V[N::, :]
    WH = multiplyConv1D(W, H)

    if audioParams:
        from SpectrogramTools import griffinLimInverse
        from scipy.io import wavfile
        import os
        [Fs, prefix] = [audioParams['Fs'], audioParams['prefix']]
        winSize = audioParams['winSize']
        pre = "%sNMF1DJointIter%i"%(prefix, iter)
        if not os.path.exists(pre):
            os.mkdir(pre)
        #Invert each Wt
        for k in range(W1.shape[2]):
            y_hat = griffinLimInverse(W1[:, :, k].T, winSize, hopLength)
            y_hat = y_hat/np.max(np.abs(y_hat))
            wavfile.write("%s/W1_%i.wav"%(pre, k), Fs, y_hat)
            y_hat = griffinLimInverse(W2[:, :, k].T, winSize, hopLength)
            y_hat = y_hat/np.max(np.abs(y_hat))
            wavfile.write("%s/W2_%i.wav"%(pre, k), Fs, y_hat)
        #Invert the audio
        y_hat = griffinLimInverse(WH[0:N, :], winSize, hopLength)
        y_hat = y_hat/np.max(np.abs(y_hat))
        wavfile.write("%s/WH1.wav"%pre, Fs, y_hat)
        y_hat = griffinLimInverse(WH[N::, :], winSize, hopLength)
        y_hat = y_hat/np.max(np.abs(y_hat))
        wavfile.write("%s/WH2.wav"%pre, Fs, y_hat)

    plt.subplot(3, 2+K, 1)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(V1), hop_length = hopLength, \
                                    y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(V1, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.title("V1")
    plt.subplot(3, 2+K, 2)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(V2), hop_length = hopLength, \
                                    y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(V2, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.title("V2")
    plt.subplot(3, 2+K, 2+K+1)
    
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(WH[0:N, :]), hop_length = hopLength,\
                                 y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(WH[0:N, :], cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.title("W1*H")
    errs = np.array(errs)
    if errs.shape[0] > 1:
        errs = errs[1::, :]
    plt.subplot(3, 2+K, (2+K)*2+1)
    plt.semilogy(errs[:, 0])
    plt.ylim([0.9*np.min(errs[:, 0]), np.max(errs[:, 0])*1.1])
    plt.title("Errors 1")
    plt.subplot(3, 2+K, (2+K)*2+2)
    plt.semilogy(errs[:, 1])
    plt.ylim([0.9*np.min(errs[:, 1]), np.max(errs[:, 1])*1.1])
    plt.title("Errors 2")

    plt.subplot(3, 2+K, 2+K+2)
    WH = multiplyConv1D(W, H)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(WH[N::, :]), hop_length = hopLength,\
                                 y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(WH[N::, :], cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.title("W1*H")

    if plotComponents:
        for k in range(K):
            plt.subplot(3, 2+K, 3+k)
            if hopLength > -1:
                librosa.display.specshow(librosa.amplitude_to_db(W1[:, :, k].T), \
                    hop_length=hopLength, y_axis='log', x_axis='time')
            else:
                plt.imshow(W1[:, :, k].T, cmap = 'afmhot', \
                        interpolation = 'nearest', aspect = 'auto')  
                plt.colorbar()
            plt.title("W1_%i"%k)
            plt.subplot(3, 2+K, 2+K+3+k)
            if hopLength > -1:
                librosa.display.specshow(librosa.amplitude_to_db(W2[:, :, k].T), \
                    hop_length=hopLength, y_axis='log', x_axis='time')
            else:
                plt.imshow(W2[:, :, k].T, cmap = 'afmhot', \
                        interpolation = 'nearest', aspect = 'auto')  
                plt.colorbar()
            plt.title("W2_%i"%k)
            plt.subplot(3, 2+K, (2+K)*2+3+k)
            plt.plot(H[k, :])
            plt.title("H%i"%k)

def plotNMF2DConvSpectraJoint(A, Ap, W1, W2, H, iter, errs, foldername, \
                                hopLength = -1, plotComponents = True, audioParams = None):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H, with the understanding
    that V and W are two songs concatenated on top of each other
    :param V: An N x M target
    :param W: An T x N x K source/corpus matrix
    :returns H: An F x K x M matrix of source activations
    :param iter: The iteration number
    :param errs: Errors over time
    :param hopLength: The hop length (for plotting)
    """
    import librosa
    import librosa.display
    import scipy.ndimage
    import os
    K = W1.shape[2]
    if not plotComponents:
        K = 0

    N = A.shape[0]
    T = W1.shape[0]
    W1H = multiplyConv2D(W1, H)
    W2H = multiplyConv2D(W2, H)

    if audioParams:
        from CQT import getiCQTGriffinLimNakamuraMatlab, getTemplateNakamura
        from scipy.io import wavfile
        
        [Fs, eng, XSizes] = [audioParams['Fs'], audioParams['eng'], audioParams['XSizes']]
        ZoomFac = audioParams['ZoomFac']
        bins_per_octave = audioParams['bins_per_octave']
        #Invert the audio
        if bins_per_octave > -1:
            #Step 1: Invert the approximations
            for (s1, Lam) in zip(["A", "Ap"], [W1H, W2H]):
                print("%s.size = %i"%(s1, XSizes[s1]))
                LamZoom = scipy.ndimage.interpolation.zoom(Lam, (1, ZoomFac))
                (y_hat, spec) = getiCQTGriffinLimNakamuraMatlab(eng, LamZoom, XSizes[s1], Fs, \
                    bins_per_octave, NIters=100, randPhase = True)
                y_hat = y_hat/np.max(np.abs(y_hat))
                sio.wavfile.write("%s/%s.wav"%(foldername, s1), Fs, y_hat)
            #Step 2: Invert the templates
            for (s1, thisW) in zip(["A", "Ap"], (W1, W2)):
                for k in range(K):
                    y_hat = getTemplateNakamura(eng, thisW[:, :, k].T, A.shape, \
                                        ZoomFac, bins_per_octave, XSizes[s1], Fs)
                    y_hat = y_hat/np.max(np.abs(y_hat))
                    sio.wavfile.write("%s/%s_W%i.wav"%(foldername, s1, k), Fs, y_hat)

    plt.subplot(3, 2+K, 1)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(A), hop_length = hopLength, \
                                    y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(A, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.title("A")
    plt.subplot(3, 2+K, 2)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(Ap), hop_length = hopLength, \
                                    y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(Ap, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.title("Ap")
    plt.subplot(3, 2+K, 2+K+1)
    
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(W1H), hop_length = hopLength,\
                                 y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(W1H, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.title("W1*H")
    errs = np.array(errs)
    if errs.shape[0] > 1:
        errs = errs[1::, :]
    plt.subplot(3, 2+K, (2+K)*2+1)
    plt.semilogy(errs[:, 0])
    plt.ylim([0.9*np.min(errs[:, 0]), np.max(errs[:, 0])*1.1])
    plt.title("Errors 1")
    plt.subplot(3, 2+K, (2+K)*2+2)
    plt.semilogy(errs[:, 1])
    plt.ylim([0.9*np.min(errs[:, 1]), np.max(errs[:, 1])*1.1])
    plt.title("Errors 2")

    plt.subplot(3, 2+K, 2+K+2)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(W2H), hop_length = hopLength,\
                                 y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(W2H, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.title("W2*H")

    if plotComponents:
        for k in range(K):
            plt.subplot(3, 2+K, 3+k)
            if hopLength > -1:
                librosa.display.specshow(librosa.amplitude_to_db(W1[:, :, k].T), \
                    hop_length=hopLength, y_axis='log', x_axis='time')
            else:
                plt.imshow(W1[:, :, k].T, cmap = 'afmhot', \
                        interpolation = 'nearest', aspect = 'auto')  
                plt.colorbar()
            plt.title("W1_%i"%k)
            plt.subplot(3, 2+K, 2+K+3+k)
            if hopLength > -1:
                librosa.display.specshow(librosa.amplitude_to_db(W2[:, :, k].T), \
                    hop_length=hopLength, y_axis='log', x_axis='time')
            else:
                plt.imshow(W2[:, :, k].T, cmap = 'afmhot', \
                        interpolation = 'nearest', aspect = 'auto')  
                plt.colorbar()
            plt.title("W2_%i"%k)
            plt.subplot(3, 2+K, (2+K)*2+3+k)
            plt.imshow(H[:, k, :], cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
            if hopLength > -1:
                plt.gca().invert_yaxis()
            plt.title("H%i"%k)


def plotNMF2DConvSpectraJoint3Way(A, Ap, B, W1, W2, H1, H2, iter, errs, foldername, \
        hopLength = -1, audioParams = None, plotElems = True, useGPU = False):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H
    :param A: An M x N1 matrix for song A
    :param Ap: An M x N1 matrix for song A'
    :param B: An M x N2 matrix for song B
    :param W1: An T x M x K source/corpus matrix for songs A and B
    :param W2: An T x M x K source/corpus matrix for song A'
    :param H1: An F x K x N1 matrix of activations for A and A'
    :param H2: An F x K x N2 matrix of activations for B
    :param iter: The iteration number
    :param errs: Errors over time
    :param hopLength: The hop length (for plotting)
    :param audioParams: Parameters for inverting CQT
    """
    import librosa
    import librosa.display
    import scipy.ndimage
    K = W1.shape[2]
    if not plotElems:
        K = 0

    if audioParams:
        from CQT import getCQTNakamuraMatlab, getiCQTGriffinLimNakamuraMatlab
        from CQT import getTemplateNakamura
        from scipy.io import wavfile
        Fs = audioParams['Fs']
        [eng, XSizes] = [audioParams['eng'], audioParams['XSizes']]
        ZoomFac = audioParams['ZoomFac']
        bins_per_octave = -1
        winSize = -1
        if 'bins_per_octave' in audioParams:
            bins_per_octave = audioParams['bins_per_octave']
        if 'winSize' in audioParams:
            winSize = audioParams['winSize']
        #Invert each Wt
        for (s1, thisW) in zip(["A", "Ap"], (W1, W2)):
            for k in range(K):
                y_hat = getTemplateNakamura(eng, thisW[:, :, k].T, A.shape, \
                                    ZoomFac, bins_per_octave, XSizes[s1], Fs)
                y_hat = y_hat/np.max(np.abs(y_hat))
                sio.wavfile.write("%s/%s_W%i.wav"%(foldername, s1, k), Fs, y_hat)
    from NMFGPU import multiplyConv2DGPU
    import pycuda.gpuarray as gpuarray
    if useGPU:
        W1G = gpuarray.to_gpu(W1)
        W2G = gpuarray.to_gpu(W2)
        H1G = gpuarray.to_gpu(H1)
        H2G = gpuarray.to_gpu(H2)
        Lam11 = multiplyConv2DGPU(W1G, H1G).get()
        Lam12 = multiplyConv2DGPU(W1G, H2G).get()
        Lam21 = multiplyConv2DGPU(W2G, H1G).get()
        Lam22 = multiplyConv2DGPU(W2G, H2G).get()
    else:
        Lam11 = multiplyConv2D(W1, H1)
        Lam12 = multiplyConv2D(W1, H2)
        Lam21 = multiplyConv2D(W2, H1)
        Lam22 = multiplyConv2D(W2, H2)
    sio.savemat("%s/Iter%i.mat"%(foldername, iter), {"W1":W1, "W2":W2, "H1":H1, "H2":H2,\
        "Lam11":Lam11, "Lam12":Lam12, "Lam21":Lam21, "Lam22":Lam22})
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
            if bins_per_octave > -1:
                print("%s.size = %i"%(s1, XSizes[s1]))
                LamZoom = scipy.ndimage.interpolation.zoom(Lam, (1, ZoomFac))
                (y_hat, spec) = getiCQTGriffinLimNakamuraMatlab(eng, LamZoom, XSizes[s1], Fs, \
                    bins_per_octave, NIters=100, randPhase = True)
                y_hat = y_hat/np.max(np.abs(y_hat))
                sio.wavfile.write("%s/%s.wav"%(foldername, s1), Fs, y_hat)
    plt.subplot(2, 8+2*K, 4)
    errs = np.array(errs)
    if len(errs) > 1:
        errs = errs[1::]
    plt.semilogy(errs)
    plt.title("Errors")

    for k in range(K):
        plt.subplot(2, 8+2*K, 4+k+1)
        if hopLength > -1:
            librosa.display.specshow(librosa.amplitude_to_db(W1[:, :, k].T), \
                hop_length=hopLength, y_axis='log', x_axis='time')
        else:
            plt.imshow(W1[:, :, k].T, cmap = 'afmhot', \
                    interpolation = 'nearest', aspect = 'auto')  
            plt.colorbar()
        plt.title("W1%i"%k)

        plt.subplot(2, 8+2*K, 8+2*K+4+k+1)
        if hopLength > -1:
            librosa.display.specshow(librosa.amplitude_to_db(W2[:, :, k].T), \
                hop_length=hopLength, y_axis='log', x_axis='time')
        else:
            plt.imshow(W2[:, :, k].T, cmap = 'afmhot', \
                    interpolation = 'nearest', aspect = 'auto')  
            plt.colorbar()
        plt.title("W2%i"%k)

        plt.subplot(2, 8+2*K, K+4+k+1)
        plt.imshow(H1[:, k, :], cmap = 'afmhot', \
            interpolation = 'nearest', aspect = 'auto')
        if hopLength > -1:
            plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("H1%i"%k)

        plt.subplot(2, 8+2*K, 8+2*K+4+K+k+1)
        plt.imshow(H2[:, k, :], cmap = 'afmhot', \
            interpolation = 'nearest', aspect = 'auto')
        if hopLength > -1:
            plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("H2%i"%k)