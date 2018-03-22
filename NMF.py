import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

def getKLError(V, WH, eps = 1e-10):
    """
    Return the Kullback-Liebler diverges between V and W*H
    """
    denom = np.array(WH)
    denom[denom == 0] = 1
    arg = V/denom
    arg[arg < eps] = eps
    return np.sum(V*np.log(arg)-V+WH)

def getEuclideanError(V, WH):
    """
    Return the Frobenius norm between V and W*H
    """
    return np.sum((V-WH)**2)

def plotNMFSpectra(V, W, H, iter, errs, hopLength = -1):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H
    :param V: An N x M target
    :param W: An N x K source/corpus matrix
    :returns H: A KxM matrix of source activations for each column of V
    :param iter: The iteration number
    :param errs: Convergence errors
    :param hopLength: The hop length (for plotting)
    """
    import librosa
    import librosa.display
    plt.subplot(151)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(V), hop_length = hopLength, \
                                y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(V, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("V")
    plt.subplot(152)
    WH = W.dot(H)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(WH), hop_length = hopLength, \
                                    y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(WH, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("W*H Iteration %i"%iter)  
    plt.subplot(153)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(W), hop_length = hopLength, \
                                    y_axis = 'log', x_axis = 'time')        
    else:
        plt.imshow(W, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("W")
    plt.subplot(154)
    plt.imshow(np.log(H + np.min(H[H > 0])), cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("H Iteration %i"%iter)
    plt.subplot(155)
    plt.semilogy(np.array(errs[1::]))
    plt.title("KL Errors")
    plt.xlabel("Iteration")             

def doNMF(V, K, L, W = np.array([]), plotfn = None):
    N = V.shape[0]
    M = V.shape[1]
    WFixed = False
    if W.size > 0:
        WFixed = True
        K = W.shape[1]
        print("W.shape = ", W.shape)
    else:
        W = np.random.rand(N, K)
    tic = time.time()
    H = np.random.rand(K, M)
    print("Time elapsed H initializing: %.3g"%(time.time() - tic))

    errs = [getKLError(V, W.dot(H))]
    if plotfn:
        res=4
        plt.figure(figsize=(res*5, res))
        plotfn(V, W, H, 0, errs) 
        plt.savefig("NMF_%i.png"%0, bbox_inches = 'tight')
    for l in range(L):
        print("NMF iteration %i of %i"%(l+1, L))            
        #KL Divergence Version
        tic = time.time()
        VLam = V/(W.dot(H))
        print("VLam Elapsed Time: %.3g"%(time.time() - tic))
        tic = time.time()
        H *= (W.T).dot(VLam)/np.sum(W, 0)[:, None]
        print("Elapsed Time H Update %.3g"%(time.time() - tic))
        if not WFixed:
            VLam = V/(W.dot(H))
            W *= (VLam.dot(H.T))/np.sum(H, 1)[None, :]
        errs.append(getKLError(V, W.dot(H)))
        if plotfn and ((l+1)==L):# or (l+1)%10 == 0):
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            plt.savefig("NMF_%i.png"%(l+1), bbox_inches = 'tight')
    return (W, H)

def doNMFDriedger(V, W, L, r = 7, p = 10, c = 3, plotfn = None):
    """
    Implement the technique from "Let It Bee-Towards NMF-Inspired
    Audio Mosaicing"
    :param V: M x N target matrix
    :param W: An M x K matrix of template sounds in some time order\
        along the second axis
    :param L: Number of iterations
    :param r: Width of the repeated activation filter
    :param p: Degree of polyphony; i.e. number of values in each column\
        of H which should be un-shrunken
    :param c: Half length of time-continuous activation filter
    """
    import scipy.ndimage
    N = V.shape[1]
    K = W.shape[1]
    tic = time.time()
    H = np.random.rand(K, N)
    print("H.shape = ", H.shape)
    print("Time elapsed H initializing: %.3g"%(time.time() - tic))
    errs = [getKLError(V, W.dot(H))]
    if plotfn:
        res=4
        plt.figure(figsize=(res*5, res))
    for l in range(L):
        print("NMF Driedger iteration %i of %i"%(l+1, L))   
        iterfac = 1-float(l+1)/L       
        tic = time.time()
        #Step 1: Avoid repeated activations
        print("Doing Repeated Activations...")
        MuH = scipy.ndimage.filters.maximum_filter(H, size=(1, r))
        H[H<MuH] = H[H<MuH]*iterfac
        #Step 2: Restrict number of simultaneous activations
        print("Restricting simultaneous activations...")
        #Use partitions instead of sorting for speed
        colCutoff = -np.partition(-H, p, 0)[p, :] 
        H[H < colCutoff[None, :]] = H[H < colCutoff[None, :]]*iterfac
        #Step 3: Supporting time-continuous activations
        if c > 0:                    
            print("Supporting time-continuous activations...")
            di = K-1
            dj = 0
            for k in range(-H.shape[0]+1, H.shape[1]):
                z = np.cumsum(np.concatenate((np.zeros(c), np.diag(H, k), np.zeros(c))))
                x2 = z[2*c::] - z[0:-2*c]
                H[di+np.arange(len(x2)), dj+np.arange(len(x2))] = x2
                if di == 0:
                    dj += 1
                else:
                    di -= 1
        #KL Divergence Version
        WH = W.dot(H)
        WH[WH == 0] = 1
        VLam = V/WH
        WDenom = np.sum(W, 0)
        WDenom[WDenom == 0] = 1
        H = H*((W.T).dot(VLam)/WDenom[:, None])
        print("Elapsed Time H Update %.3g"%(time.time() - tic))
        errs.append(getKLError(V, W.dot(H)))
        if plotfn and ((l+1)==L):# or (l+1)%20 == 0):
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            plt.savefig("NMDriedger_%i.png"%(l+1), bbox_inches = 'tight')
    return H

def doNMFDriedgerCTypes(V, W, L, r = 7, p = 10, c = 3, plotfn = None):
    """
    Implement the technique from "Let It Bee-Towards NMF-Inspired
    Audio Mosaicing," with KL divergence updates, using a C-Types 
    Python wrapper
    :param V: M x N target matrix
    :param W: An M x K matrix of template sounds in some time order\
        along the second axis
    :param L: Number of iterations
    :param r: Width of the repeated activation filter
    :param p: Degree of polyphony; i.e. number of values in each column\
        of H which should be un-shrunken
    :param c: Half length of time-continuous activation filter
    """
    import scipy.ndimage
    from _Driedger import DriedgerUpdates
    M = V.shape[0]
    N = V.shape[1]
    K = W.shape[1]
    tic = time.time()
    H = np.random.rand(K, N)
    print("H.shape = ", H.shape)
    print("Time elapsed H initializing: %.3g"%(time.time() - tic))
    errs = [getKLError(V, W.dot(H))]
    if plotfn:
        res=4
        plt.figure(figsize=(res*5, res))
        plotfn(V, W, H, 0, errs) 
        plt.savefig("NMFDriedger_%i.png"%0, bbox_inches = 'tight')

    for l in range(L):
        print("NMF Driedger iteration %i of %i"%(l+1, L))   
        iterfac = 1-float(l+1)/L
        #First 3 parts of H update
        ticouter = time.time()
        DriedgerUpdates(H, r, p, c, float(iterfac))
        tic = time.time()
        WH = W.dot(H)
        WH[WH == 0] = 1
        VLam = V/WH
        print("VLam Elapsed Time: %.3g"%(time.time() - tic))
        tic = time.time()
        WDenom = np.sum(W, 0)
        WDenom[WDenom == 0] = 1
        H = H*((W.T).dot(VLam)/WDenom[:, None])
        print("Elapsed Time Total H Update %.3g"%(time.time() - ticouter))
        errs.append(getKLError(V, W.dot(H)))
        if plotfn and ((l+1)==L):# or (l+1)%20 == 0):
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            plt.savefig("NMDriedger_%i.png"%(l+1), bbox_inches = 'tight')
    return H

def shiftMatLRUD(X, di=0, dj=0):
    XRet = np.array(X, dtype=X.dtype)
    #Shift up/down first
    if di > 0:
        XRet[di::, :] = X[0:-di, :]
        XRet[0:di, :] = 0
    elif di < 0:
        XRet[0:di, :] = X[-di::, :]
        XRet[di::, :] = 0
    #Now shift left/right
    if dj > 0:
        XRet[:, dj::] = XRet[:, 0:-dj]
        XRet[:, 0:dj] = 0
    elif dj < 0:
        XRet[:, 0:dj] = XRet[:, -dj::]
        XRet[:, dj::] = 0
    return XRet
    
def multiplyConv1D(W, H):
    """
    Perform a convolutional matrix multiplication in time
    """
    Lam = np.zeros((W.shape[1], H.shape[1]), dtype=W.dtype)
    for t in range(W.shape[0]):
        Lam += W[t, :, :].dot(shiftMatLRUD(H, dj=t))
    return Lam

def multiplyConv2D(W, H):
    """
    Perform a convolutional matrix multiplication in time and frequency
    """
    Lam = np.zeros((W.shape[1], H.shape[2]), dtype=W.dtype)
    for t in range(W.shape[0]):
        for f in range(H.shape[0]):
            Wt = np.array(W[t, :, :])
            Wt = shiftMatLRUD(Wt, di=f)
            Hf = np.array(H[f, :, :])
            Hf = shiftMatLRUD(Hf, dj=t)
            Lam += Wt.dot(Hf)
    return Lam

def multiplyConv2DWGrad(W, H, V, VLam, doDivision = True):
    """
    Compute the 2D convolutional multiplicative update for W
    under the Euclidean metric
    :param W: A TxNxK matrix of K sources over spatiotemporal spans NxT\
    :param H: A FxKxM matrix of source activations for each submatrix of W\
            over F transpositions over M time
    :param VLam: Convolutional WH multiplication
    :param doDivision: If true, return the factor Numerator/Denomenator\
        otherwise, return (Numerator, Denomenator)
    :returns Ratio: A TxNkX matrix of multiplicative updates for W\
        or (RatioNum, RatioDenom) if doDivision = False
    """
    WNums = np.zeros(W.shape) #Numerator
    WDenoms = np.zeros(W.shape) #Denomenator
    for f in range(H.shape[0]):
        thisV = shiftMatLRUD(V, di=-f)
        thisVLam = shiftMatLRUD(VLam, di=-f)
        for t in range(W.shape[0]):
            thisH = shiftMatLRUD(H[f, :, :], dj=t)
            WNums[t, :, :] += thisV.dot(thisH.T)
            WDenoms[t, :, :] += thisVLam.dot(thisH.T)
    if doDivision:
        return WNums/WDenoms
    else:
        return(WNums, WDenoms)

def multiplyConv2DHGrad(W, H, V, VLam, doDivision = True):
    """
    Compute the 2D convolutional multiplicative update for H
    under the Euclidean metric
    :param W: A TxNxK matrix of K sources over spatiotemporal spans NxT\
    :param H: A FxKxM matrix of source activations for each submatrix of W\
            over F transpositions over M time
    :param VLam: Convolutional WH multiplication
    :param doDivision: If true, return the factor Numerator/Denomenator\
        otherwise, return (Numerator, Denomenator)
    :returns Ratio: A FxKxM matrix of multiplicative updates for H \
        or (RatioNum, RatioDenom) if doDivision = False
    """
    HNums = np.zeros(H.shape) #Numerator
    HDenoms = np.zeros(H.shape) #Denomenator
    for t in range(W.shape[0]):
        thisV = shiftMatLRUD(V, dj=-t)
        thisVLam = shiftMatLRUD(VLam, dj=-t)
        for f in range(H.shape[0]):
            thisW = shiftMatLRUD(W[t, :, :], di=f)
            HNums[f, :, :] += (thisW.T).dot(thisV)
            HDenoms[f, :, :] += (thisW.T).dot(thisVLam)
    if doDivision:
        return HNums/HDenoms
    else:
        return (HNums, HDenoms)

def multiplyConv2DWGradKL(W, H, V, VLam, doDivision = True):
    """
    Compute the 2D convolutional multiplicative update for W under the
    Kullback-Liebler divergence
    :param W: A TxNxK matrix of K sources over spatiotemporal spans NxT\
    :param H: A FxKxM matrix of source activations for each submatrix of W\
            over F transpositions over M time
    :param VLam: Convolutional WH multiplication
    :param doDivision: If true, return the factor Numerator/Denomenator\
        otherwise, return (Numerator, Denomenator)
    :returns Ratio: A TxNkX matrix of multiplicative updates for W\
        or (RatioNum, RatioDenom) if doDivision = False
    """
    WNums = np.zeros(W.shape) #Numerator
    WDenoms = np.zeros((W.shape[0], W.shape[2])) #Denomenator
    VLam = VLam.copy()
    VLam[VLam == 0] = 1
    VLamQuot = V/VLam
    for f in range(H.shape[0]):
        thisVLamQuot = shiftMatLRUD(VLamQuot, di=-f)
        for t in range(W.shape[0]):
            thisH = shiftMatLRUD(H[f, :, :], dj=t)
            WNums[t, :, :] += thisVLamQuot.dot(thisH.T)
            WDenoms[t, :] += np.sum(thisH, 1)
    if doDivision:
        return WNums/WDenoms[:, None, :]
    else:
        return (WNums, WDenoms[:, None, :])

def multiplyConv2DHGradKL(W, H, V, VLam, doDivision = True):
    """
    Compute the 2D convolutional multiplicative update for H
    under the Kullback-Liebler divergence
    :param W: A TxNxK matrix of K sources over spatiotemporal spans NxT\
    :param H: A FxKxM matrix of source activations for each submatrix of W\
            over F transpositions over M time
    :param VLam: Convolutional WH multiplication
    :param doDivision: If true, return the factor Numerator/Denomenator\
        otherwise, return (Numerator, Denomenator)
    :returns Ratio: A FxKxM matrix of multiplicative updates for H\
        or (RatioNum, RatioDenom) if doDivision = False
    """
    HNums = np.zeros(H.shape) #Numerator
    HDenoms = np.zeros((H.shape[0], H.shape[1])) #Denomenator
    VLam = VLam.copy()
    VLam[VLam == 0] = 1
    VLamQuot = V/VLam
    for t in range(W.shape[0]):
        thisVLamQuot = shiftMatLRUD(VLamQuot, dj=-t)
        for f in range(H.shape[0]):
            thisW = shiftMatLRUD(W[t, :, :], di=f)
            HNums[f, :, :] += (thisW.T).dot(thisVLamQuot)
            HDenoms[f, :] += np.sum(thisW, 0)
    if doDivision:
        return HNums/HDenoms[:, :, None]
    else:
        return (HNums, HDenoms[:, :, None])

def doNMF1DConv(V, K, T, L, r = 0, p = -1, W = np.array([]), plotfn = None, plotComponents = True):
    """
    Implementing the technique described in 
    "Non-negative Matrix Factor Deconvolution; Extraction of
        Multiple Sound Sources from Monophonic Inputs"
    NOTE: Using update rules from 2DNMF instead of averaging
    :param V: An N x M target matrix
    :param K: Number of latent factors
    :param T: Time extent of W matrices
    :param L: Number of iterations
    :param r: Width of the repeated activation filter
    :param p: Degree of polyphony
    :param plotfn: A function used to plot each iteration, which should\
        take the arguments (V, W, H, iter)
    :param joint: If true, it's understood that V and W are two songs concatenated\
        on top of each other.  Functionality is the same, but errors are computed\
        separately, and plotting is done differently
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
    errs = [getKLError(V, WH)]
    if plotfn:
        res=4
        pK = K
        if not plotComponents:
            pK = 0
        plt.figure(figsize=((4+pK)*res, res))
        plotfn(V, W, H, 0, errs) 
        plt.savefig("NMF1DConv_0.png", bbox_inches = 'tight')
    for l in range(L):
        #KL Divergence Version
        print("NMF iteration %i of %i"%(l+1, L))            
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
        errs.append(getKLError(V, WH))
        if plotfn and ((l+1) == L):# or (l+1)%40 == 0):
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            plt.savefig("NMF1DConv_%i.png"%(l+1), bbox_inches = 'tight')
    return (W, H)

def doNMF2DConv(V, K, T, F, L, W = np.array([]), doKL = False, plotfn = None):
    """
    Implementing the Euclidean 2D NMF technique described in 
    "Nonnegative Matrix Factor 2-D Deconvolution
        for Blind Single Channel Source Separation"
    :param V: An N x M target matrix
    :param K: Number of latent factors
    :param T: Time extent of W matrices
    :param F: Frequency extent of H matrices
    :param L: Number of iterations
    :param doKL: Whether to do Kullback-Leibler divergence.  If false, do Euclidean
    :param plotfn: A function used to plot each iteration, which should\
        take the arguments (V, W, H, iter)
    :returns (W, H): \
        W is an TxNxK matrix of K sources over spatiotemporal spans NxT\
        H is a FxKxM matrix of source activations for each submatrix of W\
            over F transpositions over M time
    """
    N = V.shape[0]
    M = V.shape[1]
    WFixed = False
    if W.size == 0:
        W = np.random.rand(T, N, K)
    else:
        WFixed = True
    H = np.random.rand(F, K, M)
    errfn = getEuclideanError
    WGradfn = multiplyConv2DWGrad
    HGradfn = multiplyConv2DHGrad
    if doKL:
        errfn = getKLError
        WGradfn = multiplyConv2DWGradKL
        HGradfn = multiplyConv2DHGradKL
    errs = [errfn(V, multiplyConv2D(W, H))]
    if plotfn:
        res=4
        plt.figure(figsize=((2+K)*res, 2*res))
        plotfn(V, W, H, 0, errs) 
        plt.savefig("NMF2DConv_%i.png"%0, bbox_inches = 'tight')
    for l in range(L):
        print("NMF iteration %i of %i"%(l+1, L))
        #Step 1: Update Ws
        if not WFixed:
            VLam = multiplyConv2D(W, H)
            W = W*WGradfn(W, H, V, VLam)

        #Step 2: Update Hs
        VLam = multiplyConv2D(W, H)
        H = H*HGradfn(W, H, V, VLam)

        errs.append(errfn(V, multiplyConv2D(W, H)))
        if plotfn and ((l+1) == L or (l+1)%60 == 0):
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            plt.savefig("NMF2DConv_%i.png"%(l+1), bbox_inches = 'tight')
    return (W, H)


def getComplexNMF1DTemplates(S, W, H, p = 2, audioParams = None):
    """
    Given a complex spectrogram and a factorization WH ~= |S| of its magnitude
    spectrum, separate out the complex spectrogram into each of its components
    :param S: An MxN complex spectrogram
    :param W: An TxMxK matrix of K sources over frequencytemporal spans MxT
    :param H: A KxN matrix of source activations for each column of |S|
    :param p: Power for Weiner filter in soft mask matrices
    :param audioParams: {'winSize':int, 'hopSize':int, 'Fs':int, 'fileprefix':string}\
        If specified, save each component to disk as a wav file
    """
    K = W.shape[2]
    #Step 1: Compute the masked matrices raised to the power p
    AsSum = np.zeros(S.shape)
    As = []
    for k in range(K):
        Hk = np.array(H)
        Hk[0:k, :] = 0
        Hk[k+1::, :] = 0
        As.append(multiplyConv1D(W, Hk)**p)
        AsSum += As[-1]
    #Step 2: Average masked portions of the spectrogram to come up with
    #complex-valued templates
    Ss = []
    Ratios = []
    AllPow = np.abs(np.sum(S*np.conj(S), 0))
    for k in range(K):
        Ss.append(S*As[k]/AsSum)
        Pow = np.abs(np.sum(Ss[k]*np.conj(Ss[k]), 0))
        Ratios.append(Pow/AllPow)
    #Step 4: Save components if user requested
    if audioParams:
        from SpectrogramTools import iSTFT
        [winSize, hopSize] = [audioParams['winSize'], audioParams['hopSize']]
        [Fs, fileprefix] = [audioParams['Fs'], audioParams['fileprefix']]
        import matplotlib.pyplot as plt
        from scipy.io import wavfile
        X = np.array([])
        for k in range(K):
            thisS = np.array(Ss[k])
            thisS[:, Ratios[k] < 0.05] = 0
            Xk = iSTFT(thisS, winSize, hopSize)
            if k == 0:
                X = Xk
            else:
                X += Xk
            wavfile.write("%s_%i.wav"%(fileprefix, k), Fs, Xk)
            plt.clf()
            plt.plot(Ratios[k])
            plt.title("Ratio, %.3g Above 0.05"%(np.sum(Ratios[k] > 0.05)/float(Ratios[k].size)))
            plt.savefig("%s_%iPower.svg"%(fileprefix, k), bbox_inches = 'tight')
        wavfile.write("%sNMF.wav"%fileprefix, Fs, X)
    return (Ss, Ratios)


def getComplexNMF2DTemplates(C, W, H, ZoomFac, p = 2):
    """
    Given a complex CQT spectrogram and a factorization WH ~= |S| of its magnitude
    spectrum, separate out the complex spectrogram into each of its components
    :param C: An MxN complex spectrogram
    :param W: An TxMxK matrix of K sources over frequencytemporal spans MxT
    :param H: A FxKxN matrix of frequency shifted source activations
    :param ZoomFac: Factor by which the CQT spectrogram has been downsampled
    :param p: Power for Weiner filter in soft mask matrices
    """
    import scipy.ndimage
    K = W.shape[2]
    #Step 1: Compute the masked matrices raised to the power p
    AsSum = np.zeros(C.shape)
    As = []
    for k in range(K):
        Hk = np.array(H)
        Hk[:, 0:k, :] = 0
        Hk[:, k+1::, :] = 0
        Ck = multiplyConv2D(W, Hk)
        Ck = scipy.ndimage.zoom(Ck, (1, ZoomFac))**p
        As.append(Ck)
        AsSum += As[-1]
    #Step 2: Average masked portions of the CQT to come up with
    #complex-valued templates
    Cs = []
    Ratios = []
    AllPow = np.abs(np.sum(C*np.conj(C), 0))
    for k in range(K):
        Cs.append(C*As[k]/AsSum)
        Pow = np.abs(np.sum(Cs[k]*np.conj(Cs[k]), 0))
        Ratios.append(Pow/AllPow)
    return (Cs, Ratios)


def plotNMF1DConvSpectra(V, W, H, iter, errs, hopLength = -1, plotComponents = True):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H
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
    plt.subplot(1, 4+K, 1)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(V), hop_length = hopLength, \
                                    y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(V, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.title("V")
    plt.subplot(1, 4+K, 3+K)
    plt.imshow(H, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')  
    plt.colorbar()
    plt.title("H")
    if plotComponents:
        for k in range(K):
            plt.subplot(1, 4+K, 3+k)
            if hopLength > -1:
                librosa.display.specshow(librosa.amplitude_to_db(W[:, :, k].T), \
                    hop_length=hopLength, y_axis='log', x_axis='time')
            else:
                plt.imshow(W[:, :, k].T, cmap = 'afmhot', \
                        interpolation = 'nearest', aspect = 'auto')  
                plt.colorbar()
            plt.title("W%i"%k)
    plt.subplot(1, 4+K, 2)
    WH = multiplyConv1D(W, H)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(WH), hop_length = hopLength,\
                                 y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(WH, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.subplot(1, 4+K, 4+K)
    errs = np.array(errs)
    if len(errs) > 1:
        errs = errs[1::]
    plt.semilogy(errs)
    plt.ylim([0.9*np.min(errs), np.max(errs)*1.1])
    plt.title("Errors")

def plotNMF2DConvSpectra(V, W, H, iter, errs, hopLength = -1):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H
    :param V: An N x M target
    :param W: An T x N x K source/corpus matrix
    :returns H: An F x K x M matrix of source activations
    :param iter: The iteration number
    :param errs: Errors over time
    :param hopLength: The hop length (for plotting)
    """
    import librosa
    import librosa.display
    K = W.shape[2]

    plt.subplot(2, 2+K, 1)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(V), hop_length = hopLength, \
                                    y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(V, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.title("V")
    plt.subplot(2, 2+K, 3+K)
    WH = multiplyConv2D(W, H)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(WH), hop_length = hopLength, \
            y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(WH, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.title("W*H Iteration %i"%iter)

    plt.subplot(2, 2+K, 2+K+2)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(V-WH), hop_length = hopLength, \
            y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(V-WH, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.title("Residual")

    for k in range(K):
        plt.subplot(2, 2+K, 2+k+1)
        if hopLength > -1:
            librosa.display.specshow(librosa.amplitude_to_db(W[:, :, k].T), \
                hop_length=hopLength, y_axis='log', x_axis='time')
        else:
            plt.imshow(W[:, :, k].T, cmap = 'afmhot', \
                    interpolation = 'nearest', aspect = 'auto')  
            plt.colorbar()
        plt.title("W%i"%k)

        plt.subplot(2, 2+K, (2+K)+2+k+1)
        plt.imshow(H[:, k, :], cmap = 'afmhot', \
            interpolation = 'nearest', aspect = 'auto')
        if hopLength > -1:
            plt.gca().invert_yaxis()
        plt.colorbar()
        plt.title("H%i"%k)

    plt.subplot(2, 2+K, 2)
    errs = np.array(errs)
    if len(errs) > 1:
        errs = errs[1::]
    plt.semilogy(errs)
    plt.title("Errors")

if __name__ == '__main__':
    np.random.seed(10)
    X = np.random.randn(5, 10)
    plt.subplot(141)
    plt.imshow(shiftMatLRUD(X, dj=-3), interpolation = 'none')
    plt.subplot(142)
    plt.imshow(shiftMatLRUD(X, dj=3), interpolation = 'none')
    plt.subplot(143)
    plt.imshow(shiftMatLRUD(X, di=-2, dj=-3), interpolation = 'none')
    plt.subplot(144)
    plt.imshow(shiftMatLRUD(X, di=2, dj=-3), interpolation = 'none')
    plt.show()
