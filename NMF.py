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

def doNMFDreidger(V, W, L, r = 7, p = 10, c = 3, plotfn = None):
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
    N = V.shape[0]
    M = V.shape[1]
    K = W.shape[1]
    tic = time.time()
    H = np.random.rand(K, M)
    print("Time elapsed H initializing: %.3g"%(time.time() - tic))
    errs = [getKLError(V, W.dot(H))]
    if plotfn:
        res=4
        plt.figure(figsize=(res*5, res))
        plotfn(V, W, H, 0, errs) 
        plt.savefig("NMFDreidger_%i.png"%0, bbox_inches = 'tight')
    
    #Setup indicator matrices for diagonals
    [J, I] = np.meshgrid(np.arange(M), np.arange(K))

    for l in range(L):
        print("NMF Dreidger iteration %i of %i"%(l+1, L))   
        iterfac = 1-float(l+1)/L       
        #Step 1: Avoid repeated activations
        MuH = scipy.ndimage.filters.maximum_filter(H, size=(1, r))
        R = np.array(H)
        R[R<MuH] = R[R<MuH]*iterfac
        #Step 2: Restrict number of simultaneous activations
        colCutoff = -np.sort(-R, 0)[p, :]
        P = np.array(R)
        P[P < colCutoff[None, :]] = P[P < colCutoff[None, :]]*iterfac
        #Step 3: Supporting time-continuous activations
        C = np.array(P)
        if c > 0:
            for k in range(-C.shape[0]+1, C.shape[1]):
                z = np.cumsum(np.concatenate((np.zeros(c), np.diag(C, k), np.zeros(c))))
                x2 = z[2*c::] - z[0:-2*c]
                C[np.diag(I, k), np.diag(J, k)] = x2
        #KL Divergence Version
        tic = time.time()
        VLam = V/(W.dot(C))
        print("VLam Elapsed Time: %.3g"%(time.time() - tic))
        tic = time.time()
        H = C*((W.T).dot(VLam)/np.sum(W, 0)[:, None])
        print("Elapsed Time H Update %.3g"%(time.time() - tic))
        errs.append(getKLError(V, W.dot(H)))
        if plotfn and ((l+1)==L or (l+1)%20 == 0):
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            plt.savefig("NMDreidger_%i.png"%(l+1), bbox_inches = 'tight')
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
    Perform a convolutive matrix multiplication in time
    """
    Lam = np.zeros((W.shape[0], H.shape[1]), dtype=W.dtype)
    for t in range(W.shape[2]):
        Lam += W[:, :, t].dot(shiftMatLRUD(H, dj=t))
    return Lam

def multiplyConv2D(W, H):
    """
    Perform a convolutive matrix multiplication in time and frequency
    """
    Lam = np.zeros((W.shape[0], H.shape[1]), dtype=W.dtype)
    for t in range(W.shape[2]):
        for f in range(H.shape[2]):
            Wt = np.array(W[:, :, t])
            Wt = shiftMatLRUD(Wt, di=f)
            Hf = np.array(H[:, :, f])
            Hf = shiftMatLRUD(Hf, dj=t)
            Lam += Wt.dot(Hf)
    return Lam

def doNMF1DConv(V, K, T, L, W = np.array([]), plotfn = None, plotComponents = True, \
        joint = False, prefix=""):
    """
    Implementing the technique described in 
    "Non-negative Matrix Factor Deconvolution; Extraction of
        Multiple Sound Sources from Monophonic Inputs"
    NOTE: Using update rules from 2DNMF instead of averaging
    :param V: An N x M target matrix
    :param K: Number of latent factors
    :param T: Time extent of W matrices
    :param L: Number of iterations
    :param plotfn: A function used to plot each iteration, which should\
        take the arguments (V, W, H, iter)
    :param joint: If true, it's understood that V and W are two songs concatenated\
        on top of each other.  Functionality is the same, but errors are computed\
        separately, and plotting is done differently
    :returns (W, H): \
        W is an NxKxT matrix of K sources over spatiotemporal spans NxT
        H is a KxM matrix of source activations for each column of V
    """
    N = V.shape[0]
    M = V.shape[1]
    WFixed = False
    if W.size == 0:
        W = np.random.rand(N, K, T)
    else:
        WFixed = True
        K = W.shape[1]
    H = np.random.rand(K, M)
    WH = multiplyConv1D(W, H)
    if joint:
        N1 = int(N/2)
        errs = [[getKLError(V[0:N1, :], WH[0:N1, :]), \
                getKLError(V[N1::, :], WH[N1::, :])]]
    else:
        errs = [getKLError(V, WH)]
    if plotfn:
        res=4
        pK = K
        if not plotComponents:
            pK = 0
        if joint:
            plt.figure(figsize=((4+pK)*res, 3*res))
        else:
            plt.figure(figsize=((4+pK)*res, res))
        if not joint:
            plotfn(V, W, H, 0, errs) 
            plt.savefig("NMF1DConv_%i.png", bbox_inches = 'tight')
    for l in range(L):
        print("NMF iteration %i of %i"%(l+1, L))            
        #KL Divergence Version
        WH = multiplyConv1D(W, H)
        WH[WH == 0] = 1
        VLam = V/WH
        HNum = np.zeros(H.shape)
        HDenom = np.zeros((H.shape[0], 1))
        for t in range(T):
            thisW = W[:, :, t]
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
                W[:, :, t] *= (VLam.dot(HShift.T))/denom
        WH = multiplyConv1D(W, H)
        if joint:
            N1 = int(N/2)
            errs.append([getKLError(V[0:N1, :], WH[0:N1, :]), \
                    getKLError(V[N1::, :], WH[N1::, :])])
        else:
            errs.append([getKLError(V, WH)])
        if plotfn and ((l+1) == L):# or (l+1)%40 == 0):
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            if joint:
                pre = "%sNMF1DJointIter%i"%(prefix, l+1)
                plt.savefig("%s/NMF1DConv_%i.png"%(pre, l+1), bbox_inches = 'tight')
            else:
                plt.savefig("NMF1DConv_%i.png", bbox_inches = 'tight')
    return (W, H)

def getComplexNMF1DTemplates(S, W, H, p = 2, audioParams = None):
    """
    Given a complex spectrogram and a factorization WH ~= |S| of its magnitude
    spectrum, separate out the complex spectrogram into each of its components
    :param S: An MxN complex spectrogram
    :param W: An MxKxT matrix of K sources over frequencytemporal spans MxT
    :param H: A KxN matrix of source activations for each column of |S|
    :param p: Power for Weiner filter in soft mask matrices
    :param audioParams: {'winSize':int, 'hopSize':int, 'Fs':int, 'fileprefix':string}\
        If specified, save each component to disk as a wav file
    """
    N = S.shape[1]
    M = W.shape[0]
    K = W.shape[1]
    T = W.shape[2]
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
            Xk = iSTFT(Ss[k], winSize, hopSize)
            if k == 0:
                X = Xk
            else:
                X += Xk
            wavfile.write("%s_X%i.wav"%(fileprefix, k), Fs, Xk)
            plt.clf()
            plt.plot(Ratios[k])
            plt.title("Ratio, %.3g Above 0.1"%(np.sum(Ratios[k] > 0.1)/Ratios[k].size))
            plt.savefig("%s_%iPower.svg"%(fileprefix, k), bbox_inches = 'tight')
        wavfile.write("%sNMF.wav"%fileprefix, Fs, X)
    return (Ss, Ratios)

def doNMF2DConv(V, K, T, F, L, W = np.array([]), plotfn = None):
    """
    Implementing the Euclidean 2D NMF technique described in 
    "Nonnegative Matrix Factor 2-D Deconvolution
        for Blind Single Channel Source Separation"
    :param V: An N x M target matrix
    :param K: Number of latent factors
    :param T: Time extent of W matrices
    :param F: Frequency extent of H matrices
    :param L: Number of iterations
    :param plotfn: A function used to plot each iteration, which should\
        take the arguments (V, W, H, iter)
    :returns (W, H): \
        W is an NxKxT matrix of K sources over spatiotemporal spans NxT\
        H is a KxMxF matrix of source activations for each submatrix of W\
            over F transpositions over M time
    """
    N = V.shape[0]
    M = V.shape[1]
    WFixed = False
    if W.size == 0:
        W = np.random.rand(N, K, T)
    else:
        WFixed = True
    H = np.random.rand(K, M, F)
    errs = [getEuclideanError(V, multiplyConv2D(W, H))]
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
            #VLam[VLam == 0] = 1
            WNums = np.zeros(W.shape)
            WDenoms = np.zeros(W.shape)
            for f in range(F):
                tic = time.time()
                thisV = shiftMatLRUD(V, di=-f)
                thisVLam = shiftMatLRUD(VLam, di=-f)
                for t in range(T):
                    thisH = shiftMatLRUD(H[:, :, f], dj=t)
                    WNums[:, :, t] += thisV.dot(thisH.T)
                    WDenoms[:, :, t] += thisVLam.dot(thisH.T)
                toc = time.time()
                print("Elapsed Time Wf Iter: %.3g"%(toc-tic))
            #WDenoms[WDenoms == 0] = 1
            W = W*(WNums/WDenoms)

        #Step 2: Update Hs
        VLam = multiplyConv2D(W, H)
        #VLam[VLam == 0] = 1
        HNums = np.zeros(H.shape)
        HDenoms = np.zeros(H.shape)
        for t in range(T):
            thisV = shiftMatLRUD(V, dj=-t)
            thisVLam = shiftMatLRUD(VLam, dj=-t)
            for f in range(F):
                thisW = shiftMatLRUD(W[:, :, t], di=f)
                HNums[:, :, f] += (thisW.T).dot(thisV)
                HDenoms[:, :, f] += (thisW.T).dot(thisVLam)
        #HDenoms[HDenoms == 0] = 1
        H = H*(HNums/HDenoms)
        errs.append(getEuclideanError(V, multiplyConv2D(W, H)))
        if plotfn and ((l+1) == L or (l+1)%40 == 0):
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            plt.savefig("NMF2DConv_%i.png"%(l+1), bbox_inches = 'tight')
        #TODO: Balance normalize W and H
    return (W, H)

def plotNMF1DConvSpectra(V, W, H, iter, errs, hopLength = -1, plotComponents = True):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H
    :param V: An N x M target
    :param W: An N x K x T source/corpus matrix
    :returns H: A KxM matrix of source activations for each column of V
    :param iter: The iteration number
    :param errs: Errors over time
    :param hopLength: The hop length (for plotting)
    """
    import librosa
    import librosa.display
    K = W.shape[1]
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
                librosa.display.specshow(librosa.amplitude_to_db(W[:, k, :]), \
                    hop_length=hopLength, y_axis='log', x_axis='time')
            else:
                plt.imshow(W[:, k, :], cmap = 'afmhot', \
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
    plt.title("Errors")

def plotNMF1DConvSpectraJoint(V, W, H, iter, errs, hopLength = -1, plotComponents = True, \
        audioParams = None):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H, with the understanding
    that V and W are two songs concatenated on top of each other
    :param V: An N x M target
    :param W: An N x K x T source/corpus matrix
    :returns H: A KxM matrix of source activations for each column of V
    :param iter: The iteration number
    :param errs: Errors over time
    :param hopLength: The hop length (for plotting)
    """
    import librosa
    import librosa.display
    K = W.shape[1]
    if not plotComponents:
        K = 0

    N = int(W.shape[0]/2)
    W1 = W[0:N, :, :]
    W2 = W[N::, :, :]
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
        for k in range(W1.shape[1]):
            y_hat = griffinLimInverse(W1[:, k, :], winSize, hopLength)
            y_hat = y_hat/np.max(np.abs(y_hat))
            wavfile.write("%s/W1_%i.wav"%(pre, k), Fs, y_hat)
            y_hat = griffinLimInverse(W2[:, k, :], winSize, hopLength)
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
    plt.title("Errors 1")
    plt.subplot(3, 2+K, (2+K)*2+2)
    plt.semilogy(errs[:, 1])
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
                librosa.display.specshow(librosa.amplitude_to_db(W1[:, k, :]), \
                    hop_length=hopLength, y_axis='log', x_axis='time')
            else:
                plt.imshow(W1[:, k, :], cmap = 'afmhot', \
                        interpolation = 'nearest', aspect = 'auto')  
                plt.colorbar()
            plt.title("W1_%i"%k)
            plt.subplot(3, 2+K, 2+K+3+k)
            if hopLength > -1:
                librosa.display.specshow(librosa.amplitude_to_db(W2[:, k, :]), \
                    hop_length=hopLength, y_axis='log', x_axis='time')
            else:
                plt.imshow(W2[:, k, :], cmap = 'afmhot', \
                        interpolation = 'nearest', aspect = 'auto')  
                plt.colorbar()
            plt.title("W2_%i"%k)
            plt.subplot(3, 2+K, (2+K)*2+3+k)
            plt.plot(H[k, :])
            plt.title("H%i"%k)

def plotNMF2DConvSpectra(V, W, H, iter, errs, hopLength = -1):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H
    :param V: An N x M target
    :param W: An N x K x T source/corpus matrix
    :returns H: A K x M x F matrix of source activations
    :param iter: The iteration number
    :param errs: Errors over time
    :param hopLength: The hop length (for plotting)
    """
    import librosa
    import librosa.display
    K = W.shape[1]

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
            librosa.display.specshow(librosa.amplitude_to_db(W[:, k, :]), \
                hop_length=hopLength, y_axis='log', x_axis='time')
        else:
            plt.imshow(W[:, k, :], cmap = 'afmhot', \
                    interpolation = 'nearest', aspect = 'auto')  
            plt.colorbar()
        plt.title("W%i"%k)

        plt.subplot(2, 2+K, (2+K)+2+k+1)
        plt.imshow(H[k, :, :].T, cmap = 'afmhot', \
            interpolation = 'nearest', aspect = 'auto')
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
