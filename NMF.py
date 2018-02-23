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
        plt.imshow(librosa.amplitude_to_db(W), cmap = 'afmhot', \
                interpolation = 'nearest', aspect = 'auto')        
    else:
        plt.imshow(W, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("W")
    plt.subplot(154)
    if hopLength > -1:
        plt.imshow(librosa.amplitude_to_db(H), cmap = 'afmhot', \
                interpolation = 'nearest', aspect = 'auto')        
    else:
        plt.imshow(H, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("H Iteration %i"%iter)
    plt.subplot(155)
    plt.semilogy(np.array(errs[1::]))
    plt.title("KL Errors")
    plt.xlabel("Iteration")             

def doNMF(V, K, L, plotfn = None):
    N = V.shape[0]
    M = V.shape[1]
    W = np.random.rand(N, K)
    H = np.random.rand(K, M)
    errs = [getKLError(V, W.dot(H))]
    if plotfn:
        res=4
        plt.figure(figsize=(res*5, res))
        plotfn(V, W, H, 0, errs) 
        plt.savefig("NMF_%i.png"%0, bbox_inches = 'tight')
    for l in range(L):
        print("NMF iteration %i of %i"%(l+1, L))            
        #KL Divergence Version
        VLam = V/(W.dot(H))
        H *= (W.T).dot(VLam)/np.sum(W, 0)[:, None]
        VLam = V/(W.dot(H))
        W *= (VLam.dot(H.T))/np.sum(H, 1)[None, :]
        errs.append(getKLError(V, W.dot(H)))
        if plotfn and (l+1)%10 == 0:
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            plt.savefig("NMF_%i.png"%(l+1), bbox_inches = 'tight')
    return (W, H)

def doNMFWFixed(V, W, L, simple = True, plotfn = None):
    """
    Implementing the technique described in "Let It Bee"
    :param V: An N x M target
    :param W: An N x K source/corpus matrix
    :param L: Number of iterations
    :param simple: If true, do ordinary KL-Divergence based NMF\
        Otherwise, do the variant described in [1]
    :param hopLength: The hop length (for plotting)
    :param plotfn: A function used to plot each iteration, which should\
        take the arguments (V, W, H, iter)
    :returns H: A KxM matrix of source activations for each column of V
    """
    N = V.shape[0]
    M = V.shape[1]
    K = W.shape[1]
    H = np.random.rand(K, M)
    WDenom = np.sum(W, 0)
    errs = [getKLError(V, W.dot(H))]
    if plotfn:
        res = 5
        plt.figure(figsize=(res*4, res))
        plotfn(V, W, H, 0, errs) 
        plt.savefig("NMFWFixed%i.png"%0, bbox_inches = 'tight')
    for l in range(L):
        print("NMF iteration %i of %i"%(l+1, L))
        if simple:
            C = np.array(H)            
        #KL Divergence Version
        H = C*(W.T.dot(V/(W.dot(C)))/WDenom[:, None])
        errs.append(getKLError(V, W.dot(H)))
        if plotfn:
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            plt.savefig("NMFWFixed%i.png"%(l+1), bbox_inches = 'tight')
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

def doNMF1DConv(V, K, T, L, W = np.array([]), plotfn = None, plotComponents = True):
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
    H = np.random.rand(K, M)
    errs = [getKLError(V, multiplyConv1D(W, H))]
    if plotfn:
        res=4
        pK = K
        if not plotComponents:
            pK = 0
        plt.figure(figsize=((4+pK)*res, res))
        plotfn(V, W, H, 0, errs) 
        plt.savefig("NMF1DConv_%i.png"%0, bbox_inches = 'tight')
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
        errs.append(getKLError(V, multiplyConv1D(W, H)))
        if plotfn and ((l+1) == L or (l+1)%20 == 0):
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            plt.savefig("NMF1DConv_%i.png"%(l+1), bbox_inches = 'tight')
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

def doKMeansComplexSpecs(W, NKMeans):
    from sklearn.cluster import KMeans
    M = W.shape[0]
    T = W.shape[2]
    #Reshape to an (NxMxT) array
    X = np.swapaxes(W, 0, 1)
    X = np.reshape(X, [X.shape[0], X.shape[1]*X.shape[2]])
    #Now extract the real and imaginary components
    Y = np.zeros((X.shape[0], X.shape[1]*2))
    Y[:, 0:X.shape[1]] = np.real(X)
    Y[:, X.shape[1]::] = np.imag(X)
    #Do PCA to reduce dimension before KMeans, since ambient
    #dimension is usually much higher than number of points
    tic = time.time()
    YCov = Y.dot(Y.T)
    [lam, U] = np.linalg.eigh(YCov)
    pos_lam_inds = lam > 1e-10
    lam = lam[pos_lam_inds]
    U = U[:, pos_lam_inds]
    VT = U.T.dot(Y)/np.sqrt(lam[:, None])
    Yp = U*np.sqrt(lam[None, :])
    print("Elapsed Time PCA: %.3g"%(time.time() - tic))
    tic = time.time()
    kmeans = KMeans(n_clusters = NKMeans, random_state=0, n_jobs=-1).fit(Yp)
    Yp = kmeans.cluster_centers_
    Y = Yp.dot(VT)
    print("Elapsed Time KMeans: %.3g"%(time.time() - tic))
    #Now put result back to imaginary
    X = Y[:, 0:X.shape[1]] + np.complex(0, 1)*Y[:, X.shape[1]::]
    #Reshape back to proper dimensions
    X = np.reshape(X, [X.shape[0], M, T])
    X = np.swapaxes(X, 0, 1)
    return X


def doDLComplexSpecs(W, NComponents):
    import spams
    M = W.shape[0]
    N = W.shape[1]
    T = W.shape[2]
    #Reshape to an (NxMxT) array
    X = np.swapaxes(W, 0, 1)
    X = np.reshape(X, [X.shape[0], X.shape[1]*X.shape[2]])
    #Now extract the real and imaginary components
    Y = np.zeros((X.shape[0], X.shape[1]*2))
    Y[:, 0:X.shape[1]] = np.real(X)
    Y[:, X.shape[1]::] = np.imag(X)
    
    Y = np.asfortranarray(Y.T)
    param = {'K':NComponents, 'lambda1':0.15, 'numThreads':8, 'iter':100}
    Y = spams.trainDL(Y, **param).T
    
    #Now put result back to imaginary
    X = Y[:, 0:X.shape[1]] + np.complex(0, 1)*Y[:, X.shape[1]::]
    #Reshape back to proper dimensions
    X = np.reshape(X, [X.shape[0], M, T])
    X = np.swapaxes(X, 0, 1)
    return X

def doPCAComplexSpecs(W, NComponents):
    from sklearn.decomposition import PCA


def getComplexNMF1DDictionary(Ss1, W1, Ratios1, Ss2, W2, H, winSize, hopSize, Fs, \
        ratio = 0.1, shifts = np.arange(-6, 7), NKMeans = 10, fileprefix = ""):
    """
    For each component, pull out all T-length blocks from S1 and S2 whose ratio in song 1 is\
    greater than "ratio,", then perform KMeans on their joint embedding with "NKMeans",\
    components, and add those to the dictionary for every pitch shift of every component
    :param Ss1: An array of K MxN complex spectrogram components
    :param W1: An MxKxT matrix of K sources over frequencytemporal spans MxT
    :param Ratios1: An K-length array of ratios for every spectrogram frame for every component
    :param Ss2: An array of K MxN complex spectrogram components in correspondence with S1
    :param W2: An MxKxT matrix of K sources over frequencytemporal spans MxT, \
        in correspondence with W1
    :param H: A KxN matrix of source activations for each column of |S|
    :param winSize: Window size of STFT
    :param hopSize: Hop size of STFT
    :param Fs: Sample rate of audio signal
    :param ratio: Ratio above which to keep
    :param shifts: Pitch shifts to do
    :param NKMeans: Number of KMeans clusters to find for each component
    :param fileprefix: If specified, save KMeans centers to file
    """
    from SpectrogramTools import STFT, iSTFT
    import pyrubberband as pyrb
    from scipy.io import wavfile
    NF = Ss1[0].shape[0]
    N = Ss1[0].shape[1]
    M = W1.shape[0]
    K = W1.shape[1]
    T = W1.shape[2]
    W1Ret = np.array([])
    W2Ret = np.array([])
    for k in range(K):
        print("Making dictionary from component %i of %i..."%(k+1, K))
        tic = time.time()
        #Spectrograms in correspondence for this component
        S1k = Ss1[k]
        S2k = Ss2[k]
        #Arrays holding all pitch shifted versions
        Ss1k = [S1k]
        Ss2k = [S2k]
        #Zero shifted time domain
        X1k = iSTFT(S1k, winSize, hopSize)
        X2k = iSTFT(S2k, winSize, hopSize)
        #Precompute pitch shifted versions of this component
        shiftorder = [0]
        for shift in shifts:
            if shift == 0:
                continue
            shiftorder.append(shift)
            Y1 = pyrb.pitch_shift(X1k, Fs, shift)
            Y2 = pyrb.pitch_shift(X2k, Fs, shift)
            Ss1k.append(STFT(Y1, winSize, hopSize))
            Ss2k.append(STFT(Y2, winSize, hopSize))
        #Compute time delay embedding of the joint embedding of each pitch
        #shifted spectrogram for this component
        ND = N-T+1
        Wsk = []
        for (S1ki, S2ki) in zip(Ss1k, Ss2k):
            S = np.concatenate((S1ki, S2ki), 0)
            Wski = np.zeros((M*2, ND, T), dtype=S.dtype)
            for t in range(T):
                Wski[:, :, t] = S[:, t:t+ND]
            Wsk.append(Wski)
        #Retain only the components whose mean ratio exceeds "ratio"
        RatioWin = np.cumsum(np.concatenate(([0], Ratios1[k].flatten())))
        RatioWin = (RatioWin[T::] - RatioWin[0:-T])/float(T)
        idx = np.arange(ND)
        idx = idx[RatioWin >= ratio]
        print("Component %i retaining %.3g, %i total"%(k, float(len(idx))/ND, len(idx)))
        for i, Wski in enumerate(Wsk):
            #Now do KMeans on Wski
            print("Doing kmeans shift %i..."%shiftorder[i])
            Wski = Wski[:, idx, :]
            #Normalize each block by power
            #Wski /= np.sum(np.sum(Wski*np.conj(Wski), 0), 1)[None, :, None]
            #Do KMeans
            if Wski.shape[1] > NKMeans:
                Wski = doKMeansComplexSpecs(Wski, NKMeans)
            if len(fileprefix) > 0:
                #Put all of the dictionary elements from song 1 next to each other,
                #separated by blank spaces, followed by the dictionary elements
                #from song 2
                NElems = Wski.shape[1]
                S = np.zeros((M, NElems*T*4), dtype=np.complex)
                for j in range(NKMeans):
                    S[:, j*2*T:j*2*T+T] = Wski[0:NF, j, :]
                    S[:, NElems*T*2+j*2*T:NElems*T*2+j*2*T+T] = Wski[NF::, j, :]
                X = iSTFT(S, winSize, hopSize)
                X = X/np.max(np.abs(X))
                wavfile.write("%s_X%i_shift%i.wav"%(fileprefix, k, shiftorder[i]), Fs, X)
            Wski1 = Wski[0:NF, :, :]
            Wski2 = Wski[NF::, :, :]
            if W1Ret.size == 0:
                W1Ret = Wski1
                W2Ret = Wski2
            else:
                W1Ret = np.concatenate((W1Ret, Wski1), 1)
                W2Ret = np.concatenate((W2Ret, Wski2), 1)
        print("Elapsed Time: %.3g"%(time.time() - tic))
    return (W1Ret, W2Ret)

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
