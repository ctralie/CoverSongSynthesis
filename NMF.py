import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def plotNMFSpectra(V, W, H, iter, hopLength):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H
    :param V: An N x M target
    :param W: An N x K source/corpus matrix
    :returns H: A KxM matrix of source activations for each column of V
    :param iter: The iteration number
    :param hopLength: The hop length (for plotting)
    """
    import librosa
    import librosa.display
    plt.subplot(131)
    librosa.display.specshow(librosa.amplitude_to_db(V), hop_length = hopLength, \
                                y_axis = 'log', x_axis = 'time')
    plt.title("V")
    plt.subplot(132)
    plt.imshow(librosa.amplitude_to_db(H), cmap = 'afmhot', \
                interpolation = 'nearest', aspect = 'auto')        
    plt.title("H Iteration %i"%iter)
    plt.subplot(133)
    librosa.display.specshow(librosa.amplitude_to_db(W.dot(H)), hop_length = hopLength, \
                                y_axis = 'log', x_axis = 'time')        
    plt.title("W*H Iteration %i"%iter)               


def doNMF(V, W, L, simple = True, plotfn = None):
    """
    Implementing the technique described in 
    []
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
    if plotfn:
        plt.figure(figsize=(20, 6))
        plotfn(V, W, H, 0) 
        plt.savefig("NMF%i.png"%0, bbox_inches = 'tight')
    for l in range(L):
        print("NMF iteration %i of %i"%(l+1, L))
        if simple:
            C = np.array(H)            
        #KL Divergence Version
        H = C*(W.T.dot(V/(W.dot(C)))/WDenom[:, None])
        sio.savemat("NMF.mat", {"V":V, "W":W, "H":H, "l":l})
        if plotfn:
            plt.clf()
            plotfn(V, W, H, l+1)
            plt.savefig("NMF%i.png"%(l+1), bbox_inches = 'tight')
    return H

def shiftMat(X, dr):
    if dr == 0:
        return X
    XRet = 0*X
    XRet[:, dr::] = X[:, 0:-dr]
    return XRet    

def multiplyConv1D(W, H):
    Lam = np.zeros((W.shape[0], H.shape[1]))
    for t in range(W.shape[2]):
        Lam += W[:, :, t].dot(shiftMat(H, t))
    return Lam

def doNMF1DConv(V, K, L, T, plotfn = None):
    N = V.shape[0]
    M = V.shape[1]
    W = np.random.rand(N, K, T)
    H = np.random.rand(K, M)
    if plotfn:
        plt.figure(figsize=(20, 6))
        plotfn(V, W, H, 0) 
        plt.savefig("NMF1DConv_%i.png"%0, bbox_inches = 'tight')
    for l in range(L):
        print("NMF iteration %i of %i"%(l+1, L))            
        #KL Divergence Version
        VLam = V/multiplyConv1D(W, H)
        HNew = 0*H
        for t in range(T):
            thisW = W[:, :, t]
            HNew += H*(thisW.T).dot(shiftMat(VLam, t))/np.sum(thisW, 0)[:, None]
        H = HNew
        VLam = V/multiplyConv1D(W, H)
        for t in range(T):
            HShift = shiftMat(H, t)
            W[:, :, t] *= (VLam.dot(HShift.T))/np.sum(H, 1)[None, :]
        #sio.savemat("NMF.mat", {"V":V, "W":W, "H":H, "l":l})
        if plotfn:
            plt.clf()
            plotfn(V, W, H, l+1)
            plt.savefig("NMF1DConv_%i.png"%(l+1), bbox_inches = 'tight')
    return (W, H)

def plotNMF1DConvSpectra(V, W, H, iter, hopLength = -1):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H
    :param V: An N x M target
    :param W: An N x K x T source/corpus matrix
    :returns H: A KxM matrix of source activations for each column of V
    :param iter: The iteration number
    :param hopLength: The hop length (for plotting)
    """
    import librosa
    import librosa.display
    K = W.shape[1]
    plt.subplot(1, 3+K, 1)
    if hopLength > -1:
        librosa.display.specshow(librosa.amplitude_to_db(V), hop_length = hopLength, \
                                    y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(librosa.amplitude_to_db(V), cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
    plt.title("V")
    plt.subplot(1, 3+K, 3+K)
    plt.imshow(librosa.amplitude_to_db(H), cmap = 'afmhot', \
                interpolation = 'nearest', aspect = 'auto')  
    plt.colorbar()
    plt.title("H")
    for k in range(K):
        plt.subplot(1, 3+K, 3+k)
        plt.imshow(W[:, k, :], cmap = 'afmhot', \
                interpolation = 'nearest', aspect = 'auto')  
        plt.colorbar()
        plt.title("W%i"%k)
    plt.subplot(1, 3+K, 2)
    WH = librosa.amplitude_to_db(multiplyConv1D(W, H))
    if hopLength > -1:
        librosa.display.specshow(WH, hop_length = hopLength, y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(WH, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
    plt.title("W*H Iteration %i"%iter) 