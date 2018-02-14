import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def getKLError(V, WH):
    return np.sum(V*np.log(V/WH)-V+WH)

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

def shiftMatLeftRight(X, di):
    if di == 0:
        return X
    XRet = 0*X
    if di > 0:
        XRet[:, di::] = X[:, 0:-di]
    else:
        XRet[:, 0:di] = X[:, -di::]
    return XRet
    

def multiplyConv1D(W, H):
    Lam = np.zeros((W.shape[0], H.shape[1]))
    for t in range(W.shape[2]):
        Lam += W[:, :, t].dot(shiftMatLeftRight(H, t))
    return Lam

def doNMF1DConv(V, K, T, L, plotfn = None):
    """
    Implementing the technique described in 
    "Non-negative Matrix Factor Deconvolution; Extraction of
        Multiple Sound Sources from Monophonic Inputs"
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
    W = np.random.rand(N, K, T)
    H = np.random.rand(K, M)
    errs = [getKLError(V, multiplyConv1D(W, H))]
    if plotfn:
        res=4
        plt.figure(figsize=((4+K)*res, res))
        plotfn(V, W, H, 0, errs) 
        plt.savefig("NMF1DConv_%i.png"%0, bbox_inches = 'tight')
    for l in range(L):
        print("NMF iteration %i of %i"%(l+1, L))            
        #KL Divergence Version
        WH = multiplyConv1D(W, H)
        VLam = V/WH
        HNew = 0*H
        for t in range(T):
            thisW = W[:, :, t]
            fac = (thisW.T).dot(shiftMatLeftRight(VLam, -t))/np.sum(thisW, 0)[:, None]
            HNew += H*fac
        H = HNew/T
        WH = multiplyConv1D(W, H)
        VLam = V/WH
        for t in range(T):
            HShift = shiftMatLeftRight(H, t)
            W[:, :, t] *= (VLam.dot(HShift.T))/np.sum(H, 1)[None, :]
        WH = multiplyConv1D(W, H)
        errs.append(getKLError(V, WH))
        if plotfn and (l+1)%10 == 0:
            plt.clf()
            plotfn(V, W, H, l+1, errs)
            plt.savefig("NMF1DConv_%i.png"%(l+1), bbox_inches = 'tight')
    return (W, H)

def plotNMF1DConvSpectra(V, W, H, iter, errs, hopLength = -1):
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
    plt.subplot(1, 3+K, 1)
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
    for k in range(K):
        plt.subplot(1, 4+K, 3+k)
        plt.imshow(W[:, k, :], cmap = 'afmhot', \
                interpolation = 'nearest', aspect = 'auto')  
        plt.colorbar()
        plt.title("W%i"%k)
    plt.subplot(1, 4+K, 2)
    WH = multiplyConv1D(W, H)
    if hopLength > -1:
        librosa.display.specshow(WH, hop_length = hopLength, y_axis = 'log', x_axis = 'time')
    else:
        plt.imshow(WH, cmap = 'afmhot', interpolation = 'nearest', aspect = 'auto')
        plt.colorbar()
    plt.subplot(1, 4+K, 4+K)
    plt.semilogy(np.array(errs))
    plt.title("Errors")
    plt.title("W*H Iteration %i"%iter) 

if __name__ == '__main__':
    np.random.seed(10)
    X = np.random.randn(5, 10)
    plt.subplot(121)
    plt.imshow(shiftMatLeftRight(X, 3), interpolation = 'none')
    plt.subplot(122)
    plt.imshow(shiftMatLeftRight(X, -3), interpolation = 'none')
    plt.show()
