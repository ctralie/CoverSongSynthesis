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
        H = C*(W.T.dot(V/(W.dot(C)))/WDenom[:, None])
        sio.savemat("NMF.mat", {"V":V, "W":W, "H":H, "l":l})
        if plotfn:
            plt.clf()
            plotfn(V, W, H, l+1)
            plt.savefig("NMF%i.png"%(l+1), bbox_inches = 'tight')
    return H
