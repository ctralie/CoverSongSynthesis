import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from SpectrogramTools import *
import pyrubberband as pyrb

def plotNMF(V, W, H, iter, hopLength):
    """
    Plot NMF iterations on a log scale, showing V, H, and W*H
    :param V: An N x M target
    :param W: An N x K source/corpus matrix
    :returns H: A KxM matrix of source activations for each column of V
    :param iter: The iteration number
    :param hopLength: The hop length (for plotting)
    """
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

def doNMF(V, W, L, simple = True, hopLength = 256, doPlot = False):
    """
    Implementing the technique described in 
    []
    :param V: An N x M target
    :param W: An N x K source/corpus matrix
    :param L: Number of iterations
    :param simple: If true, do ordinary KL-Divergence based NMF\
        Otherwise, do the variant described in [1]
    :param hopLength: The hop length (for plotting)
    :param doPlot: Whether to plot each iteration
    :returns H: A KxM matrix of source activations for each column of V
    """
    N = V.shape[0]
    M = V.shape[1]
    K = W.shape[1]
    H = np.random.rand(K, M)
    WDenom = np.sum(W, 0)
    if doPlot:
        plt.figure(figsize=(20, 6))
        plotNMF(V, W, H, 0, hopLength) 
        plt.savefig("NMF%i.png"%0, bbox_inches = 'tight')
    for l in range(L):
        print("NMF iteration %i of %i"%(l+1, L))
        if simple:
            C = np.array(H)            
        H = C*(W.T.dot(V/(W.dot(C)))/WDenom[:, None])
        sio.savemat("NMF.mat", {"V":V, "W":W, "H":H, "l":l})
        if doPlot:
            plt.clf()
            plotNMF(V, W, H, l+1, hopLength)
            plt.savefig("NMF%i.png"%(l+1), bbox_inches = 'tight')
    return H

def getPitchShiftedSpecs(X, Fs, W, H, shiftrange = 6):
    """
    Concatenate a bunch of pitch shifted versions of the spectrograms
    of a sound, using the rubberband library
    :param X: A mono audio array
    :param Fs: Sample rate
    :param W: Window size
    :param H: Hop size
    :param shiftrange: The number of halfsteps below and above which \
        to shift the sound
    :returns SRet: The concatenate spectrogram
    """
    SRet = np.array([])
    for shift in range(-shiftrange, shiftrange+1):
        print("Computing STFT pitch shift %i"%shift)
        if shift == 0:
            Y = np.array(X)
        else:
            Y = pyrb.pitch_shift(X, Fs, shift)
        S = STFT(Y, W, H)
        if SRet.size == 0:
            SRet = S
        else:
            SRet = np.concatenate((SRet, S), 1)
    return SRet

if __name__ == '__main__':
    import librosa
    import librosa.display
    winSize = 2048
    hopSize = 256

    X, Fs = librosa.load("music/Bees_Buzzing.mp3")
    WComplex = getPitchShiftedSpecs(X, Fs, winSize, hopSize, 6)
    W = np.abs(WComplex)
    X, Fs = librosa.load("music/Beatles_LetItBe.mp3")
    V = np.abs(STFT(X, winSize, hopSize))

    #librosa.display.specshow(librosa.amplitude_to_db(H), y_axis = 'log', x_axis = 'time')
    H = doNMF(V, W, 10, hopLength = hopSize, doPlot = True)
    V2 = W.dot(H)

    print("Doing phase retrieval...")
    Y = griffinLimInverse(V2, winSize, hopSize)
    Y = Y/np.max(np.abs(Y))
    wavfile.write("letitbee.wav", Fs, Y)
