import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from SpectrogramTools import *
from NMF import *
from NMFJoint import *

def testNMFJointSynthetic():
    np.random.seed(100)
    N = 20
    M = 100
    K = 6
    H = np.random.rand(K, M)
    W1 = np.random.rand(N, K)
    W2 = np.random.rand(N, K)
    X1 = W1.dot(H)
    X2 = W2.dot(H)
    lambdas = [0.01]*2
    plotfn = lambda Xs, Us, Vs, VStar, errs: \
            plotJointNMFwGT(Xs, Us, Vs, VStar, [W1, W2], [H.T, H.T], errs)
    res = doJointNMF([X1, X2], lambdas, K, tol = 0.01, Verbose = True, plotfn = plotfn)
    res['X1'] = X1
    res['X2'] = X2
    sio.savemat("JointNMF.mat", res)

def testNMFMusaicingSimple():
    import librosa
    winSize = 2048
    hopSize = 256

    X, Fs = librosa.load("music/Bees_Buzzing.mp3")
    WComplex = getPitchShiftedSpecs(X, Fs, winSize, hopSize, 6)
    W = np.abs(WComplex)
    X, Fs = librosa.load("music/Beatles_LetItBe.mp3")
    V = np.abs(STFT(X, winSize, hopSize))

    #librosa.display.specshow(librosa.amplitude_to_db(H), y_axis = 'log', x_axis = 'time')
    fn = lambda V, W, H, iter: plotNMFSpectra(V, W, H, iter, hopSize)
    H = doNMF(V, W, 10, plotfn = fn)
    V2 = W.dot(H)

    print("Doing phase retrieval...")
    Y = griffinLimInverse(V2, winSize, hopSize)
    Y = Y/np.max(np.abs(Y))
    wavfile.write("letitbee.wav", Fs, Y)


if __name__ == '__main__':
    #testNMFMusaicingSimple()
    testNMFJointSynthetic()