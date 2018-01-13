import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from SpectrogramTools import *
from NMF import *

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
    testNMFMusaicingSimple()
