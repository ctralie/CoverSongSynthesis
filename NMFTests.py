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

def testNMF1DConvSynthetic():
    np.random.seed(100)
    N = 20
    M = 40
    K = 3
    L = 80
    T = 10
    V = 0*np.ones((N, M))
    V[5+np.arange(T), np.arange(T)] = 1
    V[5+np.arange(T), 5+np.arange(T)] = 0.5
    V[15-np.arange(T), 10+np.arange(T)] = 1
    V[5+np.arange(T), 20+np.arange(T)] = 1
    V[15-np.arange(T), 22+np.arange(T)] = 0.5
    V[5+np.arange(T), 10+np.arange(T)] += 0.7
    V *= 1000
    #doNMF(V, K*T, L, plotfn=plotNMFSpectra)
    doNMF1DConv(V, K, T+5, L, plotfn=plotNMF1DConvSpectra)

def testNMF2DConvSynthetic():
    np.random.seed(300)
    N = 20
    M = 40
    K = 2
    L = 80
    T = 10
    F = 5
    V = 0.1*np.ones((N, M))
    V[5+np.arange(T), np.arange(T)] = 1
    V[8+np.arange(T), 5+np.arange(T)] = 0.5
    V[15-np.arange(T), 10+np.arange(T)] = 1
    V[6+np.arange(T), 20+np.arange(T)] = 1
    V[10-np.arange(T), 22+np.arange(T)] = 0.5
    V[10+np.arange(T), 10+np.arange(T)] += 0.7
    doNMF2DConv(V, K, T, F, L, plotfn=plotNMF2DConvSpectra)
    doNMF1DConv(V, K, T, L, plotfn=plotNMF1DConvSpectra)

"""
    :param A: An M x N1 matrix for song A
    :param Ap: An M x N1 matrix for song A'
    :param B: An M x N2 matrix for song B
    :param W1: An M x K x T source/corpus matrix for songs A and B
    :param W2: An M x K x T source/corpus matrix for song A'
    :param H1: A K x N1 x F matrix of activations for A and A'
    :param H2: A K x N2 x F matrix of activations for B
"""

def testNMF2DConvJointSynthetic():
    np.random.seed(300)
    T = 10
    F = 10
    K = 3
    M = 20
    N1 = 60
    N2 = 40
    W1 = np.zeros((M, K, T))
    W2 = np.zeros((M, K, T))
    #Pattern 1: A tall block in A that goes to a fat block in A'
    [J, I] = np.meshgrid(np.arange(2), 4+np.arange(5))
    W1[I.flatten(), 0, J.flatten()] = 1
    [J, I] = np.meshgrid(np.arange(5), 7+np.arange(2))
    W2[I.flatten(), 0, J.flatten()] = 1
    #Pattern 2: An antidiagonal line in A that goes to a diagonal line in A'
    W1[9-np.arange(7), 1, np.arange(7)] = 1
    W2[np.arange(7), 1, np.arange(7)] = 1
    #Pattern 3: A square in A that goes into a circle in A'
    [J, I] = np.meshgrid(np.arange(5), 10+np.arange(5))
    I = I.flatten()
    J = J.flatten()
    W1[np.arange(10), 2, 0] = 1
    W1[np.arange(10), 2, 9] = 1
    W1[0, 2, np.arange(10)] = 1
    W1[10, 2, np.arange(10)] = 1
    [J, I] = np.meshgrid(np.arange(T), np.arange(T))
    I = I.flatten()
    J = J.flatten()
    idx = np.arange(I.size)
    idx = idx[np.abs((I-5)**2 + (J-5)**2 - 4**2) < 4]
    I = I[idx]
    J = J[idx]
    W2[I, 2, J] = 1

    H1 = np.zeros((K, N1, F))
    H1[0, [3, 15, 50], 9] = 1
    H1[0, 27, 0] = 1
    
    #3 diagonal lines in a row, then a gap, then 3 in a row pitch shifted
    H1[1, [5, 15, 25], 0] = 1
    H1[1, [35, 45, 55], 5] = 1

    #Squares and circles moving down then up
    H1[2, [0, 48], 1] = 1
    H1[2, [12, 36], 4] = 1
    H1[2, 24, 8] = 1


    H2 = np.random.rand(K, N2, F)
    H2[H2 > 0.98] = 1
    H2[H2 < 1] = 0

    A = multiplyConv2D(W1, H1)
    Ap = multiplyConv2D(W2, H1)
    B = multiplyConv2D(W1, H2)

    doNMF2DConvJoint(A, Ap, B, K, T, F, 200, plotfn = plotNMF2DConvJointSpectra)

def outputNMFSounds(U1, U2, winSize, hopSize, Fs, fileprefix):
    for k in range(U1.shape[1]):
        S1 = np.repeat(U1[:, k][:, None], 60, axis = 1)
        X1 = griffinLimInverse(S1, winSize, hopSize)
        X1 = X1/np.max(np.abs(X1))
        S2 = np.repeat(U2[:, k][:, None], 60, axis = 1)
        X2 = griffinLimInverse(S2, winSize, hopSize)
        X2 = X2/np.max(np.abs(X2))
        X = np.array([X1.flatten(), X2.flatten()]).T
        sio.wavfile.write("%s%i.wav"%(fileprefix, k), Fs, X)

def testNMFJointSmoothCriminal():
    Fs, X = sio.wavfile.read("music/SmoothCriminalAligned.wav")
    X1 = X[:, 0]/(2.0**15)
    X2 = X[:, 1]/(2.0**15)
    #Only take first 30 seconds for initial experiments
    X1 = X1[0:Fs*30]
    X2 = X2[0:Fs*30]
    hopSize = 256
    winSize = 2048
    S1 = np.abs(STFT(X1, winSize, hopSize))
    S2 = np.abs(STFT(X2, winSize, hopSize))
    lambdas = [1e-4]*2
    K = S1.shape[1]*2
    plotfn = lambda Xs, Us, Vs, VStar, errs: \
            plotJointNMFSpectra(Xs, Us, Vs, VStar, errs, hopSize)
    res = doJointNMF([S1, S2], lambdas, K, tol = 0.01, Verbose = True, plotfn = plotfn)
    U1 = res['Us'][0]
    U2 = res['Us'][1]
    V1 = res['Vs'][0]
    V2 = res['Vs'][1]
    S1Res = U1.dot(V1.T)
    S2Res = U2.dot(V2.T)
    X1Res = griffinLimInverse(S1Res, winSize, hopSize, NIters = 10)
    X2Res = griffinLimInverse(S2Res, winSize, hopSize, NIters = 10)
    X1Res = X1Res/np.max(np.abs(X1Res))
    X2Res = X2Res/np.max(np.abs(X2Res))
    sio.wavfile.write("MJ_%i_%.3g.wav"%(K, lambdas[0]), Fs, X1Res)
    sio.wavfile.write("AAF_%i_%.3g.wav"%(K, lambdas[0]), Fs, X2Res)
    #outputNMFSounds(U1, U2, winSize, hopSize, Fs, "MAJF")
    #sio.savemat("JointNMFSTFT.mat", res)

    #Now represent Bad in MJ's basis
    import librosa
    X, Fs = librosa.load("music/MJBad.mp3")
    X = X[0:Fs*30]
    S = np.abs(STFT(X, winSize, hopSize))
    fn = lambda V, W, H, iter, errs: plotNMFSpectra(V, W, H, iter, errs, hopSize)
    H = doNMFWFixed(S, U1, 10, plotfn = fn)
    SRes = U1.dot(H)
    XRes = griffinLimInverse(SRes, winSize, hopSize, NIters = 10)
    SResCover = U2.dot(H)
    XResCover = griffinLimInverse(SResCover, winSize, hopSize, NIters = 10)
    sio.wavfile.write("BadRes.wav", Fs, XRes)
    sio.wavfile.write("BadResCover.wav", Fs, XResCover)

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
    fn = lambda V, W, H, iter, errs: plotNMFSpectra(V, W, H, iter, errs, hopSize)
    H = doNMFWFixed(V, W, 10, plotfn = fn)
    V2 = W.dot(H)

    print("Doing phase retrieval...")
    Y = griffinLimInverse(V2, winSize, hopSize)
    Y = Y/np.max(np.abs(Y))
    wavfile.write("letitbee.wav", Fs, Y)

def testNMF2DMusic():
    import librosa2
    from scipy.io import wavfile
    X1, Fs = librosa2.load("music/SmoothCriminalMJ.mp3", sr=22050)
    X1 = X1[Fs*15:Fs*30]

    X2, Fs = librosa2.load("music/MJBad.mp3")
    X2 = X2[Fs*3:Fs*18]

    X = np.zeros(len(X1)+len(X2)+Fs)
    X[0:len(X1)] = X1
    X[-len(X2)::] = X2

    hopSize = 64
    bins_per_octave = 24
    noctaves = 7
    K = 20
    T = 40
    F = 18
    NIters = 440

    """
    D = librosa2.stft(X)
    H, P = librosa2.decompose.hpss(D)
    X_harm = librosa2.core.istft(H)
    X_perc = librosa2.core.istft(P)
    """

    C = librosa2.cqt(y=X, sr=Fs, hop_length=hopSize, n_bins=noctaves*bins_per_octave,\
               bins_per_octave=bins_per_octave)
    C = np.abs(C)
    y_hat = griffinLimCQTInverse(C, Fs, hopSize, bins_per_octave, NIters=10)
    y_hat = y_hat/np.max(np.abs(y_hat))
    sio.wavfile.write("smoothcriminalGTInverted.wav", Fs, y_hat)

    plotfn = lambda V, W, H, iter, errs: plotNMF2DConvSpectra(V, W, H, iter, errs, hopLength = hopSize)
    (W, H) = doNMF2DConv(C, K, T=T, F=18, L=NIters, plotfn=plotfn)
    sio.savemat("SmoothCriminalNMF2D.mat", {"W":W, "H":H})
    C = multiplyConv2D(W, H)
    print("C.shape = ", C.shape)
    y_hat = griffinLimCQTInverse(C, Fs, hopSize, bins_per_octave, NIters=10)
    y_hat = y_hat/np.max(np.abs(y_hat))
    sio.wavfile.write("smoothcriminalNMF.wav", Fs, y_hat)

    #Also invert each Wt
    for k in range(W.shape[1]):
        y_hat = griffinLimCQTInverse(W[:, k, :], Fs, hopSize, bins_per_octave, NIters=10)
        y_hat = y_hat/np.max(np.abs(y_hat))
        sio.wavfile.write("smoothcriminalW%i.wav"%k, Fs, y_hat)

def testNMF1DMusic():
    import librosa
    from scipy.io import wavfile
    Fs, X = sio.wavfile.read("music/SmoothCriminalAligned.wav")
    X1 = X[:, 0]/(2.0**15)
    X2 = X[:, 1]/(2.0**15)
    #Only take first 30 seconds for initial experiments
    X1 = X1[0:Fs*30]
    X2 = X2[0:Fs*30]
    hopSize = 128
    winSize = 2048
    S1 = STFT(X1, winSize, hopSize)
    N = S1.shape[0]
    S2 = STFT(X2, winSize, hopSize)
    S = np.abs(np.concatenate((S1, S2), 0))
    plotfn = lambda V, W, H, iter, errs: plotNMF1DConvSpectra(V, W, H, iter, errs, hopLength = hopSize)
    (W, H) = doNMF1DConv(S, 10, T=25, L=80, plotfn=plotfn)
    sio.savemat("SmoothCriminalNMF.mat", {"W":W, "H":H})
    S = multiplyConv1D(W, H)
    print("S.shape = ", S.shape)
    S1 = S[0:N, :]
    S2 = S[N::, :]
    y_hat = griffinLimInverse(S1, winSize, hopSize)
    y_hat = y_hat/np.max(np.abs(y_hat))
    sio.wavfile.write("smoothcriminalNMFMJ.wav", Fs, y_hat)
    y_hat = griffinLimInverse(S2, winSize, hopSize)
    y_hat = y_hat/np.max(np.abs(y_hat))
    sio.wavfile.write("smoothcriminalNMFAAF.wav", Fs, y_hat)

    #Also invert each Wt
    for k in range(W.shape[1]):
        Wk = np.array(W[:, k, :])
        Wk1 = Wk[0:N, :]
        Wk2 = Wk[N::, :]
        y_hat = griffinLimInverse(Wk1, winSize, hopSize)
        y_hat = y_hat/np.max(np.abs(y_hat))
        sio.wavfile.write("smoothcriminalW%iMJ.wav"%k, Fs, y_hat)
        y_hat = griffinLimInverse(Wk2, winSize, hopSize)
        y_hat = y_hat/np.max(np.abs(y_hat))
        sio.wavfile.write("smoothcriminalW%iAAF.wav"%k, Fs, y_hat)

def testNMF1DTranslate():
    import librosa
    from scipy.io import wavfile
    hopSize = 512
    winSize = 4096
    K=40
    T=6
    plotfn = lambda V, W, H, iter, errs: plotNMF1DConvSpectra(V, W, H, iter, errs, hopLength = hopSize)

    Fs, X = sio.wavfile.read("music/SmoothCriminalAligned44100.wav")
    print("Fs = ", Fs)
    X1 = X[:, 0]/(2.0**15)
    X2 = X[:, 1]/(2.0**15)
    #Only take first 30 seconds for initial experiments
    X1 = X1[0:Fs*30]
    X2 = X2[0:Fs*30]

    S1 = getPitchShiftedSpecs(X1, Fs, winSize, hopSize)
    N = S1.shape[0]
    S2 = getPitchShiftedSpecs(X2, Fs, winSize, hopSize)
    S = np.abs(np.concatenate((S1, S2), 0))
    (W, H) = doNMF1DConv(S, K, T=T, L=80, plotfn=plotfn)
    sio.savemat("SmoothCriminalNMF.mat", {"W":W, "H":H})

    """
    W = sio.loadmat("SmoothCriminalNMF.mat")['W']
    Fs = 22050
    N = 1025
    """

    for k in range(W.shape[1]):
        Wk = np.array(W[:, k, :])
        Wk1 = Wk[0:N, :]
        Wk2 = Wk[N::, :]
        y_hat = griffinLimInverse(Wk1, winSize, hopSize)
        y_hat = y_hat/np.max(np.abs(y_hat))
        sio.wavfile.write("smoothcriminalW%iMJ.wav"%k, Fs, y_hat)
        y_hat = griffinLimInverse(Wk2, winSize, hopSize)
        y_hat = y_hat/np.max(np.abs(y_hat))
        sio.wavfile.write("smoothcriminalW%iAAF.wav"%k, Fs, y_hat)

    W1 = W[0:N, :]
    W2 = W[N::, :]

    X, Fs = librosa.load("music/MJBad.mp3")
    X = X[Fs*3:Fs*20]
    S = np.abs(STFT(X, winSize, hopSize))
    (W, H) = doNMF1DConv(S, K, T=T, L=80, plotfn = plotfn, W=W1)
    H = np.reshape(H, (H.shape[0], H.shape[1]))
    S1 = multiplyConv1D(W1, H)
    y_hat = griffinLimInverse(S1, winSize, hopSize)
    y_hat = y_hat/np.max(np.abs(y_hat))
    sio.wavfile.write("ReconstructedBadMJ.wav", Fs, y_hat)
    S2 = multiplyConv1D(W2, H)
    y_hat = griffinLimInverse(S2, winSize, hopSize)
    y_hat = y_hat/np.max(np.abs(y_hat))
    sio.wavfile.write("ReconstructedBadAAF.wav", Fs, y_hat)

def testNMF2DJointMusic():
    import librosa2
    from scipy.io import wavfile

    Fs, X = sio.wavfile.read("music/SmoothCriminalAligned.wav")
    A = X[:, 0]/(2.0**15)
    Ap = X[:, 1]/(2.0**15)
    #Only take 15 seconds for initial experiments
    A = A[5:Fs*20]
    Ap = Ap[5:Fs*20]

    B, Fs = librosa2.load("music/MJBad.mp3")
    B = B[Fs*3:Fs*18]

    hopSize = 64
    bins_per_octave = 24
    noctaves = 7
    K = 20
    T = 40
    F = 18
    NIters = 440

    res = {}
    for (V, s) in zip([A, Ap, B], ["A", "Ap", "B"]):
        C = librosa2.cqt(y=V, sr=Fs, hop_length=hopSize, n_bins=noctaves*bins_per_octave,\
                bins_per_octave=bins_per_octave)
        C = np.abs(C)
        res[s] = C
        """
        y_hat = griffinLimCQTInverse(C, Fs, hopSize, bins_per_octave, NIters=10)
        y_hat = y_hat/np.max(np.abs(y_hat))
        sio.wavfile.write("%sGTInverted.wav"%s, Fs, y_hat)
        """
    [CA, CAp, CB] = [res['A'], res['Ap'], res['B']]

    plotfn = lambda A, Ap, B, W1, W2, H1, H2, iter, errs: \
            plotNMF2DConvJointSpectra(A, Ap, B, W1, W2, H1, H2, iter, errs,\
            hopLength = hopSize, audioParams={'Fs':Fs, 'bins_per_octave':bins_per_octave})
    (W1, W2, H1, H2) = doNMF2DConvJoint(CA, CAp, CB, K, T, F, L=NIters, plotfn=plotfn)
    sio.savemat("SmoothCriminalNMF2DJoint.mat", {"W1":W1, "W2":W2, "H1":H1, "H2":H2})
    
    LamA = multiplyConv2D(W1, H1)
    LamAp = multiplyConv2D(W2, H1)
    LamB = multiplyConv2D(W1, H2)
    LamBp = multiplyConv2D(W2, H2)
    for (C, s) in zip([LamA, LamAp, LamB, LamBp], ["A", "Ap", "B", "Bp"]):
        y_hat = griffinLimCQTInverse(C, Fs, hopSize, bins_per_octave, NIters=10)
        y_hat = y_hat/np.max(np.abs(y_hat))
        sio.wavfile.write("%s.wav"%s, Fs, y_hat)

    #Also invert each Wt
    for k in range(W1.shape[1]):
        y_hat = griffinLimCQTInverse(W1[:, k, :], Fs, hopSize, bins_per_octave, NIters=10)
        y_hat = y_hat/np.max(np.abs(y_hat))
        sio.wavfile.write("W1_%i.wav"%k, Fs, y_hat)
        y_hat = griffinLimCQTInverse(W2[:, k, :], Fs, hopSize, bins_per_octave, NIters=10)
        y_hat = y_hat/np.max(np.abs(y_hat))
        sio.wavfile.write("W2_%i.wav"%k, Fs, y_hat)

if __name__ == '__main__':
    #testNMFMusaicingSimple()
    #testNMFJointSynthetic()
    #testNMFJointSmoothCriminal()
    #testNMF1DConvSynthetic()
    #testNMF2DConvSynthetic()
    #testNMF2DConvJointSynthetic()
    #testNMF1DMusic()
    #testNMF2DMusic()
    #testNMF1DTranslate()
    testNMF2DJointMusic()