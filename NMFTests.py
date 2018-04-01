import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from SpectrogramTools import *
from CQT import *
from NMF import *
from NMFGPU import *
from NMFJoint import *
from SongAnalogies import *

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
    initParallelAlgorithms()
    np.random.seed(300)
    N = 20
    M = 40
    K = 2
    L = 200
    T = 10
    F = 5
    V = 0.1*np.ones((N, M))
    V[5+np.arange(T), np.arange(T)] = 1
    V[8+np.arange(T), 5+np.arange(T)] = 0.5
    V[15-np.arange(T), 10+np.arange(T)] = 1
    V[6+np.arange(T), 20+np.arange(T)] = 1
    V[10-np.arange(T), 22+np.arange(T)] = 0.5
    V[10+np.arange(T), 10+np.arange(T)] += 0.7
    doNMF2DConv(V, K, T, F, L, doKL = True, plotfn=plotNMF2DConvSpectra)
    #doNMF1DConv(V, K, T, L, plotfn=plotNMF1DConvSpectra)

def get2DSyntheticJointExample():
    T = 10
    F = 10
    K = 3
    M = 20
    N = 60
    
    W1 = np.zeros((T, M, K))
    W2 = np.zeros((T, M, K))
    #Pattern 1: A tall block in A that goes to a fat block in A'
    [J, I] = np.meshgrid(np.arange(2), 4+np.arange(5))
    W1[J.flatten(), I.flatten(), 0] = 1
    [J, I] = np.meshgrid(np.arange(5), 7+np.arange(2))
    W2[J.flatten(), I.flatten(), 0] = 1
    #Pattern 2: An antidiagonal line in A that goes to a diagonal line in A'
    W1[np.arange(7), 9-np.arange(7), 1] = 1
    W2[np.arange(7), np.arange(7), 1] = 1
    #Pattern 3: A square in A that goes into a circle in A'
    [J, I] = np.meshgrid(np.arange(5), 10+np.arange(5))
    I = I.flatten()
    J = J.flatten()
    W1[0, np.arange(10), 2] = 1
    W1[9, np.arange(10), 2] = 1
    W1[np.arange(10), 0, 2] = 1
    W1[np.arange(10), 10, 2] = 1
    [J, I] = np.meshgrid(np.arange(T), np.arange(T))
    I = I.flatten()
    J = J.flatten()
    idx = np.arange(I.size)
    idx = idx[np.abs((I-5)**2 + (J-5)**2 - 4**2) < 4]
    I = I[idx]
    J = J[idx]
    W2[J, I, 2] = 1

    H = np.zeros((F, K, N))
    H[9, 0, [3, 15, 50]] = 1
    H[0, 0, 27] = 1
    
    #3 diagonal lines in a row, then a gap, then 3 in a row pitch shifted
    H[0, 1, [5, 15, 25]] = 1
    H[0, 1, [35, 45, 55]] = 1

    #Squares and circles moving down then up
    H[1, 2, [0, 48]] = 1
    H[4, 2, [12, 36]] = 1
    H[8, 2, 24] = 1

    return {'W1':W1, 'W2':W2, 'H':H, 'T':T, 'F':F, 'K':K, 'M':M, 'N':N}

def testNMF2DConvJointSynthetic():
    initParallelAlgorithms()
    L = 200
    res = get2DSyntheticJointExample()
    [W1, W2, H, T, F, K] = \
        [res['W1'], res['W2'], res['H'], res['T'], res['F'], res['K']]
    A = multiplyConv2D(W1, H)
    Ap = multiplyConv2D(W2, H)
    doNMF2DConvJointGPU(A, Ap, K, T, F, L, doKL = False, plotfn=plotNMF2DConvSpectraJoint)
    #doNMF1DConv(V, K, T, L, plotfn=plotNMF1DConvSpectra)

def testNMF2DConvJoint3WaySynthetic():
    initParallelAlgorithms()
    np.random.seed(300)
    N2 = 40
    res = get2DSyntheticJointExample()
    [W1, W2, H1, T, F, K] = \
        [res['W1'], res['W2'], res['H'], res['T'], res['F'], res['K']]
    H2 = np.random.rand(F, K, N2)
    H2[H2 > 0.98] = 1
    H2[H2 < 1] = 0

    A = multiplyConv2D(W1, H1)
    Ap = multiplyConv2D(W2, H1)
    B = multiplyConv2D(W1, H2)

    doNMF2DConvJoint3WayGPU(A, Ap, B, K, T, F, 200, plotfn = plotNMF2DConvSpectraJoint3Way)

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
    """
    Trying out the technique in
    [1] "Multi-View Clustering via Joint Nonnegative Matrix Factorization"
        Jialu Liu, Chi Wang, Ning Gao, Jiawei Han
    """
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
    NIters = 100
    H = doNMF(S, 10, NIters, W=U1, plotfn = fn)
    SRes = U1.dot(H)
    XRes = griffinLimInverse(SRes, winSize, hopSize, NIters = 10)
    SResCover = U2.dot(H)
    XResCover = griffinLimInverse(SResCover, winSize, hopSize, NIters = 10)
    sio.wavfile.write("BadRes.wav", Fs, XRes)
    sio.wavfile.write("BadResCover.wav", Fs, XResCover)

def testNMFMusaicingSimple():
    """
    Try to replicate the results from the Driedger paper
    """
    import librosa
    winSize = 2048
    hopSize = 1024
    Fs = 22050

    X, Fs = librosa.load("music/Bees_Buzzing.mp3")
    WComplex = getPitchShiftedSpecs(X, Fs, winSize, hopSize, 6)
    W = np.abs(WComplex)
    X, Fs = librosa.load("music/Beatles_LetItBe.mp3")
    V = np.abs(STFT(X, winSize, hopSize))

    #librosa.display.specshow(librosa.amplitude_to_db(H), y_axis = 'log', x_axis = 'time')
    fn = lambda V, W, H, iter, errs: plotNMFSpectra(V, W, H, iter, errs, hopSize)
    NIters = 50
    #(W, H) = doNMF(V, W.shape[1], NIters, W=W, plotfn = fn)
    H = doNMFDriedger(V, W, NIters, r=7, p=10, c=6, plotfn=fn)
    H = np.array(H, dtype=np.complex)
    V2 = WComplex.dot(H)

    sio.savemat("V2.mat", {"V2":V2, "H":H})
    #V2 = sio.loadmat("V2.mat")["V2"]
    X = iSTFT(V2, winSize, hopSize)
    X = X/np.max(np.abs(X))
    wavfile.write("letitbeeISTFT.wav", Fs, X)

    print("Doing phase retrieval...")
    Y = griffinLimInverse(V2, winSize, hopSize, NIters=30)
    Y = Y/np.max(np.abs(Y))
    wavfile.write("letitbee.wav", Fs, Y)


def testHarmPercMusic():
    import librosa
    from scipy.io import wavfile
    import scipy.ndimage
    foldername = "HarmPerc"
    K = 2
    #STFT Params
    winSize = 2048
    hopSize = 256

    if not os.path.exists(foldername):
        os.mkdir(foldername)

    Fs, X = wavfile.read("music/SmoothCriminalAligned.wav")
    X = np.array(X, dtype=np.float32)
    A = X[:, 0]/(2.0**15)
    Ap = X[:, 1]/(2.0**15)
    #Take 20 seconds clips from each
    A = A[0:Fs*20]
    Ap = Ap[0:Fs*20]
    B, Fs = librosa.load("music/MJBad.mp3")
    B = B[Fs*3:Fs*23]
    #B, Fs = librosa.load("music/MJSpeedDemonClip.wav")

    SsA = []
    SsAp = []
    SsB = []
    for (V, Ss, s) in zip([A, Ap, B], [SsA, SsAp, SsB], ["A", "Ap", "B"]):
        S = STFT(V, winSize, hopSize)
        Harm, Perc = librosa.decompose.hpss(S)
        X1 = iSTFT(Harm, winSize, hopSize)
        X2 = iSTFT(Perc, winSize, hopSize)
        wavfile.write("%s/%s_0.wav"%(foldername, s), Fs, X1)
        wavfile.write("%s/%s_1.wav"%(foldername, s), Fs, X2)
        if s == "B":
            Ss.append(Harm)
            Ss.append(Perc)
        else:
            for Xk in [X1, X2]:
                Ss.append(getPitchShiftedSpecs(Xk, Fs, winSize, hopSize))          
        

    ##Do NMF Driedger on one track at a time
    fn = lambda V, W, H, iter, errs: plotNMFSpectra(V, W, H, iter, errs, hopSize)
    SFinal = np.zeros(SsB[0].shape, dtype = np.complex)
    print("SFinal.shape = ", SFinal.shape)
    for k in range(K):
        print("Doing Driedger on track %i..."%k)
        HFilename = "%s/DriedgerH%i.mat"%(foldername, k)
        if not os.path.exists(HFilename):
            H = doNMFDriedger(np.abs(SsB[k]), np.abs(SsA[k]), 100, \
            r = 7, p = 10, c = 3, plotfn = fn)
            sio.savemat(HFilename, {"H":H})
        else:
            H = sio.loadmat(HFilename)["H"]
        H = np.array(H, dtype=np.complex)
        S = SsA[k].dot(H)
        X = griffinLimInverse(S, winSize, hopSize)
        wavfile.write("%s/B%i_Driedger.wav"%(foldername, k), Fs, X)
        S = SsAp[k].dot(H)
        X = griffinLimInverse(S, winSize, hopSize)
        wavfile.write("%s/Bp%i.wav"%(foldername, k), Fs, X)
        SFinal += S

    ##Do Griffin Lim phase correction on the final mixed STFT
    X = griffinLimInverse(SFinal, winSize, hopSize)
    Y = X/np.max(np.abs(X))
    wavfile.write("%s/BpFinal.wav"%foldername, Fs, Y)


def testNMF1DMusic():
    import librosa
    from scipy.io import wavfile
    foldername = "1DNMFResults"
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    NIters = 80
    hopSize = 256
    winSize = 2048
    #Step 1: Do joint embedding on A and Ap
    K = 10
    T = 16
    Fs, X = sio.wavfile.read("music/SmoothCriminalAligned.wav")
    X1 = X[:, 0]/(2.0**15)
    X2 = X[:, 1]/(2.0**15)
    #Only take first 30 seconds for initial experiments
    X1 = X1[0:Fs*30]
    X2 = X2[0:Fs*30]
    #Load in B
    B, Fs = librosa.load("music/MJBad.mp3")
    B = B[Fs*3:Fs*23]

    S1 = STFT(X1, winSize, hopSize)
    N = S1.shape[0]
    S2 = STFT(X2, winSize, hopSize)
    SOrig = np.concatenate((S1, S2), 0)
    S = np.abs(SOrig)
    plotfn = lambda V, W, H, iter, errs: \
        plotNMF1DConvSpectraJoint(V, W, H, iter, errs, hopLength = hopSize, \
        audioParams = {'Fs':Fs, 'winSize':winSize, 'prefix':foldername})
    filename = "%s/NMFAAp.mat"%foldername
    if os.path.exists(filename):
        res = sio.loadmat(filename)
        [W, H] = [res['W'], res['H']]
    else:
        (W, H) = doNMF1DConvJoint(S, K, T, NIters, prefix=foldername, plotfn=plotfn)
        sio.savemat(filename, {"W":W, "H":H})
    W1 = W[:, 0:N, :]
    W2 = W[:, N::, :]
    S = multiplyConv1D(W, H)
    S1 = S[0:N, :]
    S2 = S[N::, :]

    y_hat = griffinLimInverse(S1, winSize, hopSize)
    y_hat = y_hat/np.max(np.abs(y_hat))
    sio.wavfile.write("%s/ANMF.wav"%foldername, Fs, y_hat)
    y_hat = griffinLimInverse(S2, winSize, hopSize)
    y_hat = y_hat/np.max(np.abs(y_hat))
    sio.wavfile.write("%s/ApNMF.wav"%foldername, Fs, y_hat)

    #Also invert each Wt
    for k in range(W.shape[2]):
        Wk = np.array(W[:, :, k].T)
        Wk1 = Wk[0:N, :]
        Wk2 = Wk[N::, :]
        y_hat = griffinLimInverse(Wk1, winSize, hopSize)
        y_hat = y_hat/np.max(np.abs(y_hat))
        sio.wavfile.write("%s/WA_%i.wav"%(foldername, k), Fs, y_hat)
        y_hat = griffinLimInverse(Wk2, winSize, hopSize)
        y_hat = y_hat/np.max(np.abs(y_hat))
        sio.wavfile.write("%s/WAp_%i.wav"%(foldername, k), Fs, y_hat)
    
    S1 = SOrig[0:N, :]
    S2 = SOrig[N::, :]
    (AllSsA, RatiosA) = getComplexNMF1DTemplates(S1, W1, H, p = 2, audioParams = {'winSize':winSize, \
        'hopSize':hopSize, 'Fs':Fs, 'fileprefix':"%s/TrackA"%foldername})
    (AllSsAp, RatiosAp) = getComplexNMF1DTemplates(S2, W2, H, p = 2, audioParams = {'winSize':winSize, \
        'hopSize':hopSize, 'Fs':Fs, 'fileprefix':"%s/TrackAp"%foldername})
    

    #Step 1a: Combine templates manually
    clusters = [[3], [5], [0, 1, 2, 4, 6, 7, 8, 9]]
    SsA = []
    SsAp = []
    for i, cluster in enumerate(clusters):
        SAi = np.zeros(S1.shape, dtype = np.complex)
        SApi = np.zeros(S2.shape, dtype = np.complex)
        for idx in cluster:
            SAi += AllSsA[idx]
            SApi += AllSsAp[idx]
        SsA.append(SAi)
        SsAp.append(SApi)
        y_hat = griffinLimInverse(SAi, winSize, hopSize)
        y_hat = y_hat/np.max(np.abs(y_hat))
        wavfile.write("%s/TrackAManual%i.wav"%(foldername, i), Fs, y_hat)
        y_hat = griffinLimInverse(SApi, winSize, hopSize)
        y_hat = y_hat/np.max(np.abs(y_hat))
        wavfile.write("%s/TrackApManual%i.wav"%(foldername, i), Fs, y_hat)
    
    #Step 2: Create a W matrix which is grouped by cluster and which has pitch shifted
    #versions of each template
    WB = np.array([])
    clusteridxs = [0]
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            thisW = W1[:, :, idx].T
            for shift in range(-6, 7):
                thisWShift = pitchShiftSTFT(thisW, Fs, shift).T[:, :, None]
                if WB.size == 0:
                    WB = thisWShift
                else:
                    WB = np.concatenate((WB, thisWShift), 2)
        clusteridxs.append(WB.shape[2])
    print("WB.shape = ", WB.shape)
    print("clusteridxs = ", clusteridxs)
    sio.savemat("%s/WB.mat"%foldername, {"WB":WB})

    #Step 3: Filter B by the new W matrix
    SBOrig = STFT(B, winSize, hopSize)
    plotfn = lambda V, W, H, iter, errs: \
        plotNMF1DConvSpectra(V, W, H, iter, errs, hopLength = hopSize)
    filename = "%s/NMFB.mat"%foldername
    if not os.path.exists(filename):
        (WB, HB) = doNMF1DConv(np.abs(SBOrig), WB.shape[2], T, NIters, W = WB)
        sio.savemat(filename, {"HB":HB, "WB":WB})
    else:
        HB = sio.loadmat(filename)["HB"]
    #Separate out B tracks
    As = []
    AsSum = np.zeros(SBOrig.shape)
    p = 2
    for i in range(len(clusters)):
        thisH = np.array(HB)
        thisH[0:clusteridxs[i], :] = 0
        thisH[clusteridxs[i+1]::, :] = 0
        As.append(multiplyConv1D(WB, thisH)**p)
        AsSum += As[-1]
    SsB = []
    for i in range(len(clusters)):
        SBi = SBOrig*As[i]/AsSum
        SsB.append(SBi)
        y_hat = griffinLimInverse(SBi, winSize, hopSize)
        y_hat = y_hat/np.max(np.abs(y_hat))
        wavfile.write("%s/TrackBManual%i.wav"%(foldername, i), Fs, y_hat)
        plt.clf()
        plt.plot(np.sum(As[i]**2, 0)/np.sum(AsSum**2, 0))
        plt.savefig("%s/TrackBManual%s.svg"%(foldername, i))

    #Step 4: Do NMF Driedger on one track of B at a time
    NIters = 100
    shiftrange = 6
    for i in range(len(SsA)):
        SsA[i] = getPitchShiftedSpecsFromSpec(SsA[i], Fs, winSize, hopSize, shiftrange=shiftrange)
        SsAp[i] = getPitchShiftedSpecsFromSpec(SsAp[i], Fs, winSize, hopSize, shiftrange=shiftrange)
    fn = lambda V, W, H, iter, errs: plotNMFSpectra(V, W, H, iter, errs, hopSize)
    XFinal = np.array([])
    for i in range(len(SsA)):
        print("Doing track %i..."%i)
        HFilename = "%s/H%i.mat"%(foldername, i)
        if not os.path.exists(HFilename):
            H = doNMFDriedger(np.abs(SsB[i]), np.abs(SsA[i]), NIters, \
            r = 7, p = 10, c = 3, plotfn = fn)
            sio.savemat(HFilename, {"H":H})
        else:
            H = sio.loadmat(HFilename)["H"]
        H = np.array(H, dtype=np.complex)
        S = SsA[i].dot(H)
        X = griffinLimInverse(S, winSize, hopSize)
        wavfile.write("%s/B%i_Driedger.wav"%(foldername, i), Fs, X)
        S = SsAp[i].dot(H)
        X = griffinLimInverse(S, winSize, hopSize)
        Y = X/np.max(np.abs(X))
        wavfile.write("%s/Bp%i.wav"%(foldername, i), Fs, Y)
        if XFinal.size == 0:
            XFinal = X
        else:
            XFinal += X
    Y = XFinal/np.max(np.abs(XFinal))
    wavfile.write("%s/BpFinal.wav"%foldername, Fs, Y)
    

def getMadmomTempo(filename):
    """
    Call Madmom Tempo Estimation
    :return: Array of tempos sorted in decreasing order of strength
    """
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.tempo import TempoEstimationProcessor
    act = RNNBeatProcessor()(filename)
    proc = TempoEstimationProcessor(fps=100)
    res = proc(act)
    return res[:, 0]

def getTempos(A, Ap, B, Fs):
    from scipy.io import wavfile
    tempos = {}
    for s, X in zip(("A", "Ap", "B"), (A, Ap, B)):
        wavfile.write("temp.wav", Fs, X)
        tempos[s] = 60.0/getMadmomTempo("temp.wav")
        print("%s: %s"%(s, tempos[s]))
    return tempos


def testNMF2DMusic(K, T, F, NIters = 300, bins_per_octave = 24, shiftrange = 6, \
                    ZoomFac = 8, Trial = 0, Joint3Way = False, \
                    W1Fixed = False, HFixed = False, doKL = False):
    """
    :param Joint3Way: If true, do a joint embedding with A, Ap, and B\
        If false, then do a joint embedding with (A, Ap) and represent\
        B in the A dictionary
    """
    import librosa
    from scipy.io import wavfile
    import pyrubberband as pyrb

    #Synthesizing AAF's "Bad"
    """
    Fs, X = wavfile.read("music/SmoothCriminalAligned.wav")
    X = np.array(X, dtype=np.float32)
    A = X[:, 0]/(2.0**15)
    Ap = X[:, 1]/(2.0**15)
    #Take 20 seconds clips from each
    A = A[0:Fs*20]
    Ap = Ap[0:Fs*20]
    B, Fs = librosa.load("music/MJBad.mp3")
    B = B[Fs*3:Fs*23]
    #A and A' tempos are from the synchronization code
    tempoA = 0.508 
    tempoAp = 0.472
    tempoB = 0.53

    songname = "mj"
    #A good separation I got before
    res = sio.loadmat("FinalExamples/MJAAF_Bad/Joint2DNMFFiltered_K3_Z4_T20_Bins24_F14_Trial2/NMF2DJoint.mat")
    W1 = res['W1']
    W2 = res['W2']
    H1 = res['H1']

    do2DFilteredAnalogy(A, Ap, B, Fs, K, T, F, NIters, bins_per_octave, shiftrange, \
        ZoomFac, Trial, Joint3Way, W1Fixed, HFixed, doKL, songname=songname, W1=W1, W2=W2, H1=H1)
    """

    #Synthesizing AAF's "Wanna Be Starting Something"
    """
    Fs, X = wavfile.read("music/SmoothCriminalAligned.wav")
    X = np.array(X, dtype=np.float32)
    A = X[:, 0]/(2.0**15)
    Ap = X[:, 1]/(2.0**15)
    #Take 20 seconds clips from each
    A = A[0:Fs*20]
    Ap = Ap[0:Fs*20]
    B, Fs = librosa.load("music/MJStartinSomething.mp3")
    
    #tempos = getTempos(A, Ap, B, Fs)
    tempoA = 0.508 
    tempoAp = 0.472
    tempoB = 0.49
    B = pyrb.time_stretch(B, Fs, tempoB/tempoA)
    B = B[0:Fs*20]
    
    songname = "wanna"
    res = sio.loadmat("FinalExamples/MJAAF_Bad/Joint2DNMFFiltered_K3_Z4_T20_Bins24_F14_Trial2/NMF2DJoint.mat")
    W1 = res['W1']
    W2 = res['W2']
    H1 = res['H1']
    res = do2DFilteredAnalogy(A, Ap, B, Fs, K, T, F, NIters, bins_per_octave, shiftrange, \
        ZoomFac, Trial, Joint3Way, W1Fixed, HFixed, doKL, songname=songname, W1=W1, W2=W2, H1=H1)
    Y = res['Y']
    foldername = res['foldername']
    Y = pyrb.time_stretch(Y, Fs, tempoA/tempoB)
    wavfile.write("%s/BpFinalStretched.wav"%foldername, Fs, Y)
    """

    #Synthesizing Marilyn Manson "Who's That Girl"
    Fs, X = wavfile.read("music/SweetDreams/SweetDreamsAlignedClip.wav")
    X = np.array(X, dtype=np.float32)
    A = X[:, 0]/(2.0**15)
    Ap = X[:, 1]/(2.0**15)
    #Take 20 seconds clips from each
    A = A[0:Fs*20]
    Ap = Ap[0:Fs*20]
    B, Fs = librosa.load("music/SweetDreams/WhosThatGirlClip.wav")
    B = B[0:Fs*20]

    tempoA = 0.477
    tempoB = 0.65
    songname = "eurythmics"
    res = do2DFilteredAnalogy(A, Ap, B, Fs, K, T, F, NIters, bins_per_octave, shiftrange, \
        ZoomFac, Trial, Joint3Way, W1Fixed, HFixed, doKL, songname=songname)
    Y = res['Y']
    foldername = res['foldername']
    Y = pyrb.time_stretch(Y, Fs, tempoA/tempoB)
    wavfile.write("%s/BpFinalStretched.wav"%foldername, Fs, Y)



def testMIDIExample(T, F, NIters = 300, bins_per_octave = 24, shiftrange = 6, \
                    ZoomFac = 8, Trial = 0, HFixed = False, doKL = True):
    import librosa
    from scipy.io import wavfile
    import pyrubberband as pyrb
    from CQT import getNSGT
    initParallelAlgorithms()
    path = "music/MIDIExample/BeeGeesTracks/"
    NTracks = 6
    W1 = np.array([])
    H1 = np.array([])
    startidx = 27648 #Where the synchronized path starts
    for track in range(NTracks):
        matfilename = "%s/WH%i_F%i_T%i_Z%i_Trial%i.mat"%(path, track+1, F, T, ZoomFac, Trial)
        if not os.path.exists(matfilename):
            X, Fs = librosa.load("%s/%i.mp3"%(path, track+1))
            X = X[startidx:startidx+Fs*10]
            wavfile.write("Track%i.wav"%track, Fs, X)
            print("Doing CQT of track %i..."%track)
            C0 = getNSGT(X, Fs, bins_per_octave)
            #Zeropad to nearest even factor of the zoom factor
            NRound = ZoomFac*int(np.ceil(C0.shape[1]/float(ZoomFac)))
            C = np.zeros((C0.shape[0], NRound), dtype = np.complex)
            C[:, 0:C0.shape[1]] = C0
            C = np.abs(C)
            C = scipy.ndimage.interpolation.zoom(C, (1, 1.0/ZoomFac))
            plotfn = lambda V, W, H, iter, errs: plotNMF2DConvSpectra(V, W, H, iter, errs, hopLength = 128)
            (Wi, Hi) = doNMF2DConvGPU(C, 1, T, F, L=100, doKL = doKL, plotfn = plotfn, plotInterval=400)
            sio.savemat(matfilename, {"W":Wi, "H":Hi})
        else:
            res = sio.loadmat(matfilename)
            Wi = res["W"]
            Hi = res["H"]
        if W1.size == 0:
            W1 = np.zeros((T, Wi.shape[1], NTracks))
            H1 = np.zeros((F, NTracks, Hi.shape[2]))
        Wi = np.reshape(Wi, [Wi.shape[0], Wi.shape[1]])
        Hi = np.reshape(Hi, [Hi.shape[0], Hi.shape[2]])
        W1[:, :, track] = Wi
        H1[:, track, :] = Hi

    K = NTracks
    Fs, X = wavfile.read("music/MIDIExample/stayinalivesyncedclip.wav")
    X = np.array(X, dtype=np.float32)
    A = X[:, 0]/(2.0**15)
    Ap = X[:, 1]/(2.0**15)
    #Take 10 seconds clips from each
    A = A[0:Fs*10]
    Ap = Ap[0:Fs*10]
    B, Fs = librosa.load("music/MIDIExample/TupacMIDIClip.mp3")
    tempoA = 0.578
    tempoB = 0.71
    B = pyrb.time_stretch(B, Fs, tempoB/tempoA)
    wavfile.write("BStretched.wav", Fs, B)
    B = B[0:Fs*10]

    songname = "madatchya"
    if not HFixed:
        H1 = np.array([])
    res = do2DFilteredAnalogy(A, Ap, B, Fs, K, T, F, NIters, bins_per_octave, shiftrange, \
        ZoomFac, Trial, False, W1Fixed=True, HFixed=HFixed, doKL = doKL, W1 = W1, H1=H1, songname=songname)
    Y = res['Y']
    foldername = res['foldername']
    Y = pyrb.time_stretch(Y, Fs, tempoA/tempoB)
    wavfile.write("%s/BpFinalStretched.wav"%foldername, Fs, Y)

    
def doTrials():
    NTrials = 4
    T = 20
    for K in [2]:
        for ZoomFac in [4]:
            for Trial in range(NTrials):
                for Joint3Way in [True, False]:
                    for doKL in [True]:
                        for W1Fixed in [True, False]:
                            testNMF2DMusic(K = K, T = T, F = 14, ZoomFac = ZoomFac, \
                                    Trial = Trial, Joint3Way = Joint3Way, \
                                    doKL = doKL, W1Fixed = W1Fixed, bins_per_octave=12)

def doTrialsMIDI():
    NTrials = 5
    T = 20
    for ZoomFac in [4]:
        for Trial in range(NTrials):
            for F in [14]:
                testMIDIExample(T, F, NIters = 300, bins_per_octave = 24, shiftrange = 6, \
                            ZoomFac = ZoomFac, Trial = Trial, doKL = True)

if __name__ == '__main__':
    #testNMFMusaicingSimple()
    #testNMFJointSynthetic()
    #testNMFJointSmoothCriminal()
    #testNMF1DConvSynthetic()
    #testNMF2DConvSynthetic()
    #testNMF2DConvJointSynthetic()
    #testNMF2DConvJoint3WaySynthetic()
    #testHarmPercMusic()
    #testNMF1DMusic()
    testNMF2DMusic(K = 3, T = 20, F = 14, bins_per_octave = 24, ZoomFac = 4, \
                    Joint3Way = False, W1Fixed = True, HFixed = False, doKL = True, Trial=0)
    #doTrials()
    #testMIDIExample(T=20, F=14, ZoomFac=2)
    #doTrialsMIDI()