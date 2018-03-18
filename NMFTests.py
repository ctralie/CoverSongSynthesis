import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from SpectrogramTools import *
from CQT import *
from NMF import *
from NMFGPU import *
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
    doNMF2DConvJoint(A, Ap, K, T, F, L, plotfn=plotNMF2DConvSpectraJoint)
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
    Try to replicate the results from the Dreidger paper
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
    H = doNMFDreidger(V, W, NIters, r=7, p=10, c=6, plotfn=fn)
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

def testNMF1DMusic():
    import librosa
    from scipy.io import wavfile
    foldername = "1DNMFResults"
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    K = 5
    T = 16
    NIters = 80
    hopSize = 256
    winSize = 2048

    #Step 1: Do joint embedding on A and Ap
    Fs, X = sio.wavfile.read("music/SmoothCriminalAligned.wav")
    X1 = X[:, 0]/(2.0**15)
    X2 = X[:, 1]/(2.0**15)
    #Only take first 30 seconds for initial experiments
    X1 = X1[0:Fs*30]
    X2 = X2[0:Fs*30]
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
    (SsA, RatiosA) = getComplexNMF1DTemplates(S1, W1, H, p = 2, audioParams = {'winSize':winSize, \
        'hopSize':hopSize, 'Fs':Fs, 'fileprefix':"%s/TrackA"%foldername})
    (SsAp, RatiosAp) = getComplexNMF1DTemplates(S2, W2, H, p = 2, audioParams = {'winSize':winSize, \
        'hopSize':hopSize, 'Fs':Fs, 'fileprefix':"%s/TrackAp"%foldername})
    

    #Step 2: Decompose B into components
    B, Fs = librosa.load("music/MJBad.mp3")
    B = B[Fs*3:Fs*23]
    SBOrig = STFT(B, winSize, hopSize)
    plotfn = lambda V, W, H, iter, errs: \
        plotNMF1DConvSpectra(V, W, H, iter, errs, hopLength = hopSize)
    filename = "%s/NMFB.mat"%foldername
    if not os.path.exists(filename):
        (WB, HB) = doNMF1DConv(np.abs(SBOrig), K, T, NIters)
        sio.savemat(filename, {"WB":WB, "HB":HB})
    else:
        res = sio.loadmat(filename)
        [WB, HB] = [res['WB'], res['HB']]
    (SsB, RatiosB) = getComplexNMF1DTemplates(SBOrig, WB, HB, p = 2, audioParams = {'winSize':winSize, \
        'hopSize':hopSize, 'Fs':Fs, 'fileprefix':"%s/TrackB"%foldername})
    

    #Step 3: Pitch shift each spectral template to create dictionary
    SADict = np.array([])
    SApDict = np.array([])
    Gap = np.zeros((N, 20), dtype=np.complex)
    for i, (SA, SAp, Ratios) in enumerate(zip(SsA, SsAp, RatiosA)):
        print("Making %ith component of dictionary"%(i+1))
        plt.clf()
        (SA, SAp) = ThresholdSpecs(SA, SAp, Ratios, 0.05, gap = 10, debugPlot = True)
        plt.savefig("%s/TrackA_%iPower.svg"%(foldername, i), bbox_inches = 'tight')
        DictAi = getPitchShiftedSpecsFromSpec(SA, Fs, winSize, hopSize, shiftrange = 3)
        DictApi = getPitchShiftedSpecsFromSpec(SAp, Fs, winSize, hopSize, shiftrange = 3)
        if SADict.size == 0:
            SADict = DictAi
            SApDict = DictApi
        else:
            SADict = np.concatenate((SADict, DictAi, Gap), 1)
            SApDict = np.concatenate((SApDict, DictApi, Gap), 1)
        print("SADict.shape = ", SADict.shape)

    #Step 4: Do NMF Dreidger on one track of B at a time
    fn = lambda V, W, H, iter, errs: plotNMFSpectra(V, W, H, iter, errs, hopSize)
    XFinal = np.array([])
    for i, SB in enumerate(SsB):
        print("Doing track %i..."%i)
        HFilename = "%s/H%i.mat"%(foldername, i)
        if not os.path.exists(HFilename):
            H = doNMFDreidger(np.abs(SB), np.abs(SADict), NIters, \
            r = 7, p = 3, c = 3, plotfn = fn)
            sio.savemat("%s/H%i.mat"%(foldername, i), {"H":H})
        else:
            H = sio.loadmat(HFilename)["H"]
        H = np.array(H, dtype=np.complex)
        S = SADict.dot(H)
        X = griffinLimInverse(S, winSize, hopSize)
        wavfile.write("%s/B%i_Dreidger.wav"%(foldername, i), Fs, X)
        S = SApDict.dot(H)
        X = griffinLimInverse(S, winSize, hopSize)
        Y = X/np.max(np.abs(X))
        wavfile.write("%s/Bp%i.wav"%(foldername, i), Fs, Y)
        if XFinal.size == 0:
            XFinal = X
        else:
            XFinal += X
        Y = XFinal/np.max(np.abs(XFinal))
        wavfile.write("%s/BpFinal.wav"%foldername, Fs, Y)
    


def testNMF2DMusic(Joint3Way = False):
    """
    :param Joint3Way: If true, do a joint embedding with A, Ap, and B\
        If false, then do a joint embedding with (A, Ap) and represent\
        B in the A dictionary
    """
    import librosa
    from scipy.io import wavfile
    import scipy.ndimage
    initParallelAlgorithms()
    eng = initMatlabEngine()
    doInitialInversions = False

    Fs, X = sio.wavfile.read("music/SmoothCriminalAligned.wav")
    X = np.array(X, dtype=np.float32)
    A = X[:, 0]/(2.0**15)
    Ap = X[:, 1]/(2.0**15)
    #Only take 15 seconds for initial experiments
    A = A[5*Fs:Fs*25]
    Ap = Ap[5*Fs:Fs*25]

    B, Fs = librosa.load("music/MJBad.mp3")
    B = B[Fs*3:Fs*23]

    XSizes = {}

    bins_per_octave = 24
    hopSize = int(np.round(Fs/100.0)) #For librosa display to know approximate timescale
    ZoomFac = 8 #Scaling factor so that each window is approximately 10ms
    K = 2
    T = 24
    F = 36
    NIters = 440

    resOrig = {}
    res = {}
    for (V, s) in zip([A, Ap, B], ["A", "Ap", "B"]):
        print("Doing CQT of %s..."%s)
        C = getCQTNakamuraMatlab(eng, V, Fs, bins_per_octave )
        resOrig[s] = C
        C = np.abs(C)
        C = scipy.ndimage.interpolation.zoom(C, (1, 1.0/ZoomFac))
        XSizes[s] = V.size
        res[s] = C
        if doInitialInversions:
            CZoom = scipy.ndimage.interpolation.zoom(C, (1, ZoomFac))
            (y_hat, spec) = getiCQTGriffinLimNakamuraMatlab(eng, CZoom, V.size, Fs, bins_per_octave, \
                NIters=100, randPhase = True)
            sio.wavfile.write("%sGTInverted.wav"%s, Fs, y_hat)
    XSizes["Bp"] = XSizes["B"]
    print(XSizes)
    [CAOrig, CApOrig, CBOrig] = [resOrig['A'], resOrig['Ap'], resOrig['B']]
    [CA, CAp, CB] = [res['A'], res['Ap'], res['B']]

    audioParams={'Fs':Fs, 'bins_per_octave':bins_per_octave, \
                'prefix':'', 'eng':eng, 'XSizes':XSizes, "ZoomFac":ZoomFac}

    if Joint3Way:
        #Do joint 2DNMF
        plotfn = lambda A, Ap, B, W1, W2, H1, H2, iter, errs: \
            plotNMF2DConvSpectraJoint3Way(A, Ap, B, W1, W2, H1, H2, iter, errs,\
            hopLength = hopSize, audioParams=audioParams, useGPU = True)
        (W1, W2, H1, H2) = doNMF2DConvJoint3WayGPU(CA, CAp, CB, K, T, F, L=NIters, \
            plotfn=plotfn, plotInterval = NIters*2)
        sio.savemat("SmoothCriminalAllNMF2DJoint.mat", {"W1":W1, "W2":W2, "H1":H1, "H2":H2})
        #res = sio.loadmat("SmoothCriminalAllNMF2DJoint.mat")
        #[W1, W2, H1, H2] = [res['W1'], res['W2'], res['H1'], res['H2']]
        
        #Output filtered sounds
        foldername = "AllJoint2DNMFFiltered"
    else:
        #Do 2DNMF on a joint embedding of A and Ap
        N = CA.shape[0]
        C = np.zeros((N*2, CA.shape[1]))
        C[0:N, :] = CA
        C[N::, :] = CAp
        plotfn = lambda V, W, H, iter, errs: \
            plotNMF2DConvSpectraJoint(V, W, H, iter, errs, \
            hopLength = hopSize, audioParams = audioParams)
        (W, H1) = doNMF2DConvGPU(C, K, T, F, L=NIters, plotfn = plotfn, \
                                joint = True, plotInterval=NIters*2)
        #Represent B in the dictionary of A
        W1 = W[:, 0:N, :]
        W2 = W[:, 0:N, :]
        plotfn = lambda V, W, H, iter, errs: \
            plotNMF2DConvSpectra(V, W, H, iter, errs, hopLength = hopSize)
        (W, H2) = doNMF2DConvGPU(CB, K, T, F, W=W1, L=NIters, \
                                plotfn=plotfn, plotInterval=NIters*2)
        sio.savemat("SmoothCriminalNMF2DJoint.mat", {"W1":W1, "W2":W2, "H1":H1, "H2":H2})
        #res = sio.loadmat("SmoothCriminalNMF2DJoint.mat")
        #[W1, W2, H1, H2] = [res['W1'], res['W2'], res['H1'], res['H2']]

        foldername = "Joint2DNMFFiltered"
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    audioParams['prefix'] = "%s/A"%foldername
    audioParams['XSize'] = XSizes['A']
    getComplexNMF2DTemplates(CAOrig, W1, H1, ZoomFac, p = 2, audioParams = audioParams)
    audioParams['prefix'] = "%s/Ap"%foldername
    audioParams['XSize'] = XSizes['Ap']
    getComplexNMF2DTemplates(CApOrig, W2, H1, ZoomFac, p = 2, audioParams = audioParams)
    audioParams['prefix'] = "%s/B"%foldername
    audioParams['XSize'] = XSizes['B']
    getComplexNMF2DTemplates(CBOrig, W1, H2, ZoomFac, p = 2, audioParams = audioParams)

def testDreidgerTranslate():
    import librosa
    from scipy.io import wavfile
    hopSize = 256
    winSize = 2048
    NIters = 100
    shiftrange = 6

    #Step 1: Load in A, Ap, and B
    SsA = []
    SsB = []
    SsAp = []
    for i in range(2):
        print("Loading i = %i"%i)
        Fs, A = sio.wavfile.read("Example/A_%i.wav"%i)
        SsA.append(getPitchShiftedSpecs(A, Fs, winSize, hopSize, shiftrange=shiftrange))
        Fs, B = sio.wavfile.read("Example/B_%i.wav"%i)
        SsB.append(STFT(B, winSize, hopSize))
        Fs, Ap = sio.wavfile.read("Example/Ap_%i.wav"%i)
        SsAp.append(getPitchShiftedSpecs(Ap, Fs, winSize, hopSize, shiftrange=shiftrange))
    
    #Step 2: Do NMF Dreidger on one track at a time
    fn = lambda V, W, H, iter, errs: plotNMFSpectra(V, W, H, iter, errs, hopSize)
    XFinal = np.array([])
    for i in range(2):
        print("Doing track %i..."%i)
        HFilename = "Example/H%i.mat"%i
        if not os.path.exists(HFilename):
            H = doNMFDreidger(np.abs(SsB[i]), np.abs(SsA[i]), NIters, \
            r = 7, p = 10, c = 0, plotfn = fn)
            sio.savemat("Example/H%i.mat"%i, {"H":H})
        else:
            H = sio.loadmat(HFilename)["H"]
        H = np.array(H, dtype=np.complex)
        S = SsA[i].dot(H)
        X = griffinLimInverse(S, winSize, hopSize)
        wavfile.write("Example/B%i_Dreidger.wav"%i, Fs, X)
        S = SsAp[i].dot(H)
        X = griffinLimInverse(S, winSize, hopSize)
        wavfile.write("Example/Bp%i.wav"%i, Fs, X)
        if XFinal.size == 0:
            XFinal = X
        else:
            XFinal += X
    wavfile.write("Example/BpFinal.wav", Fs, XFinal)
    

    


if __name__ == '__main__':
    #testNMFMusaicingSimple()
    #testNMFJointSynthetic()
    #testNMFJointSmoothCriminal()
    #testNMF1DConvSynthetic()
    #testNMF2DConvSynthetic()
    testNMF2DConvJointSynthetic()
    #testNMF2DConvJoint3WaySynthetic()
    #testNMF1DMusic()
    #testNMF2DMusic(Joint3Way = False)
    #testDreidgerTranslate()