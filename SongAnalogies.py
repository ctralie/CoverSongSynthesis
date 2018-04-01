import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
from scipy.io import wavfile
import scipy.io as sio
import pyrubberband as pyrb
from Synchronize import *
from GeometricCoverSongs.CSMSSMTools import *
from GeometricCoverSongs.pyMIRBasic.Chroma import *
from GeometricCoverSongs.pyMIRBasic.MFCC import *
from SpectrogramTools import *
from NMFGPU import *
from CQT import *

def getNormedFeatures(S, winSize, hopSize):
    """
    
    """
    Norms = np.sqrt(np.sum(X**2, 1))
    Norms[Norms == 0] = 1
    return X/Norms[:, None]

def doSpectrogramAnalogiesSimple(SA, SAp, SB, Kappa):
    #Compute normalized spectrogram similarities for each window
    SBp = np.zeros_like(SB)
    MagSA = np.sqrt(np.abs(np.sum(SA*np.conj(SA), 0)))
    MagSA[MagSA == 0] = 1
    MagSB = np.sqrt(np.abs(np.sum(SB*np.conj(SB), 0)))
    MagSB[MagSB == 0] = 1
    SA = SA/MagSA[None, :]
    SB = SB/MagSB[None, :]
    SAM = np.abs(SA)
    for k in range(SB.shape[1]):
        x = np.abs(SB[:, k].flatten())
        D = getCSM(x[None, :], SAM.T).flatten()
        idx = np.argmin(D)
        dist = D[idx]
        if k > 0:
            if D[idx-1] < (1+Kappa)*dist:
                idx = idx-1
                dist = D[idx]
        #Scale by power ratio
        SBp[:, k] = SAp[:, idx]*MagSB[k]/MagSA[idx]
    return SBp

def doSpectrogramAnalogies(SA, SAp, SB, T, Fs, useMFCCs = True):
    N = SA.shape[1]
    M = N-T+1
    if useMFCCs:
        MA = getMFCCsFromSpec(SA, Fs)
        MAp = getMFCCsFromSpec(SAp, Fs)
        MB = getMFCCsFromSpec(SB, Fs)
    else:
        (MA, MAp, MB) = (SA, SAp, SB)
    #Step 1: Make sliding windows for A and Ap
    SWs = []
    for i in range(M):
        S = np.concatenate((MA[:, i:i+T], MAp[:, i:i+T-1]), 1)
        S = S.flatten()
        S = S/np.sqrt(np.sum(S**2))
        SWs.append(S)
    SWs = np.array(SWs)
    #Step 2: Step through and fill in windows for SB
    N = SB.shape[1]
    M = N-T+1
    SBp = np.zeros_like(SB)
    MBp = np.zeros_like(MB)
    #Fill in some stuff at the beginning
    SBp[:, 0:T-1] = SAp[:, 0:T-1]
    MBp[:, 0:T-1] = MAp[:, 0:T-1]
    idxs = []
    for i in range(T-1, M):
        S = np.concatenate((MB[:, i-T+1:i+1], MBp[:, i-T+1:i]), 1)
        S = S.flatten()
        S = S/np.sqrt(np.sum(S**2))
        D = getCSM(S[None, :], SWs)
        idx = np.argmin(D.flatten())
        idxs.append(idx)
        s = SAp[:, idx+T-1]
        SBp[:, i] = s
        MBp[:, i] = getMFCCsFromSpec(s[:, None], Fs).flatten()
    idxs = np.array(idxs)
    return SBp

def makeAnalogy(X, Fs, beatsA, filename_b, hopSize, winSize, ws, TempoBias, MFCCWeight = 1.0, HPCPWeight = 1.0):
    """
    Make a cover song analogy; given audio for (A, A'), and B, \
        make B'
    :param X: Audio waveform for A and A'; A is in first column, A' in second
    :param Fs: Sample rate of all audio files
    :param beatsA: Beat onsets (in samples)
    :param hopSize: Feature hop size
    :param winSize: Window size for MFCCs and HPCPs
    :param ws: Window weights of all features
    :param TempoBias: Tempo bias for beat tracking of song B
    """
    #Step 1: Load in new example from artist 1 (B song)
    print("Loading new example...")
    XA = X[:, 0]
    XAp = X[:, 1]
    XB, Fs2 = librosa.load(filename_b)
    XB = librosa.core.to_mono(XB)

    #Step 2: Use rubberband library to change tempo of B so that
    #it's in line with tempo of song A
    tempoB, beatsB = librosa.beat.beat_track(XB, Fs2, start_bpm = TempoBias, hop_length = hopSize)
    tempoA = 60.0/(np.mean(beatsA[1::] - beatsA[0:-1])/float(Fs))
    print("tempoA = %g, tempoB = %g"%(tempoA, tempoB))
    ratio = float(tempoA)/tempoB
    print("Shifting by ratio: %g"%ratio)
    XB = pyrb.time_stretch(XB, Fs2, ratio)

def do2DFilteredAnalogy(A, Ap, B, Fs, K, T, F, NIters = 300, bins_per_octave = 24, \
                    shiftrange = 6, ZoomFac = 8, Trial = 0, Joint3Way = False, \
                    W1Fixed = False, HFixed = False, doKL = False, songname = "", 
                    W1 = np.array([]), W2 = np.array([]), H1 = np.array([])):
    """
    :param Joint3Way: If true, do a joint embedding with A, Ap, and B\
        If false, then do a joint embedding with (A, Ap) and represent\
        B in the A dictionary
    """
    from scipy.io import wavfile
    import scipy.ndimage
    initParallelAlgorithms()
    doInitialInversions = False

    #STFT Parameters
    hopSize = 256
    winSize = 2048

    ## Step 1: Compute CQTs
    XSizes = {}
    librosahopSize = int(np.round(Fs/100.0)) #For librosa display to know approximate timescale

    resOrig = {}
    res = {}
    for (V, s) in zip([A, Ap, B], ["A", "Ap", "B"]):
        print("Doing CQT of %s..."%s)
        C0 = getNSGT(V, Fs, bins_per_octave)
        print("%s.shape = "%s, C0.shape)
        #Zeropad to nearest even factor of the zoom factor
        NRound = ZoomFac*int(np.ceil(C0.shape[1]/float(ZoomFac)))
        C = np.zeros((C0.shape[0], NRound), dtype = np.complex)
        C[:, 0:C0.shape[1]] = C0
        resOrig[s] = C
        C = np.abs(C)
        C = scipy.ndimage.interpolation.zoom(C, (1, 1.0/ZoomFac))
        XSizes[s] = V.size
        res[s] = C
        if doInitialInversions:
            CZoom = scipy.ndimage.interpolation.zoom(C, (1, ZoomFac))
            y_hat = getiNSGTGriffinLim(CZoom, V.size, Fs, bins_per_octave, \
                                                NIters = 100, randPhase = True)
            sio.wavfile.write("%sGTInverted.wav"%s, Fs, y_hat)
    XSizes["Bp"] = XSizes["B"]
    print(XSizes)

    [CAOrig, CApOrig, CBOrig] = [resOrig['A'], resOrig['Ap'], resOrig['B']]
    [CA, CAp, CB] = [res['A'], res['Ap'], res['B']]
    sio.savemat("Cs.mat", {"CA":CA, "CAp":CAp, "CB":CB})
    #Compute length of convolutional window in milliseconds
    t = 1000.0*(XSizes['A']/float(Fs))*float(T)/CA.shape[1]
    print("Convolutional window size: %.3g milliseconds"%t)

    audioParams={'Fs':Fs, 'bins_per_octave':bins_per_octave, \
                'prefix':'', 'XSizes':XSizes, "ZoomFac":ZoomFac}
    audioParams = None
    ## Step 2: Compute joint NMF
    if Joint3Way:
        #Do Joint 3 Way 2DNMF
        foldername = "%s_Joint2DNMFFiltered3Way_K%i_Z%i_T%i_Bins%i_F%i_Trial%i_KL%i"%\
                        (songname, K, ZoomFac, T, bins_per_octave, F, Trial, doKL)
        filename = "%s/NMF2DJoint.mat"%foldername
        if not os.path.exists(foldername):
            os.mkdir(foldername)
        if not os.path.exists(filename):
            plotfn = lambda A, Ap, B, W1, W2, H1, H2, iter, errs, foldername: \
                plotNMF2DConvSpectraJoint3Way(A, Ap, B, W1, W2, H1, H2, iter, errs,\
                foldername, hopLength = librosahopSize, audioParams=audioParams, useGPU = True)
            (W1, W2, H1, H2) = doNMF2DConvJoint3WayGPU(CA, CAp, CB, K, T, F, L=NIters, \
                doKL = doKL, plotfn=plotfn, plotInterval = NIters*2, foldername = foldername)
            sio.savemat(filename, {"W1":W1, "W2":W2, "H1":H1, "H2":H2})
        else:
            res = sio.loadmat(filename)
            [W1, W2, H1, H2] = [res['W1'], res['W2'], res['H1'], res['H2']]
    else:
        #Do 2DNMF jointly on A and Ap, then filter B by A's dictionary
        foldername = "%s_Joint2DNMFFiltered_K%i_Z%i_T%i_Bins%i_F%i_Trial%i_KL%i"%\
                        (songname, K, ZoomFac, T, bins_per_octave, F, Trial, doKL)
        if HFixed:
            foldername += "_HFixed"
        if W1Fixed:
            foldername += "_W1Fixed"
        filename = "%s/NMF2DJoint.mat"%foldername
        if not os.path.exists(foldername):
            os.mkdir(foldername)
        plotfn = lambda A, Ap, W1, W2, H, iter, errs, foldername: \
            plotNMF2DConvSpectraJoint(A, Ap, W1, W2, H, iter, errs, \
            foldername, hopLength = librosahopSize, audioParams = audioParams)
        if not os.path.exists(filename):
            if H1.size == 0 or W1.size == 0 or W2.size == 0:
                if W1Fixed or HFixed:
                    #Learn W1 and H on the first song only first, then use that
                    #on the second song
                    if W1.size == 0:
                        (W1, H1) = doNMF2DConvGPU(CA, K, T, F, L=NIters, doKL = doKL)
                        if not W1Fixed:
                            W1 = np.array([])
                        if not HFixed:
                            H1 = np.array([])
                (W1, W2, H1) = doNMF2DConvJointGPU(CA, CAp, K, T, F, L=NIters, \
                                                    W1 = W1, H = H1, doKL = doKL,\
                                                    plotfn = plotfn, plotInterval=NIters*2, \
                                                    foldername = foldername)
            #Represent B in the dictionary of A
            plotfn2 = lambda V, W, H, iter, errs: \
                plotNMF2DConvSpectra(V, W, H, iter, errs, hopLength = librosahopSize)
            (W, H2) = doNMF2DConvGPU(CB, K, T, F, W=W1, L=NIters, doKL = doKL, \
                                    plotfn=plotfn2, plotInterval=NIters*2)
            sio.savemat(filename, {"W1":W1, "W2":W2, "H1":H1, "H2":H2})
        else:
            res = sio.loadmat(filename)
            [W1, W2, H1, H2] = [res['W1'], res['W2'], res['H1'], res['H2']]
            plotfn(CA, CAp, W1, W2, H1, 0, np.array([[1, 1]]), ".")
    
    ## Step 3: Compute components and pitch shifted dictionaries
    (CsA, RatiosA) = getComplexNMF2DTemplates(CAOrig, W1, H1, ZoomFac, p = 2)
    (CsAp, RatiosAp) = getComplexNMF2DTemplates(CApOrig, W2, H1, ZoomFac, p = 2)
    (CsB, RatiosB) = getComplexNMF2DTemplates(CBOrig, W1, H2, ZoomFac, p = 2)
    (SsA, SsAp, SsB) = ([], [], [])
    plt.figure(figsize=(20, 3))
    for k, (CAk, CApk, CBk) in enumerate(zip(CsA, CsAp, CsB)):
        for s1, Ss, Ck, Ratios in zip(["A", "Ap", "B"], (SsA, SsAp, SsB),\
                 (CAk, CApk, CBk), (RatiosA[k], RatiosAp[k], RatiosB[k])):
            #First do phase correction and save result to disk
            Xk = getiNSGT(Ck, XSizes[s1], Fs, bins_per_octave)
            wavfile.write("%s/%s%i_iCQT.wav"%(foldername, s1, k), Fs, Xk)
            Xk = Xk.flatten()
            plt.clf()
            plt.plot(Ratios)
            plt.xlim([0, len(Ratios)])
            plt.title("Ratio, %.3g Above 0.1"%(np.sum(Ratios[k] > 0.1)/Ratios[k].size))
            plt.savefig("%s/%s_%iPower.svg"%(foldername, s1, k), bbox_inches = 'tight')
            if s1 in ["A", "Ap"]:
                #Make pitch shifted templates for A and A'
                Ss.append(getPitchShiftedSpecs(Xk, Fs, winSize, hopSize, shiftrange))
            else:
                Ss.append(STFT(Xk, winSize, hopSize))
    
    ## Step 4: Do NMF STFT on one track at a time
    fn = lambda V, W, H, iter, errs: plotNMFSpectra(V, W, H, iter, errs, librosahopSize)
    SFinal = np.zeros(SsB[0].shape, dtype = np.complex)
    PowerRatios = []
    for k in range(K):
        print("Doing Driedger on track %i..."%k)
        HFilename = "%s/DriedgerH%i.mat"%(foldername, k)
        if not os.path.exists(HFilename):
            H = doNMFDriedger(np.abs(SsB[k]), np.abs(SsA[k]), 100, r = 7, p = 10, c = 3)
            sio.savemat(HFilename, {"H":H})
        else:
            H = sio.loadmat(HFilename)['H']
        #First invert the translation STFT
        SB = SsA[k].dot(H)
        SBp = SsAp[k].dot(H)
        PowerRatio =  np.sqrt(np.sum(SsA[k]*np.conj(SsA[k])))
        PowerRatio /= np.sqrt(np.sum(SsAp[k]*np.conj(SsAp[k])))
        PowerRatios.append(np.abs(PowerRatio))
        print("PowerRatio = %.3g"%np.abs(PowerRatio))
        SFinal += PowerRatio*SBp
        XB = griffinLimInverse(SB, winSize, hopSize)
        XBp = griffinLimInverse(SBp, winSize, hopSize)
        wavfile.write("%s/B%i_DriedgerSTFT.wav"%(foldername, k), Fs, XB)
        wavfile.write("%s/Bp%i_Translated.wav"%(foldername, k), Fs, XBp)
    sio.savemat("%s/PowerRatios.mat"%foldername, {"PowerRatios":PowerRatios})
    ## Step 5: Do Griffin Lim phase correction on the final mixed STFTs
    X = griffinLimInverse(SFinal, winSize, hopSize)
    Y = X/np.max(np.abs(X))
    wavfile.write("%s/BpFinalSTFT_Translated.wav"%foldername, Fs, Y)
    return {'Y':Y, 'foldername':foldername}

if __name__ == '__main__':
    #Sync parameters
    filename1 = "music/SmoothCriminalMJ.mp3"
    filename2 = "music/SmoothCriminalAAF.mp3"
    artist1 = "Michael Jackson"
    artist2 = "Alien Ant Farm"
    jointfileprefix = "SmoothCriminalSync"
    filename_b = "music/MJBad.mp3"
    TempoBiases = [180]
    bSub = 1
    Kappa = 0.1
    FeatureParams = {'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':200, 'DPixels':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}

    #Analogy parameters
    ws = np.ones(1)
    hopSize = 512
    #TODO: Tweak window size
    #(tradeoff for being localized in time but also not sensitive to window position)
    winSize = hopSize*4

    #Step 1: Load in and synchronize example pair
    print("Loading and synchronizing example pair...")
    if os.path.exists("%s.mat"%jointfileprefix):
        res = sio.loadmat("%s.mat"%jointfileprefix)
        [X, Fs, beats] = [res['X'], res['Fs'], res['beatsFinal']]
    else:
        res = synchronize(filename1, filename2, hopSize, TempoBiases, bSub, FeatureParams, CSMTypes, Kappa)
        [X, Fs, beats] = [res['X'], res['Fs'], res['beatsFinal']]
        sio.savemat("%s.mat"%jointfileprefix, res)
        sio.wavfile.write("%s.wav"%jointfileprefix, Fs, X)
    beats = beats.flatten()
    #X = np.array(X, dtype = np.single)

    makeAnalogy(X, Fs, beats, filename_b, hopSize, winSize, ws, TempoBiases[0])
