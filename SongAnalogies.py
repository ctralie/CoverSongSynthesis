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

def getNormedFeatures(X):
    Norms = np.sqrt(np.sum(X**2, 1))
    Norms[Norms == 0] = 1
    return X/Norms[:, None]

def makeAnalogy(X, Fs, beatsA, filename_b, hopSize, winSize, ws, TempoBias, MFCCWeight = 1.0, HPCPWeight = 1.0):
    """
    Make a cover song analogy; given audio for (A, A'), and B, \
        make B'
    :param X: Audio waveform for A and A'; A is in first column, A' in second
    :param Fs: Sample rate of all audio files
    :param beatsA: Beat onsets (in sampl)
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

    #Step 3: Compute all features and make sliding window
    #Compute normalized HPCP features for all three songs
    print("Getting HPCP features...")
    HA = getNormedFeatures(getHPCPEssentia(XA, Fs, winSize, hopSize, NChromaBins=12).T)
    HAp = getNormedFeatures(getHPCPEssentia(XAp, Fs, winSize, hopSize, NChromaBins=12).T)
    HB = getNormedFeatures(getHPCPEssentia(XB, Fs, winSize, hopSize, NChromaBins=12).T)
    #Compute OTI between A and B and shift A's HPCPs accordingly
    oti = getOTI(np.mean(HA, 0), np.mean(HB, 0))
    HA = np.roll(HA, oti, axis=1)
    #Compute normalized MFCC features for all three songs
    print("Getting MFCC features...")
    MA = getNormedFeatures(getMFCCsLibrosa(XA, Fs, winSize, hopSize, lifterexp = 0.6).T)
    MAp = getNormedFeatures(getMFCCsLibrosa(XAp, Fs, winSize, hopSize, lifterexp = 0.6).T)
    MB = getNormedFeatures(getMFCCsLibrosa(XA, Fs, winSize, hopSize, lifterexp = 0.6).T)
    print("HA.shape = %s, HAp.shape = %s, HB.shape = %s, MA.shape = %s, MAp.shape = %s, MB.shape = %s"%(HA.shape, HAp.shape, HB.shape, MA.shape, MAp.shape, MB.shape))
    N = min(HA.shape[0], MA.shape[0])
    print("N = %i"%N)
    HA = HA[0:N, :]
    HAp = HAp[0:N, :]
    MA = MA[0:N, :]
    MAp = MAp[0:N, :]
    N = min(HB.shape[0], MB.shape[0])
    HB = HB[0:N, :]
    MB = MB[0:N, :]

    #Step 4: Run analogies
    N = len(ws)
    XBp = XAp[0:hopSize*(N-1)+winSize]
    HBp = HAp[0:N, :]
    MBp = MAp[0:N, :]

    DABM = getCSM(MA, MB[0:N, :])
    DApBpM = getCSM(MAp, MBp)
    DABH = getCSMCosine(HA, HB[0:N, :])
    DApBpH = getCSMCosine(HAp, HBp)
    idxs = range(N)
    #Now add hopSize samples chunk by chunk
    for i in range(N, MB.shape[0]):
        #Computed weighted sliding window distances wrt every
        #window in the example pair
        D = MFCCWeight*np.sum(DABM*ws[None, :], 1) + \
            HPCPWeight*np.sum(DABH*ws[None, :], 1) + \
            MFCCWeight*np.sum(DApBpM*ws[None, :], 1) + \
            HPCPWeight*np.sum(DApBpH*ws[None, :], 1)
        #Pick out appropriate hopSize chunk from XAp
        idx = np.argmin(D)
        print("Synthesizing window %i of %i, min %i: %g"%(i, MB.shape[0], idx, D[idx]))
        idxs.append(idx)
        idx0 = (idx-1)*hopSize+winSize
        XBp = np.concatenate((XBp, XAp[idx0:idx0+hopSize]))
        #Update sliding window feature distances
        DABM = np.roll(DABM, -1, 1)
        row = MB[i, :]
        DABM[:, -1] = getCSM(MA, row[None, :]).flatten()
        DABH = np.roll(DABH, -1, 1)
        row = HB[i, :]
        DABH[:, -1] = getCSMCosine(HA, row[None, :]).flatten()
        DApBpM = np.roll(DApBpM, -1, 1)
        row = MAp[idx, :]
        DApBpM[:, -1] = getCSM(MAp, row[None, :]).flatten()
        DApBpH = np.roll(DApBpH, -1, 1)
        row = HAp[idx, :]
        DApBpH[:, -1] = getCSMCosine(HAp, row[None, :]).flatten()
        if i%200 == 0:
            sio.savemat("synth.mat", {"X":XBp, "Fs":Fs, "idxs":np.array(idxs)})





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